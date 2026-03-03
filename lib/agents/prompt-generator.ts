import { generateText } from "ai";
import { prisma } from "@/lib/prisma";
import { getModel, getProviderApiKey, resolveProvider } from "@/lib/ai/providers";
import { trackAIUsage } from "@/lib/ai/call";
import { generateTextZai } from "@/lib/ai/zai";
import { getCompactProjectContext } from "@/lib/agents/project-context";
import { loadProjectVibe } from "@/lib/vibe/parser";
import { createLLMAbortSignal, HEAVY_LLM_TIMEOUT_MS } from "@/lib/ai/timeout";

export const MAX_PROMPT_CHARS = 60000;

/**
 * Small helper that retries transient LLM/network failures a few times before giving up.
 *
 * Retries when the error message looks like a transport problem (fetch failed, timeouts, DNS, etc.).
 * For non-transient errors, it fails fast on the first attempt.
 */
export async function callWithRetries<T>(
  fn: () => Promise<T>,
  attempts: number = 3,
  delayMs: number = 2000
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);

      const transient =
        /fetch failed|ECONNRESET|ETIMEDOUT|timeout|ENOTFOUND|ECONNREFUSED/i.test(
          msg
        );

      const isLastAttempt = attempt === attempts;

      if (!transient || isLastAttempt) {
        throw err;
      }

      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error(String(lastError ?? "Unknown error"));
}

export function enforcePromptBudget(
  system: string,
  projectStateBlock: string,
  taskBlock: string,
  instructionsBlock: string
): {
  system: string;
  projectStateBlock: string;
  taskBlock: string;
  instructionsBlock: string;
} {
  const hardLimit = MAX_PROMPT_CHARS;

  // Start by ensuring system + instructions fit; shrink task + project as needed.
  const baseLen = system.length + taskBlock.length + instructionsBlock.length;

  let nextTaskBlock = taskBlock;
  let nextProjectStateBlock = projectStateBlock;
  let nextInstructionsBlock = instructionsBlock;

  if (baseLen >= hardLimit) {
    // Prefer to keep at least 10% of the budget for project state if possible.
    const reservedForProjectState = Math.floor(hardLimit * 0.1);
    const targetForNonProject = hardLimit - reservedForProjectState;

    const currentNonProjectLen = system.length + taskBlock.length + instructionsBlock.length;
    if (currentNonProjectLen > targetForNonProject) {
      const overflow = currentNonProjectLen - targetForNonProject;
      const shrinkAmount = Math.min(overflow, taskBlock.length);
      if (shrinkAmount > 0) {
        const keepLen = Math.max(taskBlock.length - shrinkAmount, 0);
        nextTaskBlock = taskBlock.slice(0, keepLen);
      }
    }
  }

  const usedWithoutProject =
    system.length + nextTaskBlock.length + nextInstructionsBlock.length;
  let remainingForProject = hardLimit - usedWithoutProject;

  if (remainingForProject <= 0) {
    // No room left for project state; drop it but inform the instructions so caller understands.
    nextProjectStateBlock = "";
    nextInstructionsBlock +=
      `\n\n[Note: Project state was truncated due to context limits.]`;
    return {
      system,
      projectStateBlock: nextProjectStateBlock,
      taskBlock: nextTaskBlock,
      instructionsBlock: nextInstructionsBlock,
    };
  }

  if (nextProjectStateBlock.length > remainingForProject) {
    // Try to drop low-priority sections (RAG / CodeMap / Knowledge) from the tail first.
    const markers = ["RAG", "Knowledge", "CodeMap", "Code Map"];
    let cutIndex = -1;
    for (const marker of markers) {
      const idx = nextProjectStateBlock.lastIndexOf(marker);
      if (idx !== -1) {
        cutIndex = cutIndex === -1 ? idx : Math.min(cutIndex, idx);
      }
    }

    if (cutIndex !== -1 && cutIndex >= 0) {
      nextProjectStateBlock = nextProjectStateBlock.slice(0, cutIndex);
    }

    if (nextProjectStateBlock.length > remainingForProject) {
      nextProjectStateBlock = nextProjectStateBlock.slice(0, remainingForProject);
    }
  }

  return {
    system,
    projectStateBlock: nextProjectStateBlock,
    taskBlock: nextTaskBlock,
    instructionsBlock: nextInstructionsBlock,
  };
}

export async function generateTaskPrompt(
  taskId: string,
  forceRegenerate: boolean = false,
  skipSave: boolean = false,
  extraRequirement?: string
): Promise<string> {
  const task = await prisma.task.findUnique({
    where: { id: taskId },
    include: {
      dependencies: {
        include: {
          dependsOn: {
            select: {
              id: true,
              title: true,
              status: true,
              comments: {
                where: { authorRole: "DEVOPS" },
                orderBy: { createdAt: "desc" },
                take: 1,
                select: { content: true },
              },
            },
          },
        },
      },
      plan: {
        include: {
          project: { include: { files: true } },
        },
      },
      attachments: true,
      comments: {
        where: { authorRole: "QA" },
        orderBy: { createdAt: "desc" },
        take: 10,
        select: { content: true, createdAt: true },
      },
    },
  });

  if (!task || !task.plan?.project) {
    throw new Error("Task not found");
  }

  // Return cached prompt unless forceRegenerate is true
  if (task.generatedPrompt && !forceRegenerate) {
    return task.generatedPrompt;
  }

  const project = task.plan.project;
  const plan = task.plan;

  const projectContext = await getCompactProjectContext(project.id);

  const globalContext = project.context?.trim() ?? "";

  const vibe = await loadProjectVibe(project.id);
  let vibeSection = "";

  if (vibe) {
    const architectureLines =
      vibe.architecture
        ? [
            "**Architecture:**",
            `- Preferred Pattern: ${vibe.architecture.preferred_pattern ?? "not specified"}`,
            `- Forbidden Patterns: ${
              vibe.architecture.forbidden_patterns?.join(", ") ?? "none specified"
            }`,
          ].join("\n")
        : "";

    const codeStyleLines =
      vibe.code_style
        ? [
            "**Code Style:**",
            `- Naming: ${vibe.code_style.naming ? JSON.stringify(vibe.code_style.naming) : "not specified"}`,
            `- Error Handling: ${vibe.code_style.error_handling ?? "not specified"}`,
          ].join("\n")
        : "";

    const testingLines =
      vibe.testing
        ? [
            "**Testing:**",
            `- Framework: ${vibe.testing.framework ?? "not specified"}`,
            `- Require For: ${
              vibe.testing.require_for && vibe.testing.require_for.length > 0
                ? vibe.testing.require_for.join(", ")
                : "not specified"
            }`,
          ].join("\n")
        : "";

    const qaLines =
      vibe.qa_rules
        ? [
            "**QA & Evidence Rules:**",
            `- Mandatory Evidence: ${
              vibe.qa_rules.mandatory_evidence && vibe.qa_rules.mandatory_evidence.length > 0
                ? vibe.qa_rules.mandatory_evidence.join(", ")
                : "not specified"
            }`,
            `- Strict Guidelines:${
              vibe.qa_rules.strict_guidelines && vibe.qa_rules.strict_guidelines.length > 0
                ? "\n- " + vibe.qa_rules.strict_guidelines.join("\n- ")
                : " not specified"
            }`,
          ].join("\n")
        : "";

    const sections = [architectureLines, codeStyleLines, testingLines, qaLines].filter(
      (s) => s && s.trim().length > 0
    );

    if (sections.length > 0) {
      vibeSection =
        "\n\n### PROJECT ENGINEERING STANDARDS (VIBE) ###\n" +
        "You MUST strictly adhere to the following project-specific guidelines:\n\n" +
        sections.join("\n\n");
    }
  }
  const contextInstruction = globalContext
    ? `Учитывай глобальный контекст проекта при генерации ответа: ${globalContext}\n\n`
    : "";

  const executorRole = task.executorAgent ?? "TEAMLEAD";

  const baseContextSystem =
    "### DYNAMIC CONTEXT AWARENESS\n" +
    "You are generating instructions for the CURRENT task.\n" +
    "Look at the **Project State** provided below and ADAPT your instructions accordingly:\n" +
    "1. **Check Completed Tasks**: What was just built? Does this task depend on it?\n" +
    "2. **Check Key Decisions (ADR)**: Did tech stack change? New libraries added?\n" +
    "3. **ADAPT Instructions**:\n" +
    "   - If Task 1 created `auth.ts` using Clerk, and Task 2 is \"Login Page\", explicitly tell executor to import from `auth.ts`, even if original plan didn't mention Clerk.\n" +
    "   - If a recent task switched to Tailwind CSS, make sure UI tasks use Tailwind classes.\n" +
    "   - If a task added TypeScript types, reference those types instead of `any`.\n" +
    "   - ALWAYS check the project state first, then generate context-aware instructions.\n\n";

  const commonTail =
    (contextInstruction ? contextInstruction : "") +
    (vibeSection ? `${vibeSection}\n\n` : "") +
    "Включи в промпт:\n" +
    "- Контекст проекта.\n" +
    `- Технический стек (${plan.techStack}).\n` +
    "- Четкие шаги реализации (Step-by-step).\n" +
    "- Какие файлы создать или изменить.\n" +
    "- Примерный код или структуру.\n\n" +
    "PROJECT STRUCTURE (STRICT): " +
    "When generating commands for scaffolding (e.g. Vite, Next.js, CRA, Django, Cargo), YOU MUST instruct the executor to use the current directory. NEVER instruct the executor to create a named subfolder. " +
    "FORBIDDEN: Do NOT generate scaffold commands with a project name (e.g. npm create vite my-app, npx create-next-app my-app, cargo new my-app). Only generate commands that target the current directory. " +
    "Examples (ONLY these forms): npm create vite@latest . -- --template react-ts; npx create-next-app@latest . --yes; django-admin startproject config .; cargo init . " +
    "When generating file paths for the next steps, always assume the project root is the current directory (e.g. src/App.tsx, package.json). Never use paths like my-app/src/App.tsx or todo-pwa/package.json. All commands run from project root; do not assume a nested project-name folder. " +
    "CRITICAL FALLBACK: If a framework forces the executor to scaffold into a named subfolder (e.g., creating a my-app directory instead of using .), instruct the executor to IMMEDIATELY move all generated files up to the current root using a command like mv my-app/* my-app/.* . 2>/dev/null || true && rm -rf my-app. The package.json MUST be located exactly in the project root before any other step.";

  const system =
    executorRole === "TASK_EXECUTOR"
      ? [
          "You are the TASK_EXECUTOR agent. Your goal is to generate precise, stepwise implementation instructions that you yourself will follow in a ReAct loop (Thought -> Action -> Observation) to complete this specific task.",
          "",
          baseContextSystem,
          "### REACT LOOP MENTAL MODEL\n",
          "- You will work in a loop of **Thought -> Action -> Observation**.\n",
          "- Never attempt to do everything at once. Each Action should be a small, concrete step (e.g., read a file, patch a function, run tests).\n",
          "- After each Action, you MUST wait for the Observation (tool result or error) and update your next Thought based on that.\n",
          "- If a command or tool call fails (Observation contains an error or non-zero exitCode), your very next Thought MUST explicitly analyze that error and propose how to fix it before taking further Actions.\n",
          "- Prefer using `getCodeMap`, `searchCodebase`, and `readFile` to understand the existing project before editing.\n",
          "- Use `writeFile` to create or overwrite files; for edits, read the file, modify the content, then write the full file with `writeFile` (or use executeCommand with a small script).\n",
          "- Never use placeholders like \"...\" or \"// rest of file\". All code you describe must be complete and production-ready.\n",
          "- Use `executeCommand` only for focused commands (tests, builds, linters). Never start long-running dev servers.\n",
          "- Use a logical tool `submitForQA` only when you have personally verified that the project builds and the code changes satisfy the task.\n",
          "",
          "Your generated instructions will guide this ReAct loop. Make them incremental, verifiable, and aligned with the tools listed above.\n",
          commonTail,
        ].join("")
      : [
          "Ты — Tech Lead. Твоя цель — написать идеальный промпт для разработчика (или для Cursor AI), чтобы он выполнил эту конкретную задачу.\n\n",
          baseContextSystem,
          commonTail,
        ].join("");

  const deps = task.dependencies ?? [];
  const incompleteDeps = deps.filter((d) => d.dependsOn.status !== "DONE");
  const doneDeps = deps.filter((d) => d.dependsOn.status === "DONE");
  const dependencyWarnings =
    incompleteDeps.length > 0
      ? "\n\nВНИМАНИЕ — зависимости:\n" +
        incompleteDeps
          .map(
            (d) =>
              `Эта задача зависит от выполнения задачи «${d.dependsOn.title}». Убедись, что код совместим с результатами той задачи.`
          )
          .join("\n")
      : "";
  const dependencyContext =
    doneDeps.length > 0
      ? "\n\nУчитывай результаты уже выполненных задач: " +
        doneDeps.map((d) => `«${d.dependsOn.title}»`).join(", ")
      : "";

  // Cap each dependency report to avoid blowing prompt budget (taskBlock is trimmed by enforcePromptBudget).
  const MAX_DEPENDENCY_REPORT_CHARS = 10_000;
  let dependencyReportsBlock = "";
  if (doneDeps.length > 0) {
    const sections: string[] = ["\n\n=== ОТЧЕТЫ ПО ЗАВЕРШЕННЫМ ЗАВИСИМОСТЯМ ==="];
    for (const d of doneDeps) {
      const dep = d.dependsOn as { id: string; title: string; status: string; comments?: { content: string }[] };
      const comments = dep.comments ?? [];
      const reportContent = comments[0]?.content?.trim();
      if (reportContent) {
        const capped =
          reportContent.length > MAX_DEPENDENCY_REPORT_CHARS
            ? reportContent.slice(0, MAX_DEPENDENCY_REPORT_CHARS) +
              "\n\n[Dependency report truncated due to length limit.]"
            : reportContent;
        sections.push(`Задача: ${dep.title}\nОтчет о реализации:\n${capped}\n`);
      }
    }
    if (sections.length > 1) {
      dependencyReportsBlock = sections.join("\n");
    }
  }

  let cssFeedbackBlock = "";
  const taskComments = (task as { comments?: { content: string }[] }).comments ?? [];
  const cssNeedsImprovement = taskComments.find(
    (c) =>
      c.content.includes("CSS Review") &&
      c.content.includes("Status: NEEDS_IMPROVEMENT")
  );
  if (cssNeedsImprovement?.content) {
    const suggestionsMatch = cssNeedsImprovement.content.match(/Suggestions:\s*([\s\S]*?)(?=\n\n|$)/i);
    const suggestions = suggestionsMatch
      ? suggestionsMatch[1].trim().split(/\n/).map((s) => s.trim()).filter(Boolean)
      : [];
    const reasoningMatch = cssNeedsImprovement.content.match(/Reasoning:\s*([\s\S]*?)(?=Suggestions:|$)/i);
    const reasoning = reasoningMatch ? reasoningMatch[1].trim().slice(0, 500) : "";
    if (suggestions.length > 0 || reasoning) {
      cssFeedbackBlock =
        "\n\n=== PREVIOUS CSS FEEDBACK (apply these fixes) ===\n" +
        (reasoning ? `Reasoning: ${reasoning}\n` : "") +
        (suggestions.length > 0 ? "Suggestions:\n" + suggestions.map((s) => `- ${s}`).join("\n") : "");
    }
  }

  const visionContexts = (task.attachments ?? [])
    .filter((a) => a.visionAnalysis && a.visionAnalysis.trim().length > 0)
    .map(
      (a) =>
        `=== DESIGN IMAGE: ${a.fileName} ===\n` +
        `Image URL: ${a.filePath}\n\n` +
        `AI Vision Analysis:\n${a.visionAnalysis}\n`
    )
    .join("\n");

  const visionInstruction =
    visionContexts.length > 0
      ? `\n\n=== DESIGN CONTEXT FROM IMAGES ===\n` +
        visionContexts +
        `\nIMPORTANT: The AI Vision Analysis above describes the design. Use these details to implement the UI/layout correctly.`
      : "";

  const vc = (task as any).verificationCriteria as
    | {
        artifacts?: string[];
        automatedCheck?: string;
        manualCheck?: string;
      }
    | null
    | undefined;

  const verificationBlock = vc
    ? [
        "",
        "=== ACCEPTANCE CRITERIA ===",
        "The following will be verified after you finish:",
        ...(Array.isArray(vc.artifacts) && vc.artifacts.length
          ? [
              "Required files (MUST exist at exact paths):",
              ...vc.artifacts.map((a) => `  - ${a}`),
            ]
          : []),
        ...(vc.automatedCheck
          ? [
              `Automated check command: ${vc.automatedCheck}`,
              "IMPORTANT: Ensure your implementation satisfies this command.",
            ]
          : []),
        ...(vc.manualCheck ? [`Manual check: ${vc.manualCheck}`] : []),
        "===========================",
        "",
      ].join("\n")
    : "";

  const projectStateBlock = `=== PROJECT STATE ===\n${projectContext}\n\n`;
  const taskBlock =
    `=== CURRENT TASK ===\n` +
    `Задача: ${task.title}\n` +
    `Описание задачи: ${task.description}\n` +
    verificationBlock +
    `Роль исполнителя: ${task.executorAgent ?? "TEAMLEAD"}\n\n` +
    dependencyWarnings +
    dependencyContext +
    dependencyReportsBlock +
    cssFeedbackBlock +
    visionInstruction;

  let instructionsBlock =
    `\n\n=== INSTRUCTIONS ===\n` +
    `Generate detailed coding instructions for the task above.\n` +
    `IMPORTANT: Consider the Project State above and adapt instructions based on:\n` +
    `1. What was already built in completed tasks\n` +
    `2. Any tech stack changes or additions from ADRs\n` +
    `3. Dependencies on previous tasks\n` +
    `4. Design details from Vision Analysis (if provided above)`;

  if (extraRequirement && extraRequirement.trim()) {
    instructionsBlock +=
      `\n\n=== NEW REQUIREMENT (TICKET) ===\n` +
      `${extraRequirement.trim()}\n\n` +
      `Regenerate or adapt the instructions for this task so that this new requirement is fully satisfied, while preserving the original intent of the task.`;
  }

  const budgeted = enforcePromptBudget(
    system,
    projectStateBlock,
    taskBlock,
    instructionsBlock
  );
  const prompt =
    budgeted.projectStateBlock + budgeted.taskBlock + budgeted.instructionsBlock;

  const resolvedProvider = resolveProvider(
    project.aiProvider ?? process.env.AI_PROVIDER
  );
  const resolvedModel =
    project.aiModel ??
    process.env.AI_MODEL ??
    (resolvedProvider === "anthropic"
      ? "claude-3-5-sonnet-latest"
      : resolvedProvider === "zai"
        ? "glm-4.7"
        : "gpt-4o-mini");

  const apiKey = getProviderApiKey(resolvedProvider);
  if (!apiKey) {
    throw new Error(`Missing API key for provider: ${resolvedProvider}`);
  }

  let aiResult: any = null;
  let resultText: string;

  if (resolvedProvider === "zai") {
    resultText = await callWithRetries(() =>
      generateTextZai({
        systemPrompt: system,
        userMessage: prompt,
        model: resolvedModel,
        temperature: 0.3,
        maxTokens: 8000,
        signal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
      })
    );
  } else {
    aiResult = await callWithRetries(() =>
      generateText({
        model: getModel(resolvedProvider, resolvedModel),
        system,
        prompt,
        temperature: 0.3,
        maxTokens: 8000,
        abortSignal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
      })
    );
    resultText = aiResult.text ?? "";
  }

  if (aiResult) {
    await trackAIUsage(aiResult, { projectId: project.id, taskId: task.id, actionType: "prompt_gen", model: resolvedModel });
  }

  const generatedPrompt = resultText.trim();

  if (!skipSave) {
    await prisma.task.update({
      where: { id: task.id },
      data: { generatedPrompt },
    });
  }

  return generatedPrompt;
}

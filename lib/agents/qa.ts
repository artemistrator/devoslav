import { generateText } from "ai";
import { z } from "zod";
import { prisma } from "@/lib/prisma";
import { getModel, resolveProvider } from "@/lib/ai/providers";
import { createAgentTools } from "./tools";
import { trackAIUsage } from "@/lib/ai/call";
import { analyzeRejectReason } from "./debug";
import { getCompactProjectContext } from "./project-context";
import { getProjectDir } from "@/lib/project-workspace";
import { detectStack } from "@/lib/utils/stack-detection";
import { logQA } from "@/lib/qa-logger";
import { getLLMSettings } from "@/lib/settings";
import { createLLMAbortSignal } from "@/lib/ai/timeout";

async function addSystemComment(taskId: string, content: string, emoji: string = "🔍") {
  try {
    await prisma.comment.create({
      data: {
        taskId,
        content: `${emoji} ${content}`,
        authorRole: "QA",
        isSystem: true,
      },
    });
  } catch (error) {
    console.error("[QA] Failed to add system comment:", error);
  }
}

const qaVerificationSchema = z.object({
  status: z.enum(["APPROVED", "REJECTED"]),
  reasoning: z.string(),
  confidence: z.number().min(0).max(1),
  rejectionSummary: z.string().optional(),
});

export async function verifyTaskCompletion(taskId: string, reportContent: string) {
  logQA(taskId, 'START', 'Starting QA verification', {
    reportLength: reportContent.length,
    reportPreview: reportContent.slice(0, 200)
  });

  await addSystemComment(taskId, '🔍 Начинаю QA проверку выполнения задачи...');

  const task = await prisma.task.findUnique({
    where: { id: taskId },
    include: {
      plan: {
        include: {
          project: true,
        },
      },
      comments: {
        orderBy: { createdAt: "desc" },
        take: 5,
      },
    },
  }) as any;

  if (!task || !task.plan?.project) {
    logQA(taskId, 'ERROR', 'Task not found');
    throw new Error("Task not found");
  }

  const project = task.plan.project;
  const plan = task.plan;

  logQA(taskId, 'TASK_INFO', 'Task loaded', {
    title: task.title,
    executorAgent: task.executorAgent,
    status: task.status
  });

  await addSystemComment(taskId, '📋 Загружена задача: ' + task.title);

  const hasTestLogs = reportContent.toLowerCase().includes("test") || 
                      reportContent.toLowerCase().includes("pass") ||
                      reportContent.toLowerCase().includes("✓") ||
                      reportContent.toLowerCase().includes("✅");

  const hasBuildLogs = reportContent.toLowerCase().includes("build") ||
                       reportContent.toLowerCase().includes("compiled") ||
                       reportContent.toLowerCase().includes("npm run") ||
                       reportContent.toLowerCase().includes("npm run build");

  const recentComments = task.comments.map((c: any) => c.content).join("\n---\n");

  const vc = task.verificationCriteria as { artifacts?: string[]; manualCheck?: string; automatedCheck?: string } | null;

  logQA(taskId, 'VC_LOADED', 'Verification criteria loaded', vc);

  if (vc) {
    const criteriaSummary = [];
    if (vc.artifacts?.length) criteriaSummary.push(`📁 Файлы: ${vc.artifacts.join(', ')}`);
    if (vc.automatedCheck) criteriaSummary.push('🤖 Авто-проверка: ' + vc.automatedCheck);
    if (vc.manualCheck) criteriaSummary.push('👤 Ручная проверка: ' + vc.manualCheck);
    await addSystemComment(taskId, criteriaSummary.join('\n'), '✅');
  }

  const projectStateContext = await getCompactProjectContext(project.id);
  logQA(taskId, 'CONTEXT', 'Project context loaded', {
    contextLength: projectStateContext.length,
    contextPreview: projectStateContext.slice(0, 200)
  });

  let stack: { type: string; buildCommand: string | null } = { type: "unknown", buildCommand: null };
  try {
    const projectDir = getProjectDir(project.id);
    stack = await detectStack(projectDir);
  } catch {
    // project dir missing or readdir failed; keep unknown
  }

  const stackPromptBlock =
    `### STACK DETECTION / PROJECT STACK\n` +
    `This project's detected stack: **${stack.type}**. ` +
    `Typical build/check command for this stack: ${stack.buildCommand ?? "none (unknown stack)"}. ` +
    `When looking for automatedCheck evidence, accept success output appropriate for this stack (e.g. Node: npm/yarn output, Rust: cargo build/test, Go: go build/test, Python: test runner or py_compile, Java: mvn compile/test).\n\n`;

  const prompt = `You are a strict QA Specialist following GSD Goal-Backward Verification methodology.

${projectStateContext}

=== TASK INFORMATION ===
Task Title: ${task.title}
Description: ${task.description}
Executor Role: ${task.executorAgent || "Not specified"}

Project Plan: ${plan.title}
Tech Stack: ${plan.techStack}

=== VERIFICATION CRITERIA (from Architect) ===
${vc ? `${vc.artifacts && vc.artifacts.length > 0 ? `[Artifacts]\nExpected files:\n${vc.artifacts.map(a => `  - ${a}`).join('\n')}\n` : ''}
${vc.manualCheck ? `[manualCheck]\n${vc.manualCheck}\n` : ''}
${vc.automatedCheck ? `[automatedCheck]\n${vc.automatedCheck}\n` : ''}` : ''}
=== END VERIFICATION CRITERIA ===

=== EXECUTOR REPORT (evidence) ===
${reportContent}
=== END EXECUTOR REPORT ===

${recentComments ? `=== PREVIOUS COMMENTS ===\n${recentComments}\n=== END PREVIOUS COMMENTS ===\n\n` : ""}

=== YOUR MISSION ===
Follow the GSD Verification Protocol to decide if this task is COMPLETE.

**PRIORITY ORDER (apply this order when weighing evidence):**
1. **Build passes** — This is the primary success signal. It outweighs missing individual files.
2. **Core functionality is present** — The intended output exists (e.g. in dist/, build artifacts, generated files).
3. **Artifact checklist** — Treat as secondary. If a listed file is absent but the build passes and the feature works, the file may be obsolete (newer toolchain or alternative approach). APPROVE and note the discrepancy in your reasoning instead of rejecting.
4. **Never reject solely** because a source or config file is absent when the build succeeds and the intended output is present.

**STEP 1**: Read the Verification Criteria above.
**STEP 2**: Search the Executor Report for evidence for EACH criteria point:
   - [Artifacts]: Look for ls -la output, file content, or file creation logs
   - [automatedCheck]: Look for test/build success logs, PASS markers
   - [manualCheck]: Look for confirmation text or screenshot descriptions
**STEP 3**: Check evidence against the PRIORITY ORDER above. If build passes and functionality is evidenced, do not require every artifact file to be present.
**STEP 4**: Make your decision.

**CRITICAL RULE**: If build fails or automatedCheck has no success evidence — REJECT with clear evidence needed. If build passes and core functionality is evidenced, do NOT REJECT only for a missing artifact file; APPROVE and note the discrepancy (e.g. "Artifact X not present; build passed and functionality evidenced; approved with note.").

**HEADLESS EXCEPTION**: Executor runs in Docker/headless (no browser). If manualCheck only asks for "open in browser", "screenshot", or "visual confirmation", and Artifacts + automatedCheck both have evidence, APPROVE and state that manual check is deferred to user.

**TICKET RUN EXCEPTION**: If the report contains "[Ticket run]" and states that "automatedCheck from the original task was skipped" (implementation was modified by the ticket), do NOT require automatedCheck evidence. APPROVE when: (1) Artifacts have evidence (files exist), and (2) the report describes the implementation (e.g. file content or tool outputs). The original automatedCheck may fail on modified output (e.g. HTML with new tags splitting text); that is expected for ticket runs.

Only APPROVE when:
✓ automatedCheck (build/test) shows success (PASS, Build success, no errors) — this is the primary signal
✓ Core functionality or intended output is evidenced (build artifacts, generated files, or manualCheck confirmation)
✓ Artifacts: either all are mentioned/shown, OR build passes and functionality works (then approve with note for any missing artifact)
✓ No visible syntax/logic errors in code

Evidence examples you should ACCEPT:
- "ls -la" output showing the file exists
- Full file content in code blocks
- "PASS" or "✓" in test output
- "Build success" or "Compiled successfully" messages (or build/compile success for the project stack)
- Success output from the task's automatedCheck (any stack)
- "I verified X and it works" text

Evidence examples you should REJECT:
- No mention of required files at all
- Test output with FAIL or errors
- Build output with compilation errors
- No confirmation for manual checks

ВАЖНО: Ответь ТОЛЬКО валидным JSON в следующем формате:
{
  "status": "APPROVED" или "REJECTED",
  "reasoning": "For APPROVED: List each criteria and the evidence found. For REJECTED: Start with 'Verification Failed. Missing evidence for:' and list missing criteria with specific evidence needed.",
  "confidence": число от 0.0 до 1.0,
  "rejectionSummary"?: "Краткое одно предложение, объясняющее корневую причину отказа (ТОЛЬКО если status = \"REJECTED\")"
}`;

  const resolvedProvider = resolveProvider(project.aiProvider ?? undefined);
  const resolvedModel = project.aiModel ?? "gpt-4o-mini";
  const model = getModel(resolvedProvider, resolvedModel);
  const modelDisplay = project.aiModel ?? `${resolvedProvider}/${resolvedModel}`;

  logQA(taskId, 'AI_CALL', 'Calling AI for QA verification', {
    model: modelDisplay,
    reportLength: reportContent.length,
    hasVC: !!vc
  });

  await addSystemComment(taskId, '🤖 Анализирую отчет с помощью AI (' + modelDisplay + ')...');

  const tools = createAgentTools(
    project.id,
    undefined,
    "local",
    (level, message) => {
      logQA(taskId, "TOOLS", message, { level });
    }
  );
  const llmSettings = await getLLMSettings();
  const qaTemperature = Math.min(llmSettings.temperature, 0.1);

  const system = `You are a strict QA Specialist following the GSD (Goal-Backward Verification) methodology.
Your job is to verify task completion based on the report provided by the Executor Agent (User/Cursor).

### CRITICAL RUNTIME CONTEXT
- **You are running inside a Docker Container.**
- **The Code is on the User's Host Machine.**
- **YOU DO NOT HAVE DIRECT ACCESS TO THE FILE SYSTEM.**
- **YOU CANNOT ACCESS LOCALHOST (127.0.0.1) OF THE USER.**

${stackPromptBlock}

### UNIVERSAL CHECKS
Apply these regardless of stack:
1. Build/compile passes for the detected stack.
2. Required files exist per task (artifacts).
3. Main functionality implemented.
4. No obvious bugs.

### STACK-SPECIFIC (use detected stack)
- **Frontend** (e.g. Node/React/Vue): UI components render, styles work (styling config if applicable).
- **Backend**: API responds, DB connects if applicable.
- **CLI**: commands run successfully.

### GSD VERIFICATION PROTOCOL (STRICT)

Follow this exact algorithm:

#### STEP 1: READ VERIFICATION CRITERIA
Extract the verificationCriteria from the task:
- **Artifacts**: File paths that must exist
- **automatedCheck**: Commands that must succeed (tests, builds)
- **manualCheck**: What must be verified manually

#### STEP 2: COLLECT EVIDENCE FROM REPORT
For each criteria point, search for evidence:

**A. For Artifacts:**
- For each artifact listed in [Artifacts], the implementation report MUST contain explicit proof of existence: either ls -la output showing the file, or the file content itself, or a build log referencing it.
- A verbal statement "file was created" without proof is NOT sufficient evidence.
- If proof is missing for any artifact — REJECT.
- Look for \`ls -la\` output showing the file exists
- Look for full file content in the report (code blocks, \`\`\`filename\`\`\`)
- Look for \`cat\` or \`cat <filename>\` commands with file content
- Look for build tool output that references the filename (e.g. "compiled index.ts")

**B. For automatedCheck:**
- Look for test execution logs (e.g., "PASS", "✓", "All tests passed")
- Look for build success messages (e.g., "Build success", "Compiled successfully")
- Look for command output from the specified automatedCheck command
- Look for successful execution of the automatedCheck command (e.g. npm/yarn for Node, cargo for Rust, go for Go, mvn for Java, pytest/py_compile for Python)

**C. For manualCheck:**
- Look for textual confirmation ("I verified X", "Y works as expected")
- Look for screenshot descriptions or base64 image data
- Look for UI behavior descriptions matching the criteria

### EVIDENCE RULES
Valid evidence types (in order of strength):
- ls -la output containing the filename — valid
- cat / file content shown in report — valid
- build tool output referencing filename (e.g. "compiled index.ts") — valid
- verbal statement without any of the above — not valid

This applies universally regardless of project type or tech stack.

#### STEP 3: VALIDATE EVIDENCE (TRUTH CHECK)
For each criteria point, decide if the evidence is sufficient:
- **Artifacts**: Sufficient only if the report contains filesystem proof per EVIDENCE RULES (ls -la output, file content, or build output referencing the file). Verbal "file was created" or "written successfully" alone is NOT sufficient.
- **automatedCheck**: Sufficient if logs show successful execution OR no errors in output
- **manualCheck**: Sufficient if user explicitly confirms verification OR describes expected behavior

#### STEP 4: MAKE DECISION (BUILD-FIRST PRIORITY)
**Priority rule:** If **automatedCheck** (build/test) shows success AND **core functionality** is evidenced (e.g. build artifacts, generated output, or manualCheck confirmed), do NOT REJECT only for missing artifact files. APPROVE and in reasoning note: "Artifact X was not present; build passed and functionality evidenced; approved with note."

**APPROVE (status: "APPROVED"):**
- ALL artifacts from verificationCriteria have filesystem proof in the report (ls -la / file content / build output referencing the file by name).
- automatedCheck has success evidence (build/test passed) AND (all artifacts have evidence OR build passed and functionality is evidenced)
- No visible syntax/logic errors in provided code
- No errors in logs

**REJECT (status: "REJECTED"):**
- Any artifact from verificationCriteria lacks filesystem proof in the report.
- automatedCheck has NO success evidence (build/test failed or no output), OR
- Evidence contradicts criteria, OR
- Visible syntax/logic errors, OR
- Report contains no code/logs at all
- **Exception:** Do NOT REJECT solely for missing artifact file(s) when build passes and intended output/functionality is present.

#### STEP 5: GENERATE REASONING + SHORT SUMMARY
**If APPROVED:**
- List each criteria point and the evidence found
- Example: "✓ Artifacts: Found src/app/api/auth/route.ts in ls output. ✓ automatedCheck: automatedCheck command shows success (e.g. PASS, Build success, exit 0)."

**If REJECTED:**
- Start with: "Verification Failed. Missing evidence for:"
- List ALL missing criteria points
- For each missing point, specify what evidence is needed
- Example: "Verification Failed. Missing evidence for: [Artifacts] Please provide ls -la output or file content showing src/app/api/auth/route.ts exists. [automatedCheck] Please provide automatedCheck command output showing success (e.g. test/build output for the project stack)."
- In addition, you MUST fill the field \"rejectionSummary\":
  - It MUST be one short sentence (no more than ~120 characters).
  - It MUST capture the CORE reason for failure in plain language.
  - Examples:
    - "Build failed due to missing dependency."
    - "Required artifact package.json not found."
    - "Automated test command returned non-zero exit code."
    - "No evidence provided for manualCheck browser verification."

### CODE REVIEW CHECKLIST (when evidence exists)
When reviewing provided code snippets, check for:
- Hardcoded secrets, API keys, or sensitive data
- Missing type annotations (any in TypeScript)
- Logic errors
- Security vulnerabilities
- Missing error handling
- Incorrect imports

### EXTENSION FLEXIBILITY
- When verifying files in verificationCriteria.artifacts, treat \`.js\`, \`.ts\`, \`.jsx\`, and \`.tsx\` interchangeably. If the criteria asks for \`vite.config.js\` but the evidence shows \`vite.config.ts\`, consider it a PASS.

### BUILD-FIRST PRIORITY
When weighing evidence, apply this order: (1) Build passes = primary; (2) Core functionality present = intended output exists; (3) Artifact checklist = secondary. Approve only if filesystem proof (ls -la or file content or build output) is present for artifacts; if a listed file has no such proof, do not approve.

### IMPORTANT RULES
- **NEVER REJECT** because "File not found" or "Cannot connect to localhost" — THIS IS EXPECTED.
- NEVER REJECT solely because a source or config file from the artifact list is absent when the build succeeds and the intended output is present.
- Use GSD terminology: "Artifacts", "Truths" (evidence that proves criteria).

### ENVIRONMENT LIMITATIONS (approve when artifact + manual evidence exist)
- If **Artifacts** have filesystem proof (ls -la or file content or build output is present in the report) AND **manualCheck** is confirmed in the report (e.g. "отображается текст", "I verified"), but **automatedCheck** failed only because a shell command is missing in the environment (e.g. "file: command not found", "command not found", "not found"), then **APPROVE**.
- In the reasoning, state: artifact and manual check satisfied; automatedCheck failed due to environment (missing command), not due to task implementation.

### HEADLESS / DOCKER: manualCheck requiring browser or screenshot
- **Executor runs in a container/headless environment. It CANNOT open a real browser, take screenshots, or perform visual UI checks.**
- If **Artifacts** and **automatedCheck** BOTH have sufficient evidence in the report (files exist, automated command succeeded), and **manualCheck** ONLY requires "open in browser", "screenshot", "visual confirmation", or "displayed on page", then **APPROVE**.
- In the reasoning, state: Artifacts and automatedCheck satisfied; manualCheck requires browser/visual verification which is not available in this environment — deferred to user. Do NOT REJECT solely because the report lacks screenshot or "I opened in browser" text.

### TICKET RUN: automatedCheck skipped
- If the report contains **"[Ticket run]"** and states that **"automatedCheck from the original task was skipped"** (implementation was modified by the ticket), do NOT require automatedCheck evidence.
- APPROVE when: (1) **Artifacts** have evidence (files exist, ls -la or content), and (2) the report describes the implementation (tool outputs, file content). The original task's automatedCheck (e.g. grep for exact phrase) may fail when output was changed by the ticket (e.g. HTML with spans); that is expected.
- In the reasoning, state: Ticket run; automatedCheck skipped per ticket-run rule. Artifacts satisfied; evidence from report.

ВАЖНО: Ответь ТОЛЬКО валидным JSON в следующем формате:
{
  "status": "APPROVED" или "REJECTED",
  "reasoning": "подробное объяснение решения с указанием доказательств или отсутствия доказательств по каждому критерию",
  "confidence": число от 0.0 до 1.0,
  "rejectionSummary"?: "Краткое одно предложение, объясняющее корневую причину отказа (ТОЛЬКО если status = \"REJECTED\")"
}`;

  const LOW_CONFIDENCE_THRESHOLD = 0.7;

  const extractJsonFromText = (raw: string): string | null => {
    if (!raw) return null;
    const trimmed = raw.trim();

    // 1) Попробуем вытащить JSON из ```json ``` блока
    const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fencedMatch && fencedMatch[1]) {
      return fencedMatch[1].trim();
    }

    // 2) Если блока нет — ищем первый и последний фигурные скобки и берём срез
    const firstBrace = trimmed.indexOf("{");
    const lastBrace = trimmed.lastIndexOf("}");
    if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
      return trimmed.slice(firstBrace, lastBrace + 1).trim();
    }

    return null;
  };

  const runOneQA = async (promptOverride?: string) => {
    const res = await generateText({
      model,
      system,
      prompt: promptOverride ?? prompt,
      tools,
      maxSteps: 5,
      maxTokens: Math.min(llmSettings.maxTokens ?? 16384, 16384),
      temperature: qaTemperature,
      abortSignal: createLLMAbortSignal(),
    });
    await trackAIUsage(res, { projectId: project.id, taskId, actionType: "qa_check", model: modelDisplay });

    const rawText = res.text ?? "";
    logQA(taskId, "RAW_RESPONSE", "Raw QA model response captured", {
      length: rawText.length,
      preview: rawText.slice(0, 300),
    });

    const extracted = extractJsonFromText(rawText);
    if (!extracted) {
      throw new Error("QA response parsing failed: no JSON-like content detected");
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(extracted);
    } catch (e) {
      throw new Error(
        `QA response parsing failed: JSON.parse error: ${
          e instanceof Error ? e.message : String(e)
        }`
      );
    }

    const p = qaVerificationSchema.safeParse(parsed);
    if (!p.success) {
      throw new Error("QA response parsing failed: schema validation error");
    }
    return p.data;
  };

  const PARSE_RETRY_MESSAGE =
    "\n\nSystem: Your previous response was invalid or truncated JSON. Please return ONLY a valid JSON object with keys: status (APPROVED or REJECTED), reasoning, confidence (0-1), and optionally rejectionSummary.";

  let parsedData: z.infer<typeof qaVerificationSchema> | undefined;
  try {
    let promptForAttempt = prompt;
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        parsedData = await runOneQA(promptForAttempt);
        break;
      } catch (e) {
        if (attempt === 3) throw e;
        logQA(taskId, "PARSE_RETRY", `Attempt ${attempt} failed, retrying`, {
          error: e instanceof Error ? e.message : String(e),
        });
        promptForAttempt = promptForAttempt + PARSE_RETRY_MESSAGE;
      }
    }
    if (!parsedData) throw new Error("QA parse failed");
    if (parsedData.status === "REJECTED" && parsedData.confidence < LOW_CONFIDENCE_THRESHOLD) {
      logQA(taskId, "RETRY", "Low-confidence REJECTED, requesting second opinion", {
        confidence: parsedData.confidence,
        threshold: LOW_CONFIDENCE_THRESHOLD,
      });
      await addSystemComment(taskId, "🔄 Низкая уверенность при отклонении — повторная проверка...");
      try {
        const second = await runOneQA(
          prompt + "\n\n[SECOND OPINION] Re-evaluate. If evidence is borderline or partially present for the criteria, prefer APPROVED."
        );
        if (second.status === "APPROVED") {
          parsedData = second;
          await addSystemComment(
            taskId,
            "✅ После повторной проверки: решение изменено на APPROVED.\n\n" + second.reasoning
          );
        }
      } catch (e) {
        if (process.env.NODE_ENV !== "production") {
          console.error("[QA] Second opinion failed:", e);
        }
      }
    }
  } catch (parseErr) {
    const errMsg = parseErr instanceof Error ? parseErr.message : String(parseErr);
    logQA(taskId, "ERROR", "QA response was unparseable", { error: errMsg });
    await addSystemComment(
      taskId,
      "System Error: QA Agent provided an unparseable response. Retrying via ticket.",
      "⚠️"
    );

    await prisma.task.update({
      where: { id: taskId },
      data: { status: "REJECTED" as any },
    });

    return {
      taskId,
      status: "REJECTED" as const,
      finalStatus: "REJECTED" as const,
      reasoning: "System Error: QA Agent provided an unparseable response. Retrying via ticket.",
      confidence: 0,
      taskTitle: task.title,
      debugSummary: null,
      rejectionSummary: null,
    };
  }

  const { status, reasoning, confidence, rejectionSummary } = parsedData;

  const computedFinalStatus =
    status === "APPROVED"
      ? project.requireApproval
        ? "WAITING_APPROVAL"
        : "DONE"
      : "REJECTED";

  logQA(taskId, 'DECISION', 'QA decision made', {
    status,
    finalStatus: computedFinalStatus,
    confidence,
    reasoningPreview: reasoning.slice(0, 300)
  });

  let finalStatus: string =
    status === "APPROVED"
      ? project.requireApproval
        ? "WAITING_APPROVAL"
        : "DONE"
      : "REJECTED";
  if (status === "APPROVED") {
    await addSystemComment(taskId, '✅ QA ПРОВЕРКА ПРОЙДЕНА!\n\n' + reasoning, '🎉');
  } else {
    await addSystemComment(taskId, '❌ QA ПРОВЕРКА НЕ ПРОЙДЕНА!\n\n' + reasoning, '🚫');
  }

  await prisma.task.update({
    where: { id: taskId },
    data: { status: finalStatus as any },
  });

  let debugSummary = null;
  if (status === "REJECTED") {
    try {
      debugSummary = await analyzeRejectReason({
        projectId: project.id,
        taskId,
        qaReasoning: reasoning,
        verificationCriteria: vc,
        executorReport: reportContent,
      });

      const debugSummaryText = `<symptoms>
expected: ${debugSummary.symptoms.expected}
actual: ${debugSummary.symptoms.actual}
missing_evidence: ${debugSummary.symptoms.missing_evidence}
</symptoms>`;

      await prisma.comment.create({
        data: {
          taskId,
          content: `🐛 Debug Summary (GSD)\n\n${debugSummaryText}`,
          authorRole: "QA",
        },
      });
    } catch (error) {
      if (process.env.NODE_ENV !== "production") {
        console.error("[QA] Failed to generate debug summary:", error);
      }
    }
  }

  return {
    taskId,
    status,
    finalStatus,
    reasoning,
    confidence,
    taskTitle: task.title,
    debugSummary,
    rejectionSummary: rejectionSummary ?? null,
  };
}

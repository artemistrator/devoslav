import { promises as fs } from "fs";
import { BaseAgent, AgentConfig } from "./base-agent";
import { AgentMessage, AgentRole, MessageType } from "@prisma/client";
import { createAgentTools } from "./tools";
import { createAgentForRole } from "@/lib/execution/agent-factory";
import { generateText, type CoreMessage } from "ai";
import { getModel, resolveProvider } from "@/lib/ai/providers";
import { prisma } from "@/lib/prisma";
import { getCompactProjectContext } from "./project-context";
import { getProjectDir } from "@/lib/project-workspace";
import { CONTAINER_WORKSPACE_ROOT } from "@/lib/execution/container-manager";
import { detectStack, isWebProject } from "@/lib/utils/stack-detection";
import { trackAIUsage } from "@/lib/ai/call";
import { generateTaskPrompt, callWithRetries } from "./prompt-generator";
import { getLLMSettings } from "@/lib/settings";
import { createLLMAbortSignal, HEAVY_LLM_TIMEOUT_MS } from "@/lib/ai/timeout";
import {
  FIND_MANIFEST_CMD,
  MANIFEST_AT_ROOT_CHECK_CMD,
  resolveArtifactBaseFromFindResult,
  getVerificationPath,
  allArtifactsArePathLike,
  isArtifactBaseInBuildDir,
} from "./verification-paths";

/** Normalize command string for comparison (e.g. automatedCheck vs stack.buildCommand). */
function normalizeBuildCommand(cmd: string): string {
  return cmd.replace(/\s+/g, " ").trim();
}

function isHardGateInfrastructureError(message: string): boolean {
  const m = (message || "").toLowerCase();
  return (
    m.includes("docker exec") ||
    m.includes("no such container") ||
    m.includes("etimedout") ||
    m.includes("econnreset") ||
    m.includes("command failed: docker")
  );
}

const HARD_GATE_STOP_HEADER =
  "STOP! Your code failed the Hard Gate (build/compile check). You cannot submit for QA until you fix these errors.\n\n";

async function getFallbackHardGateCommand(
  stackType: string,
  projectDir: string
): Promise<string | null> {
  try {
    const files = await fs.readdir(projectDir);

    const hasRequirements = files.includes("requirements.txt");
    const hasPyproject = files.includes("pyproject.toml");
    const hasMainPy = files.includes("main.py");

    switch (stackType) {
      case "nodejs": {
        if (files.includes("tsconfig.json")) {
          return "$(which npx 2>/dev/null || echo npx) tsc --noEmit";
        }
        return null;
      }
      case "python": {
        if (hasRequirements || hasPyproject || hasMainPy) {
          return "$(which python3 2>/dev/null || which python 2>/dev/null || echo python3) -m compileall -q .";
        }
        return null;
      }
      case "static": {
        const hasAnyManifest =
          files.includes("package.json") ||
          files.includes("Cargo.toml") ||
          files.includes("go.mod") ||
          hasRequirements ||
          hasPyproject ||
          files.includes("pom.xml");
        if (!hasAnyManifest && files.includes("index.html")) {
          return "$(which npx 2>/dev/null || echo npx) htmlhint *.html";
        }
        return null;
      }
      default:
        return null;
    }
  } catch {
    return null;
  }
}

type ReActHistorySegmentType = "task" | "thought" | "action" | "observation";

interface ReActHistorySegment {
  type: ReActHistorySegmentType;
  text: string;
}

type ReActAction =
  | {
      toolName: string;
      params?: Record<string, unknown>;
    }
  | "FINISH";

interface ReActStep {
  thought: string;
  action: ReActAction;
}

interface AIExecutionPlan {
  steps: Array<{
    thought?: string;
    toolName?: string;
    params?: Record<string, unknown>;
  }>;
}

const MAX_REACT_ITERATIONS = 30;
const REACT_ITERATIONS_WARN_THRESHOLD = 5;
const MAX_HISTORY_CHARS = 40000;

const REACT_ITERATION_WARNING_MESSAGE = `⚠️ Warning: ${REACT_ITERATIONS_WARN_THRESHOLD} iterations remaining. You must wrap up your work and call submitForQA within the next ${REACT_ITERATIONS_WARN_THRESHOLD} iterations. Stop adding new features — focus only on ensuring the build passes and submitting.`;

/** Unified message shown to the model when ReAct JSON parse fails; used as thought and in RETRY_PARSE observation. */
const RETRY_PARSE_MESSAGE =
  'PARSE ERROR: Your last response was not valid JSON or missing required fields.\n\n' +
  'You MUST respond with EXACTLY this format:\n' +
  '{"thought": "<brief reasoning>", "action": {"toolName": "<toolName>", "params": {...}}}\n\n' +
  'Do NOT:\n' +
  '- Write prose explanations\n' +
  '- Use markdown code blocks\n' +
  '- Omit thought or action\n\n' +
  'Retry now with valid JSON.';

/** System prompt for Native Function Calling path (no JSON format instructions). */
function buildSystemPromptForNativeFC(injectedSystemSuffix?: string | null): string {
  return (
    `You are the TASK_EXECUTOR agent: a senior software engineer with access to tools. Do ONE small, concrete action per turn, then wait for the tool result.

Rules:
- Use the tool results (observations) to decide what to do next. If a command or tool failed, explain briefly how you will fix it, then fix it.
- If executeCommand returns failure for a build, type-check, or test, you MUST fix the code and re-run the command before finishing. Do NOT finish while the last build/type-check/test failed.
- Prefer getCodeMap, searchCodebase, readFile, and searchWeb to locate code or docs before editing.
- FILE EDITING RULES:
  - For SMALL or PARTIAL edits in existing files, you MUST use replaceInFile. Provide the exact searchString (a sufficiently unique block of code, minimum 3-4 lines to avoid ambiguous matches) and the replaceString.
  - DO NOT use writeFile to update existing files unless you are completely rewriting a very small file. writeFile is strictly for CREATING new files. Overwriting a 500-line file to change 1 line will cause critical data loss and token waste.
- executeCommand (sync) is STATELESS: 'cd folder' does NOT persist. Chain commands every time (e.g. 'cd folder && npm run build') or use the cwd parameter.
- Use executeCommand for builds, tests, linters, and for running short Node.js or Bash scripts you write to do complex edits. Never start long-running dev servers.
- When implementation is ready, call submitForQA or respond with text only to finish.
If you are completely stuck, missing critical dependencies, or the environment is broken beyond what you can fix (e.g. external APIs, credentials, or tools are unavailable), you MUST call a logical toolName "escalateToTeamLead" to stop execution and ask the Team Lead for help or task replanning.

## WORKSPACE SCAFFOLDING RULE
You are operating inside a dedicated project root directory. Whenever you initialize or scaffold a new project—regardless of the language, package manager, or framework (Node, Python, Rust, C++, etc.)—you MUST initialize it directly in the CURRENT directory (e.g., using '.' or the equivalent current-directory flag for your specific tool). NEVER create a nested project subfolder unless the user explicitly instructs you to do so.
CRITICAL FALLBACK: If a framework forces you to scaffold into a named subfolder (e.g., creating a my-app directory instead of using .), you MUST IMMEDIATELY move all generated files up to the current root directory using a command like mv my-app/* my-app/.* . 2>/dev/null || true && rm -rf my-app. The package.json MUST be located exactly in the project root before you proceed to any other step.

## Project Initialization
Choose appropriate approach based on task requirements:

**If starting fresh project:**
- React: npm create vite@latest . -- --template react-ts (or use . as target)
- Vue: npm create vue@latest .
- Next.js: npx create-next-app@latest . (or --directory .)
- Django: django-admin startproject mysite .
- FastAPI: mkdir app && create files
- Rust: cargo init . (or cargo new then move files to root)
- Go: go mod init . && mkdir as needed

**If project exists:**
- Read existing structure first
- Adapt to current stack

To initialize a new project, use executeCommand with the appropriate init command targeting the CURRENT directory (e.g. npx create-next-app@latest . --yes, or npm create vite@latest . -- --template react-ts). After scaffolding, run npm install from project root. Do not assume a nested project-name folder; run all commands from root.

DEPENDENCY INSTALLATION: Before running any install command, collect ALL required dependencies (runtime and dev) and install them in a single command. Never run multiple sequential install commands when one will do. This applies to all package managers (npm, pip, cargo, go mod, etc.).

SearchWeb: Use only when build/config fails and you need current syntax, or for specific library errors. Use precise queries. If the same fix failed twice, search before retrying.

Dynamic tooling: For complex file edits, use executeCommand to run a temporary Node.js script (fs.readFileSync, .replace(), fs.writeFileSync) or Bash/sed.

File system: All paths relative to project root (e.g. src/app/page.tsx). writeFile and replaceInFile operate on project-root-relative paths; writeFile creates parent dirs. New files: kebab-case. No ../ or absolute paths.

NPM: On ERESOLVE, retry with --legacy-peer-deps. Do not use --silent or --quiet.` +
    (injectedSystemSuffix?.trim() ? `\n\n${injectedSystemSuffix.trim()}` : "")
  );
}

function buildHistoryText(segments: ReActHistorySegment[]): string {
  return segments
    .map((s) => {
      switch (s.type) {
        case "task":
          return `=== TASK CONTEXT ===\n${s.text.trim()}\n`;
        case "thought":
          return `Thought: ${s.text.trim()}\n`;
        case "action":
          return `Action: ${s.text.trim()}\n`;
        case "observation":
          return `Observation: ${s.text.trim()}\n`;
        default:
          return s.text.trim();
      }
    })
    .join("\n")
    .trim();
}

function pruneHistorySegments(
  segments: ReActHistorySegment[],
  maxChars: number = MAX_HISTORY_CHARS
): ReActHistorySegment[] {
  let result = [...segments];
  let history = buildHistoryText(result);

  if (history.length <= maxChars) {
    return result;
  }

  const preservedPrefix: ReActHistorySegment[] = [];
  let i = 0;
  while (i < result.length && result[i].type === "task") {
    preservedPrefix.push(result[i]);
    i++;
  }

  let tail = result.slice(i);

  while (tail.length > 0 && buildHistoryText([...preservedPrefix, ...tail]).length > maxChars) {
    const idx = tail.findIndex((s) => s.type === "observation");
    if (idx === -1) {
      tail.shift();
    } else {
      const removeCount =
        idx + 1 < tail.length && (tail[idx + 1].type === "thought" || tail[idx + 1].type === "action")
          ? 2
          : 1;
      tail.splice(idx, removeCount);
    }
  }

  if (result.length !== preservedPrefix.length + tail.length) {
    const summary: ReActHistorySegment = {
      type: "observation",
      text: "[Earlier Observations were pruned to stay within context limits. They contained only older tool outputs and thoughts.]",
    };
    tail = [summary, ...tail];
  }

  return [...preservedPrefix, ...tail];
}

function extractJsonAndCode(
  rawText: string
): { jsonStr: string | null; trailingCode: string } {
  if (typeof rawText !== "string" || !rawText) {
    return { jsonStr: null, trailingCode: "" };
  }

  let startIndex = -1;

  // 1) Prefer an explicit ```json fenced block if present.
  const fencedJsonMatch = rawText.match(/```json\s*([\s\S]*?)```/i);
  if (fencedJsonMatch) {
    const fencedBlockStart = rawText.indexOf(fencedJsonMatch[0]);
    if (fencedBlockStart !== -1) {
      const innerBraceIndex = rawText.indexOf("{", fencedBlockStart);
      if (innerBraceIndex !== -1) {
        startIndex = innerBraceIndex;
      }
    }
  }

  // 2) Otherwise, look for objects that start with our known schema keys.
  if (startIndex === -1) {
    const thoughtIdx = rawText.indexOf('{"thought"');
    const actionIdx = rawText.indexOf('{"action"');

    if (thoughtIdx !== -1 && actionIdx !== -1) {
      startIndex = Math.min(thoughtIdx, actionIdx);
    } else if (thoughtIdx !== -1) {
      startIndex = thoughtIdx;
    } else if (actionIdx !== -1) {
      startIndex = actionIdx;
    }
  }

  // 3) As a last resort, fall back to the first '{', but bail out if the
  // prefix clearly looks like TypeScript/JS code (e.g. interfaces/classes).
  if (startIndex === -1) {
    const firstBrace = rawText.indexOf("{");
    if (firstBrace === -1) {
      return { jsonStr: null, trailingCode: "" };
    }

    const prefix = rawText.slice(0, firstBrace).toLowerCase();
    if (/\b(interface|class|function|type|enum|export|import)\b/.test(prefix)) {
      // Looks like pure code (e.g. `export interface X {`), not a JSON action.
      return { jsonStr: null, trailingCode: "" };
    }

    startIndex = firstBrace;
  }

  let braceCount = 0;
  let inString = false;
  let escapeNext = false;
  let endIndex = -1;

  for (let i = startIndex; i < rawText.length; i++) {
    const char = rawText[i];

    if (escapeNext) {
      escapeNext = false;
      continue;
    }

    if (char === "\\") {
      escapeNext = true;
      continue;
    }

    if (char === '"') {
      inString = !inString;
      continue;
    }

    if (!inString) {
      if (char === "{") {
        braceCount++;
      } else if (char === "}") {
        braceCount--;
        if (braceCount === 0) {
          endIndex = i;
          break;
        }
      }
    }
  }

  if (endIndex === -1) return { jsonStr: null, trailingCode: "" };

  const jsonStr = rawText.substring(startIndex, endIndex + 1);
  let trailingCode = rawText.substring(endIndex + 1).trim();

  if (trailingCode.startsWith("```")) {
    // Strip opening fence with optional language.
    trailingCode = trailingCode.replace(/^```[a-zA-Z0-9_-]*\n/, "");
    // Strip closing fence at the end.
    trailingCode = trailingCode.replace(/\n```$/, "");
  }

  return { jsonStr, trailingCode };
}

export class TaskExecutorAgent extends BaseAgent {
  constructor(config: AgentConfig) {
    super({ ...config, agentRole: config.agentRole ?? AgentRole.TASK_EXECUTOR });
  }

  async processMessage(message: AgentMessage): Promise<Record<string, unknown>> {
    switch (message.eventType) {
      case MessageType.TASK_REQUEST:
        return await this.handleTaskRequest(message);
      case MessageType.TICKET_REQUEST:
        return await this.handleTicketRequest(message);
      case MessageType.STYLE_RESPONSE: {
        this.log("info", `[${this.config.agentRole}] Received CSS styling completion`);
        const payload = message.payload as { taskId?: string; filePath?: string; improvedContent?: string };
        if (payload?.improvedContent && payload?.filePath && payload?.taskId) {
          try {
            const task = await prisma.task.findUnique({
              where: { id: payload.taskId },
              include: { plan: { include: { project: true } } },
            });
            if (task?.plan?.project) {
              await this.applyCssImprovedContent({
                projectId: task.plan.project.id,
                filePath: payload.filePath,
                improvedContent: payload.improvedContent,
              });
            }
          } catch (err) {
            this.log("error", `[${this.config.agentRole}] STYLE_RESPONSE apply failed: ${err instanceof Error ? err.message : String(err)}`);
          }
        }
        return {};
      }
      default:
        throw new Error(`Unknown event type: ${message.eventType}`);
    }
  }

  private async applyCssImprovedContent(params: {
    projectId: string;
    filePath: string;
    improvedContent: string;
  }): Promise<void> {
    const tools = createAgentTools(
      params.projectId,
      this.config.sessionId,
      this.config.mode ?? "local",
      this.log.bind(this)
    );
    const writeTool = tools.writeFile as unknown as {
      execute: (p: { filePath: string; content: string }) => Promise<{ success?: boolean; error?: string }>;
    };
    const writeResult = await writeTool.execute({
      filePath: params.filePath,
      content: params.improvedContent,
    });
    if (writeResult && typeof writeResult === "object" && !writeResult.success) {
      this.log(
        "error",
        `[${this.config.agentRole}] applyCssImprovedContent writeFile failed: ${writeResult.error ?? "unknown"}`
      );
    }
  }

  private async getNextReActStep(
    historyText: string,
    projectId: string,
    projectProvider?: string | null,
    projectModel?: string | null,
    injectedSystemSuffix?: string | null
  ): Promise<ReActStep> {
    const resolvedProvider = resolveProvider(projectProvider || undefined);
    const resolvedModel = projectModel || "gpt-4o-mini";
    const llmSettings = await getLLMSettings();
    const reactMaxTokens = Math.min(llmSettings.maxTokens ?? 16384, 16384);

    const systemPrompt = `You are the TASK_EXECUTOR agent running in a strict Reasoning loop of Thought -> Action -> Observation.

You MUST always respond with a single JSON object on each turn, no markdown or extra text.

Expected JSON schema:
{ "thought": "natural language reasoning about what to do next", "action": { "toolName": "<name>", "params": { ... } } | "FINISH" }

Rules:
- Never attempt to complete the whole task in one response. Do ONE small, concrete Action, then wait for the Observation.
- Use the Observations (previous tool outputs and errors) in the history to decide what to do next.
- If an Observation indicates an error or a failed command, your very next Thought MUST explicitly explain how you will fix that exact error before running more tools.
- THOUGHT CONCISENESS: When writing your "thought", DO NOT copy-paste or quote large blocks of error logs or code. Keep your thought under 3 sentences. Just state what (if anything) failed and what you will do next.
- SELF-CORRECTION RULE: If you run a build, type-check, tests, linters, or any custom script via executeCommand/cloudExecuteCommand and it FAILS (success: false or a non-zero exitCode in the last Observation), you MUST carefully read the stdout/stderr (including any logSnippet), identify the concrete errors, fix the code or configuration, and re-run the command. You are FORBIDDEN to return "action": "FINISH" while there are known failing build/type-check/test commands that you have not yet attempted to fix and re-run successfully.
- Prefer using getCodeMap, searchCodebase, readFile, and searchWeb (for external docs) to locate relevant code or documentation before editing.
- FILE EDITING RULES:
  - For SMALL or PARTIAL edits in existing files, you MUST use replaceInFile. Provide the exact searchString (a sufficiently unique block of code, minimum 3-4 lines to avoid ambiguous matches) and the replaceString.
  - DO NOT use writeFile to update existing files unless you are completely rewriting a very small file. writeFile is strictly for CREATING new files. Overwriting a 500-line file to change 1 line will cause critical data loss and token waste.
- Never use placeholders like "...", "// rest of file", or incomplete code blocks. All code you write must be complete and production-ready.
- Use executeCommand for focused commands (tests, builds, linters, simple shell checks) and for running short-lived helper scripts (Node.js or Bash) you write to perform complex file modifications. Never start long-running dev servers.
- Each executeCommand call runs in a fresh shell — directory changes via cd do not persist to the next call. Always either (a) use absolute paths, (b) use tool flags like --prefix or --cwd when available, or (c) pass the target directory explicitly in each command as cd /absolute/path && <your command>. In cloud mode, the working directory is persisted per session so cd does carry over; when in doubt use (a)–(c).
- WORKSPACE SCAFFOLDING RULE: You are operating inside a dedicated project root directory. Whenever you initialize or scaffold a new project—regardless of the language, package manager, or framework (Node, Python, Rust, C++, etc.)—you MUST initialize it directly in the CURRENT directory (e.g., using '.' or the equivalent current-directory flag for your specific tool). NEVER create a nested project subfolder unless the user explicitly instructs you to do so.
- When initializing a new project using any scaffolding tool (npm create, npx create-*, cargo init, go mod init, rails new, django-admin startproject, etc.), always target the current workspace root directory rather than creating a named subdirectory. Pass . or --directory . or equivalent as the target when the tool supports it. If the tool does not support in-place init and always creates a subdirectory, immediately move all files to the workspace root afterwards: mv <subdir>/{.,}* . 2>/dev/null; rm -rf <subdir>. The goal is that package.json (or the language-equivalent project manifest) always ends up directly in the workspace root.
- CRITICAL FALLBACK: If a framework forces you to scaffold into a named subfolder (e.g., creating a my-app directory instead of using .), you MUST IMMEDIATELY move all generated files up to the current root directory using a command like mv my-app/* my-app/.* . 2>/dev/null || true && rm -rf my-app. The package.json MUST be located exactly in the project root before you proceed to any other step.
- For initializing new projects, use executeCommand with the appropriate init command (e.g. npx create-next-app@latest . --yes, npm create vite@latest . -- --template react-ts). After scaffolding, run npm install from project root. Do not assume a nested project-name folder; run all commands from root.
- When working with Tailwind CSS, assume Tailwind v4 (or the version already configured in this project) and follow current best practices. Do NOT follow outdated guides that rely on manual PostCSS config if the project already has a working Tailwind setup.
- When you believe implementation is ready and verification is done, set "action": "FINISH". Optionally, you may call a logical toolName "submitForQA" as your final step to indicate readiness for QA.
- If you are completely stuck, missing critical dependencies, or the environment is broken beyond what you can fix, you may instead call a logical toolName "escalateToTeamLead" to stop execution and request human help from the Team Lead.

RESPONSE FORMAT - ALWAYS use this exact JSON structure:

CORRECT RESPONSES:
{"thought": "I need to install dependencies first", "action": {"toolName": "executeCommand", "params": {"command": "npm install tailwindcss"}}}
{"thought": "File was truncated, I'll use heredoc approach", "action": {"toolName": "executeCommand", "params": {"command": "cat << 'EOF' > file.ts\\ncontent\\nEOF"}}}
{"thought": "Previous command failed. I'll try with --legacy-peer-deps flag", "action": {"toolName": "executeCommand", "params": {"command": "npm install react-beautiful-dnd --legacy-peer-deps"}}}

WRONG RESPONSES (NEVER do this):
- "Let me try again because the error happened..." (prose only, no JSON)
- I will now fix the configuration file. (prose only)
- {"thought": "Fixing the issue"} (missing action)

FILE GENERATION PROTOCOL:
- When creating or completely rewriting a file, NEVER put the full file content inside the JSON params.content field. Keep the JSON small and stable to avoid JSON.parse crashes and __RETRY_PARSE__ loops.
- For such full-file writes, set "action": { "toolName": "writeCodeBlock", "params": { "filePath": "<relative/path/to/file>", "content": "<CODE_BELOW>" } } or omit params.content entirely.
- Immediately AFTER the JSON object, output the complete file content in a standard triple-backtick markdown code block. Example:
  { "thought": "Writing App.tsx", "action": { "toolName": "writeCodeBlock", "params": { "filePath": "src/App.tsx" } } }
  \`\`\`tsx
  // full contents of src/App.tsx here
  \`\`\`
- The executor will treat "writeCodeBlock" as a logical alias for "writeFile" and automatically capture the code from the following markdown block into params.content, as long as content is empty, very short, or set to "<CODE_BELOW>".
- For smaller changes in existing files, prefer replaceInFile with a precise searchString/replaceString pair instead of rewriting the whole file with writeFile.

SMART WEB SEARCH:
- You have a searchWeb tool. Use it ONLY for complex situations where local reasoning and the existing codebase are insufficient, such as:
  1) npm run build (or similar) fails due to configuration errors (for example: Vite PWA, Tailwind v4, Next.js config changes) and you do not know the correct modern syntax.
  2) You encounter a specific TypeScript or library error that you cannot resolve by inspecting the local code and your existing knowledge.
- Do NOT use searchWeb for basic logic, straightforward refactors, or questions that can be answered by reading the repository.
- When you call searchWeb, use precise, focused queries derived from the exact error or technology, such as "Tailwind v4 Vite setup", "Next.js app router static export error", or the exact TypeScript error code and library name.
- You can combine searchWeb with executeCommand: run a command via executeCommand, inspect the error in the Observation, then call searchWeb with a distilled version of that error text before updating the code or configuration.

EMERGENCY SEARCH RULES:
- If a command (build, tests, type-check, or tooling) fails and you do NOT understand the error from local context, you MUST call searchWeb with a focused query based on the exact error message before trying the same approach again.
- If you have retried essentially the same command or fix 2 times in a row and it is still failing, you MUST call searchWeb to look up a modern, correct solution before the next attempt.
- NEVER guess framework or library versions, or outdated configuration patterns for tools like Vite, Next.js, Turbopack, Tailwind, ESLint, or Vitest. When editing such configs or installing packages, use searchWeb to confirm current best practices and version constraints.

🪄 DYNAMIC TOOLING & FILE MODIFICATIONS:
If you need to make complex edits to a file or refactor code, use executeCommand to write and run your own temporary Node.js or Bash scripts.
- Example: Write a temp script (e.g. temp-edit.js) using a heredoc: \`cat << 'EOF' > temp-edit.js\\n...your script...\\nEOF\`, then run \`node temp-edit.js\` via executeCommand. In the script use fs.readFileSync, regex or string replacement, fs.writeFileSync, then verify the result and delete the temp file if desired.
- You can use Bash/sed/jq for simple text edits, or Node.js for multi-file or regex-heavy changes. You are Turing-complete through executeCommand; write the tools you need on the fly.

FILE SYSTEM RULES:
- All file paths for readFile, writeFile MUST be strictly relative to the project root (e.g. src/components/app.tsx). Never use \`../\` to escape the project root or absolute paths; they may be sanitized or rejected.
- The writeFile tool automatically creates any missing parent directories for a given path. Do NOT run separate mkdir commands just to create folders before writing files.
- When creating NEW files, ALWAYS use kebab-case for filenames (e.g. ios-button.tsx, user-profile-page.tsx). Do not create new files in PascalCase or camelCase. When modifying existing files, preserve the exact on-disk casing.
- Use consistent path casing and POSIX-style slashes (e.g. src/app/page.tsx).

NPM INSTALL RULES:
- DEPENDENCY INSTALLATION: Collect ALL required dependencies (runtime and dev) and install in one command. Do not run multiple sequential installs (npm/pip/cargo/go mod) when a single command suffices.
- If npm install fails with ERESOLVE error, immediately retry with --legacy-peer-deps flag.
- Example: npm install react-beautiful-dnd --legacy-peer-deps
- Do NOT use --silent or --quiet flags. You need to see errors to debug.

🆘 PRO-TIP:
- If readFile returns an empty string, the file might be empty or missing. Check your path. Do not keep reading the same file. Use searchWeb if you don't know how to configure Tailwind v4.` +
      (injectedSystemSuffix && injectedSystemSuffix.trim()
        ? `\n\n${injectedSystemSuffix.trim()}`
        : "");

    let rawText: string;

    try {
      if (resolvedProvider === "zai") {
        const { generateTextZai } = await import("@/lib/ai/zai");
        rawText = await generateTextZai({
          systemPrompt,
          userMessage: historyText,
          model: resolvedModel,
          temperature: llmSettings.temperature,
          maxTokens: reactMaxTokens,
          signal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
        });
      } else {
        const aiResult = await generateText({
          model: getModel(resolvedProvider, resolvedModel),
          system: systemPrompt,
          prompt: historyText,
          temperature: llmSettings.temperature,
          maxTokens: reactMaxTokens,
          abortSignal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
        });
        if (!aiResult) {
          throw new Error("AI returned null");
        }
        rawText = aiResult.text ?? "";

        try {
          await trackAIUsage(aiResult, {
            projectId,
            actionType: "execute_task",
            model: resolvedModel,
            executionSessionId: this.config.sessionId,
          });
        } catch (trackErr) {
          this.log(
            "error",
            `trackAIUsage failed for ReAct step: ${
              trackErr instanceof Error ? trackErr.message : String(trackErr)
            }`
          );
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      this.log("error", `[${this.config.agentRole}] ReAct LLM call failed: ${msg}`);
      throw new Error(`ReAct LLM call failed: ${msg}`);
    }

    let parsed: any;
    // Use a bracket-balancing extractor to safely isolate the first complete
    // JSON object, then allow any trailing markdown code (for writeCodeBlock)
    // after the closing brace.
    const { jsonStr, trailingCode } = extractJsonAndCode(rawText);

    if (!jsonStr) {
      const preview = rawText.slice(0, 500);
      this.log(
        "error",
        `[${this.config.agentRole}] No JSON object found in ReAct step after bracket-balanced extraction. Raw text: ${preview}`
      );

      const hasExplicitCodeFence = /```(?:ts|tsx|typescript|js|jsx|javascript)[\s\S]*?```/i.test(
        rawText
      );
      const looksLikeTsOrJs =
        hasExplicitCodeFence ||
        /\bexport\s+(interface|class|function|const|let|var|type|enum)\b/.test(
          preview
        );

      if (looksLikeTsOrJs) {
        return {
          thought: RETRY_PARSE_MESSAGE,
          action: { toolName: "__RETRY_PARSE__", params: {} },
        };
      }

      return {
        thought: RETRY_PARSE_MESSAGE,
        action: { toolName: "__RETRY_PARSE__", params: {} },
      };
    }

    // Optional: quick schema check before JSON.parse to fail fast on clearly invalid structure
    if (
      !jsonStr.includes('"thought"') ||
      !jsonStr.includes('"action"')
    ) {
      this.log(
        "error",
        `[${this.config.agentRole}] ReAct step JSON missing "thought" or "action" key in raw string; skipping parse.`
      );
      return {
        thought: RETRY_PARSE_MESSAGE,
        action: { toolName: "__RETRY_PARSE__", params: {} },
      };
    }

    try {
      parsed = JSON.parse(jsonStr);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.log(
        "error",
        `[${this.config.agentRole}] Failed to parse ReAct step JSON after extraction: ${msg}. Extracted length=${jsonStr.length}. Raw text (truncated): ${rawText.slice(
          0,
          500
        )}`
      );
      return {
        thought: RETRY_PARSE_MESSAGE,
        action: { toolName: "__RETRY_PARSE__", params: {} },
      };
    }

    if (!parsed || typeof parsed !== "object") {
      this.log(
        "error",
        `[${this.config.agentRole}] ReAct step JSON is not an object. Raw parsed type: ${typeof parsed}`
      );
      return {
        thought: RETRY_PARSE_MESSAGE,
        action: { toolName: "__RETRY_PARSE__", params: {} },
      };
    }

    const hasThought =
      typeof parsed.thought === "string" && parsed.thought.trim().length > 0;
    const hasAction = parsed.action !== undefined && parsed.action !== null;

    if (!hasThought || !hasAction) {
      this.log(
        "error",
        `[${this.config.agentRole}] ReAct step missing required "thought" or "action"; forcing __RETRY_PARSE__.`
      );
      return {
        thought: RETRY_PARSE_MESSAGE,
        action: { toolName: "__RETRY_PARSE__", params: {} },
      };
    }

    const rawThought = (parsed.thought as string).trim();
    let rawAction = parsed.action;

    const thought: string = rawThought;

    let action: ReActAction;

    // Normalize action shape and support writeCodeBlock/markdown extraction.
    if (typeof rawAction === "string") {
      if (rawAction.toUpperCase() === "FINISH") {
        action = "FINISH";
      } else {
        this.log(
          "error",
          `[${this.config.agentRole}] ReAct step has invalid string action: ${rawAction}; forcing __RETRY_PARSE__.`
        );
        return {
          thought: RETRY_PARSE_MESSAGE,
          action: { toolName: "__RETRY_PARSE__", params: {} },
        };
      }
    } else if (rawAction && typeof rawAction === "object") {
      let toolName =
        typeof (rawAction as any).toolName === "string"
          ? (rawAction as any).toolName.trim()
          : "";
      const paramsRaw =
        (rawAction as any).params && typeof (rawAction as any).params === "object"
          ? ((rawAction as any).params as Record<string, unknown>)
          : {};

      if (!toolName) {
        this.log(
          "error",
          `[${this.config.agentRole}] ReAct step.action missing "toolName"; forcing __RETRY_PARSE__.`
        );
        return {
          thought: RETRY_PARSE_MESSAGE,
          action: { toolName: "__RETRY_PARSE__", params: {} },
        };
      }

      // Remember the original toolName so we can apply slightly different
      // heuristics for the writeCodeBlock alias vs plain writeFile.
      const originalToolName = toolName;

      // Map logical alias writeCodeBlock -> writeFile.
      if (toolName === "writeCodeBlock") {
        toolName = "writeFile";
      }

      const params: Record<string, unknown> = { ...paramsRaw };

      // For writeFile/writeCodeBlock, if content is missing/placeholder, try to
      // inject code from the trailing markdown block that follows the JSON.
      if (
        toolName === "writeFile" &&
        typeof params.filePath === "string" &&
        params.filePath.trim().length > 0
      ) {
        const rawContent =
          typeof params.content === "string" ? params.content.trim() : "";

        const isWriteCodeBlock = originalToolName === "writeCodeBlock";
        const isPlaceholderOrEmpty = isWriteCodeBlock
          ? rawContent.length < 50 || rawContent.includes("<CODE_BELOW>")
          : !rawContent ||
            rawContent === "<CODE_BELOW>" ||
            rawContent.length < 16;

        if (isWriteCodeBlock && trailingCode) {
          // For the explicit writeCodeBlock alias, always prefer the trailing
          // markdown code when present.
          params.content = trailingCode;
        } else if (isPlaceholderOrEmpty && trailingCode) {
          params.content = trailingCode;
        }
      }

      action = { toolName, params };
    } else {
      this.log(
        "error",
        `[${this.config.agentRole}] ReAct step.action must be a string or object; forcing __RETRY_PARSE__.`
      );
      return {
        thought: RETRY_PARSE_MESSAGE,
        action: { toolName: "__RETRY_PARSE__", params: {} },
      };
    }

    return { thought, action };
  }

  private async compressHistory(
    historySegments: ReActHistorySegment[],
    options: {
      projectId: string;
      projectProvider?: string | null;
      projectModel?: string | null;
      llmSettings: any;
    }
  ): Promise<string> {
    const historyText = buildHistoryText(historySegments);
    if (!historyText.trim()) {
      return "- No meaningful history yet; nothing to summarize.";
    }

    const { projectId, projectProvider, projectModel, llmSettings } = options;

    const resolvedProvider = resolveProvider(projectProvider || undefined);
    const resolvedModel = projectModel || "gpt-4o-mini";
    const diaryMaxTokens = Math.min(llmSettings?.maxTokens ?? 16384, 512);

    const systemPrompt =
      "You are an AI assistant summarizing an execution history. " +
      "Summarize the actions taken so far, the errors or failures encountered, " +
      "and the current goal or blockers in 3-5 concise bullet points. " +
      "Do NOT write code or pseudo-code. Keep it brief and high-level.";

    try {
      let rawSummary: string;

      if (resolvedProvider === "zai") {
        const { generateTextZai } = await import("@/lib/ai/zai");
        rawSummary = await generateTextZai({
          systemPrompt,
          userMessage: historyText,
          model: resolvedModel,
          temperature: llmSettings?.temperature,
          maxTokens: diaryMaxTokens,
          signal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
        });
      } else {
        const aiResult = await generateText({
          model: getModel(resolvedProvider, resolvedModel),
          system: systemPrompt,
          prompt: historyText,
          temperature: llmSettings?.temperature,
          maxTokens: diaryMaxTokens,
          abortSignal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
        });
        if (!aiResult) {
          throw new Error("AI returned null while summarizing history");
        }
        rawSummary = aiResult.text ?? "";

        try {
          await trackAIUsage(aiResult, {
            projectId,
            actionType: "execute_task",
            model: resolvedModel,
            executionSessionId: this.config.sessionId,
          });
        } catch (trackErr) {
          this.log(
            "error",
            `trackAIUsage failed for Agent Diary summary: ${
              trackErr instanceof Error ? trackErr.message : String(trackErr)
            }`
          );
        }
      }

      const trimmed = (rawSummary || "").trim();
      if (!trimmed) {
        return "- Agent Diary summary was empty; continue using the latest observations and try a fresh approach.";
      }
      return trimmed;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.log(
        "error",
        `[${this.config.agentRole}] Agent Diary summarization failed: ${msg}`
      );
      return `- Unable to summarize history due to LLM error: ${msg}`;
    }
  }

  private async runReActLoop(options: {
    projectId: string;
    projectProvider?: string | null;
    projectModel?: string | null;
    initialTaskContext: string;
    tools: ReturnType<typeof createAgentTools>;
    primaryTaskId: string;
    parentMessageId: string;
  }): Promise<{
    results: any[];
    lastBuildPassed: boolean;
    shouldSubmitForQA: boolean;
    lastSuccessfulCssWrite: { filePath: string; content: string } | null;
    shouldEscalate: boolean;
    lastCompletedIteration: number;
  }> {
    const {
      projectId,
      projectProvider,
      projectModel,
      initialTaskContext,
      tools,
      primaryTaskId,
      parentMessageId,
    } = options;

    const workDir =
      this.config.mode === "cloud" ? CONTAINER_WORKSPACE_ROOT : getProjectDir(projectId);
    const workDirForRead = getProjectDir(projectId);
    this.log("info", `[${this.config.agentRole}] Working directory: ${workDir}`);
    try {
      const topLevel = await fs.readdir(workDirForRead).catch(() => []);
      const names = Array.isArray(topLevel) ? topLevel : [];
      this.log("info", `[${this.config.agentRole}] Project structure: ${names.join(", ") || "(empty or inaccessible)"}`);
    } catch {
      this.log("info", `[${this.config.agentRole}] Project structure: (could not read directory)`);
    }

    const llmSettings = await getLLMSettings();

    let initializedProjectContext: string | null = null;

    let historySegments: ReActHistorySegment[] = [
      {
        type: "task",
        text: initialTaskContext,
      },
    ];

    const results: any[] = [];
    const commandFailureMap = new Map<
      string,
      {
        lastLog: string;
        count: number;
      }
    >();
    let shouldSubmitForQA = false;
    let lastBuildPassed = true;
    let parseRetryCount = 0;
    let consecutiveFailedIterations = 0;
    let stuckEscalationTriggered = false;
    let lastActionKey: string | null = null;
    let sameActionCount = 0;
    let lastSuccessfulCssWrite: { filePath: string; content: string } | null =
      null;
    let shouldEscalate = false;

    const loopStartTime = Date.now();
    const toolCallCounts = new Map<string, number>();
    let lastCompletedIteration = 0;
    const resolvedProvider = resolveProvider(projectProvider || undefined);
    const useNativeFC = true;

    if (!useNativeFC) {
      // Legacy segment loop (kept for reference or fallback)
      let iterationWarningShown = false;
    for (let iteration = 1; iteration <= MAX_REACT_ITERATIONS; iteration++) {
      const iterationStart = Date.now();
      lastCompletedIteration = iteration;

      const historyText = buildHistoryText(historySegments);

      this.log(
        "info",
        `[${this.config.agentRole}] ReAct iteration ${iteration}/${MAX_REACT_ITERATIONS} (history length=${historyText.length})`
      );

      if (!iterationWarningShown && iteration === MAX_REACT_ITERATIONS - REACT_ITERATIONS_WARN_THRESHOLD) {
        iterationWarningShown = true;
        historySegments.push({
          type: "observation",
          text: JSON.stringify({
            system: "ITERATION_WARNING",
            message: REACT_ITERATION_WARNING_MESSAGE,
            iterationsRemaining: REACT_ITERATIONS_WARN_THRESHOLD,
          }),
        });
      }

      const step = await this.getNextReActStep(
        historyText,
        projectId,
        projectProvider || undefined,
        projectModel || undefined,
        initializedProjectContext
      );

      this.log("info", `[${this.config.agentRole}] 🧠 ${step.thought}`);
      historySegments.push({ type: "thought", text: step.thought });

      if (step.action === "FINISH") {
        if (!lastBuildPassed) {
          this.log(
            "info",
            `[${this.config.agentRole}] Agent attempted to FINISH with a broken build/type-check. Intercepting.`
          );
          const enforcementObservation = {
            system: "FINISH_INTERCEPT",
            message:
              "SYSTEM ENFORCEMENT: You attempted to FINISH, but your last build/type-check command failed. You are TRAPPED in this loop. You MUST read the compiler errors, use writeFile or executeCommand to fix the code, and run the build command again successfully before finishing.",
            lastBuildPassed,
          };
          historySegments.push({
            type: "observation",
            text: JSON.stringify(enforcementObservation),
          });
          historySegments = pruneHistorySegments(historySegments);
          const _elapsed0 = Date.now() - iterationStart;
          if (_elapsed0 > 30000) {
            this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_elapsed0}ms`);
          }
          continue;
        }
        const _elapsed1 = Date.now() - iterationStart;
        if (_elapsed1 > 30000) {
          this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_elapsed1}ms`);
        }
        this.log(
          "info",
          `[${this.config.agentRole}] Received FINISH signal from ReAct loop`
        );
        historySegments.push({ type: "action", text: "FINISH" });
        break;
      }

      const { toolName, params = {} } = step.action;

      if (toolName === "escalateToTeamLead") {
        shouldEscalate = true;
        historySegments.push({
          type: "observation",
          text: JSON.stringify({ system: "ESCALATION_REQUESTED" }),
        });
        const _elapsedEsc = Date.now() - iterationStart;
        if (_elapsedEsc > 30000) {
          this.log(
            "warn",
            `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_elapsedEsc}ms`
          );
        }
        break;
      }

      if (toolName === "__RETRY_PARSE__") {
        if (parseRetryCount >= 2) {
          const _elapsed2 = Date.now() - iterationStart;
          if (_elapsed2 > 30000) {
            this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_elapsed2}ms`);
          }
          this.log(
            "error",
            `[${this.config.agentRole}] Max ReAct JSON parse retries reached; finishing to avoid infinite loop.`
          );
          const observation = {
            system: "PARSE_RETRY_LIMIT",
            message:
              "SYSTEM: Your responses repeatedly contained invalid or truncated JSON. The executor is finishing this task to avoid an infinite loop.",
            parseRetryCount,
          };
          historySegments.push({
            type: "observation",
            text: JSON.stringify(observation),
          });
          historySegments = pruneHistorySegments(historySegments);
          break;
        }

        parseRetryCount += 1;
        let retryMessage = RETRY_PARSE_MESSAGE;
        if (parseRetryCount >= 2) {
          this.log(
            "warn",
            `[${this.config.agentRole}] Multiple parse errors, agent may be stuck`
          );
          retryMessage += " You seem stuck. Try a different approach or simpler action.";
        }
        const observation = {
          system: "RETRY_PARSE",
          message: retryMessage,
          parseRetryCount,
        };
        historySegments.push({
          type: "observation",
          text: JSON.stringify(observation),
        });
        historySegments = pruneHistorySegments(historySegments);
        const _elapsed3 = Date.now() - iterationStart;
        if (_elapsed3 > 30000) {
          this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_elapsed3}ms`);
        }
        continue;
      }

      // Repeat Action Monitor: detect and break tight loops where the agent
      // keeps calling the exact same tool with identical params.
      let currentActionKey: string | null = null;
      try {
        currentActionKey = `${toolName}:${JSON.stringify(params ?? {})}`;
      } catch {
        currentActionKey = `${toolName}:__UNSERIALIZABLE_PARAMS__`;
      }

      if (currentActionKey) {
        if (currentActionKey === lastActionKey) {
          sameActionCount += 1;
        } else {
          lastActionKey = currentActionKey;
          sameActionCount = 1;
        }

        if (sameActionCount >= 3) {
          const loopObservation = {
            system: "REPEAT_ACTION_DETECTED",
            message:
              "ERROR: You are stuck in a loop repeating the same action. You must change your approach, use searchWeb, or fix the code.",
            toolName,
            params,
            sameActionCount,
          };
          historySegments.push({
            type: "observation",
            text: JSON.stringify(loopObservation),
          });
          historySegments = pruneHistorySegments(historySegments);
          const _elapsed4 = Date.now() - iterationStart;
          if (_elapsed4 > 30000) {
            this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_elapsed4}ms`);
          }
          continue;
        }
      } else {
        lastActionKey = null;
        sameActionCount = 0;
      }

      const isExecTool =
        toolName === "executeCommand" || toolName === "cloudExecuteCommand";
      const commandParam =
        isExecTool && typeof (params as any)?.command === "string"
          ? (params as any).command
          : undefined;
      const actionSummary = JSON.stringify({ toolName, params }, null, 2).slice(
        0,
        1000
      );
      historySegments.push({ type: "action", text: actionSummary });

      if (toolName === "submitForQA") {
        shouldSubmitForQA = true;
        const _elapsed5 = Date.now() - iterationStart;
        if (_elapsed5 > 30000) {
          this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_elapsed5}ms`);
        }
        this.log(
          "info",
          `[${this.config.agentRole}] Logical submitForQA action received from ReAct loop`
        );
        const observation = {
          toolName,
          status: "queuedForQA",
          note: "ReAct agent indicated readiness for QA. Proceeding to verification and QA pipeline.",
        };
        historySegments.push({
          type: "observation",
          text: JSON.stringify(observation),
        });
        break;
      }

      if (!(toolName in tools)) {
        if (toolName === "escalateToTeamLead") {
          const _elapsedEscUnknown = Date.now() - iterationStart;
          if (_elapsedEscUnknown > 30000) {
            this.log(
              "warn",
              `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_elapsedEscUnknown}ms`
            );
          }
          continue;
        }
        const errorMsg = `Unknown toolName requested by ReAct agent: ${toolName}`;
        this.log("error", `[${this.config.agentRole}] ${errorMsg}`);
        const observation = {
          toolName,
          success: false,
          error: errorMsg,
        };
        historySegments.push({
          type: "observation",
          text: JSON.stringify(observation),
        });
        historySegments = pruneHistorySegments(historySegments);
        const _elapsed6 = Date.now() - iterationStart;
        if (_elapsed6 > 30000) {
          this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_elapsed6}ms`);
        }
        continue;
      }

      const commandSuffix = commandParam ? `: ${commandParam}` : "";

      this.log(
        "info",
        `[${this.config.agentRole}] 🛠️ Executing: ${toolName}${commandSuffix}`
      );

      toolCallCounts.set(toolName, (toolCallCounts.get(toolName) ?? 0) + 1);

      let iterationSucceeded = false;

      try {
        const tool = tools[toolName as keyof typeof tools];
        const result = await (tool as any).execute(params);

        const anyResult = result as any;
        const rawExitCode =
          typeof anyResult?.exitCode === "number" ? anyResult.exitCode : undefined;
        const explicitSuccess =
          typeof anyResult?.success === "boolean" ? anyResult.success : undefined;

        let normalizedSuccess = true;
        if (explicitSuccess !== undefined || rawExitCode !== undefined) {
          normalizedSuccess =
            explicitSuccess !== false && (rawExitCode == null || rawExitCode === 0);
        }

        if (isExecTool && commandParam) {
          const lowerCmd = commandParam.toLowerCase();
          if (
            lowerCmd.includes("build") ||
            lowerCmd.includes("tsc") ||
            lowerCmd.includes("type-check") ||
            lowerCmd.includes("typecheck")
          ) {
            lastBuildPassed = normalizedSuccess;
          }
        }

        let repeatedFailureNote: string | undefined;

        if (!normalizedSuccess && isExecTool && commandParam) {
          const normalizedLog =
            (typeof anyResult?.stderr === "string" && anyResult.stderr.trim()) ||
            (typeof anyResult?.error === "string" && anyResult.error.trim()) ||
            (typeof anyResult?.stdout === "string" && anyResult.stdout.trim()) ||
            "";

          if (normalizedLog) {
            const existing = commandFailureMap.get(commandParam);
            if (existing && existing.lastLog === normalizedLog) {
              const newCount = existing.count + 1;
              commandFailureMap.set(commandParam, {
                lastLog: normalizedLog,
                count: newCount,
              });
              if (newCount >= 3) {
                repeatedFailureNote = `This command has already failed ${newCount} times with essentially the same logs. You must change the code or configuration before running it again; do not just retry the command without fixes.`;
              }
            } else {
              commandFailureMap.set(commandParam, {
                lastLog: normalizedLog,
                count: 1,
              });
            }
          }
        }

        if (normalizedSuccess) {
          iterationSucceeded = true;
          results.push({ toolName, result, success: true });
          this.log(
            "success",
            `[${this.config.agentRole}] ✅ ${toolName} completed`
          );
        } else {
          const stderr =
            typeof anyResult?.stderr === "string" ? anyResult.stderr : "";
          const errorField =
            typeof anyResult?.error === "string" ? anyResult.error : "";
          const snippet = (errorField || stderr).slice(0, 500);
          const exitCodeInfo =
            rawExitCode != null ? ` (exitCode=${rawExitCode})` : "";

          this.log(
            "error",
            `[${this.config.agentRole}] ❌ Tool ${toolName} reported failure${exitCodeInfo}${
              snippet ? `: ${snippet}` : ""
            }`
          );

          results.push({ toolName, result, success: false });
        }

        const stepParams = params as Record<string, unknown>;
        if (
          normalizedSuccess &&
          toolName === "writeFile" &&
          typeof stepParams.filePath === "string"
        ) {
          const filePath = stepParams.filePath;
          if (
            filePath.endsWith(".css") ||
            filePath.endsWith(".scss") ||
            filePath.endsWith(".module.css")
          ) {
            lastSuccessfulCssWrite = {
              filePath,
              content:
                typeof stepParams.content === "string" ? stepParams.content : "",
            };
          }
        }

        const fullStdout =
          typeof anyResult?.stdout === "string" ? anyResult.stdout : "";
        const fullStderr =
          typeof anyResult?.stderr === "string" ? anyResult.stderr : "";
        const fullError =
          typeof anyResult?.error === "string" ? anyResult.error : "";

        const logSourceBase = [fullStdout, fullStderr].filter(Boolean).join("\n");
        const primaryLogSource = logSourceBase || fullError;

        const maxLogChars = isExecTool ? 4000 : 1000;
        let logSnippet: string | undefined;
        if (primaryLogSource) {
          logSnippet =
            primaryLogSource.length <= maxLogChars
              ? primaryLogSource
              : primaryLogSource.slice(-maxLogChars);
        }

        const observation: {
          toolName: string;
          success: boolean;
          exitCode?: number;
          stdoutSnippet?: string;
          stderrSnippet?: string;
          errorSnippet?: string;
          logSnippet?: string;
          repeatedFailureNote?: string;
          projectPath?: string;
          projectName?: string;
          isRootProject?: boolean;
          message?: string;
        } = {
          toolName,
          success: normalizedSuccess,
          exitCode: rawExitCode,
          stdoutSnippet: fullStdout ? fullStdout.slice(0, 500) : undefined,
          stderrSnippet: fullStderr ? fullStderr.slice(0, 500) : undefined,
          errorSnippet: fullError ? fullError.slice(0, 500) : undefined,
          logSnippet,
        };

        if (repeatedFailureNote) {
          observation.repeatedFailureNote = repeatedFailureNote;
        }

        historySegments.push({
          type: "observation",
          text: JSON.stringify(observation),
        });
      } catch (toolErr) {
        const errMsg = toolErr instanceof Error ? toolErr.message : String(toolErr);
        this.log("error", `[${this.config.agentRole}] ${toolName} failed: ${errMsg}`);
        results.push({ toolName, error: errMsg, success: false });

        if (isExecTool && commandParam) {
          const lowerCmd = commandParam.toLowerCase();
          if (
            lowerCmd.includes("build") ||
            lowerCmd.includes("tsc") ||
            lowerCmd.includes("type-check") ||
            lowerCmd.includes("typecheck")
          ) {
            lastBuildPassed = false;
          }
        }

        const observation = {
          toolName,
          success: false,
          errorSnippet: errMsg.slice(0, 500),
        };
        historySegments.push({
          type: "observation",
          text: JSON.stringify(observation),
        });
      }

      if (iterationSucceeded) {
        consecutiveFailedIterations = 0;
      } else {
        consecutiveFailedIterations += 1;
      }

      if (
        !stuckEscalationTriggered &&
        (iteration >= 20 && consecutiveFailedIterations >= 3)
      ) {
        const stuckObservation = {
          system: "STUCK_LOOP",
          message:
            "SYSTEM: You have had multiple iterations without a successful tool call. You appear stuck. Summarize what you tried and what is blocking you, then either make a concrete fix using writeFile or executeCommand, or FINISH to hand off.",
          iteration,
          consecutiveFailedIterations,
        };
        historySegments.push({
          type: "observation",
          text: JSON.stringify(stuckObservation),
        });
        stuckEscalationTriggered = true;
      }

      if (iteration > 0 && iteration % 8 === 0) {
        let lastObservation: ReActHistorySegment | null = null;
        for (let i = historySegments.length - 1; i >= 0; i--) {
          if (historySegments[i].type === "observation") {
            lastObservation = historySegments[i];
            break;
          }
        }
        const lastText = lastObservation?.text ?? "";
        const hasErrorContext =
          lastText.includes("error") ||
          lastText.includes("Error") ||
          lastText.includes("failed") ||
          lastText.includes("cannot find");

        if (!hasErrorContext) {
          try {
            this.log(
              "info",
              `[${this.config.agentRole}] Compressing agent history (Agent Diary) at iteration ${iteration}...`
            );

            const summary = await this.compressHistory(historySegments, {
              projectId,
              projectProvider,
              projectModel,
              llmSettings,
            });

            const effectiveSummary =
              typeof summary === "string" && summary.trim()
                ? summary.trim()
                : "- Failed to summarize history; continuing with last 2 steps only.";

            const prefixTasks: ReActHistorySegment[] = [];
            let idx = 0;
            while (
              idx < historySegments.length &&
              historySegments[idx].type === "task"
            ) {
              prefixTasks.push(historySegments[idx]);
              idx += 1;
            }

            const nonTaskTail = historySegments.slice(idx);
            const recentTail =
              nonTaskTail.length <= 2
                ? nonTaskTail
                : nonTaskTail.slice(nonTaskTail.length - 2);

            const diarySegment: ReActHistorySegment = {
              type: "observation",
              text:
                "SYSTEM: AGENT DIARY (CONTEXT COMPRESSION):\n" +
                effectiveSummary,
            };

            historySegments = [...prefixTasks, diarySegment, ...recentTail];
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            this.log(
              "error",
              `[${this.config.agentRole}] Failed to compress history (Agent Diary): ${msg}`
            );
          }
        }
      }

      const _elapsedEnd = Date.now() - iterationStart;
      if (_elapsedEnd > 30000) {
        this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_elapsedEnd}ms`);
      }
      historySegments = pruneHistorySegments(historySegments);
    }
    } else {
      // Native Function Calling path (OpenAI, Anthropic): no JSON parsing
      const resolvedModel = projectModel || "gpt-4o-mini";
      const reactMaxTokens = Math.min(llmSettings.maxTokens ?? 16384, 16384);
      const systemPromptFC = buildSystemPromptForNativeFC(initializedProjectContext);
      let messages: CoreMessage[] = [{ role: "user", content: initialTaskContext }];
      const MAX_MESSAGES_BEFORE_DIARY = 30;
      const DIARY_KEEP_TAIL = 4;
      let iterationWarningShownFC = false;

      for (let iteration = 1; iteration <= MAX_REACT_ITERATIONS; iteration++) {
        const iterationStart = Date.now();
        lastCompletedIteration = iteration;
        this.log(
          "info",
          `[${this.config.agentRole}] ReAct iteration ${iteration}/${MAX_REACT_ITERATIONS} (messages=${messages.length})`
        );

        if (!iterationWarningShownFC && iteration === MAX_REACT_ITERATIONS - REACT_ITERATIONS_WARN_THRESHOLD) {
          iterationWarningShownFC = true;
          messages.push({ role: "user", content: REACT_ITERATION_WARNING_MESSAGE });
        }

        let aiResult;
        try {
          aiResult = await generateText({
            model: getModel(resolvedProvider, resolvedModel),
            system: systemPromptFC,
            messages,
            tools,
            maxSteps: 1,
            temperature: llmSettings.temperature,
            maxTokens: reactMaxTokens,
            abortSignal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
          });
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          this.log("error", `[${this.config.agentRole}] Native FC generateText failed: ${msg}`);
          throw new Error(`Native FC generateText failed: ${msg}`);
        }

        try {
          await trackAIUsage(aiResult, {
            projectId,
            actionType: "execute_task",
            model: resolvedModel,
            executionSessionId: this.config.sessionId,
          });
        } catch (trackErr) {
          this.log(
            "error",
            `trackAIUsage failed: ${trackErr instanceof Error ? trackErr.message : String(trackErr)}`
          );
        }

        const step = aiResult.steps?.[0];
        const responseMessages = aiResult.response?.messages ?? [];
        const thoughtForLog = (step?.text ?? aiResult.text ?? "").trim() || "(tool call)";
        this.log("info", `[${this.config.agentRole}] 🧠 ${thoughtForLog}`);

        const hasToolCalls = step && Array.isArray(step.toolCalls) && step.toolCalls.length > 0;
        const finishText = (step?.text ?? aiResult.text ?? "").trim();

        if (!step && !finishText && responseMessages.length === 0) {
          this.log("warn", `[${this.config.agentRole}] Native FC: no step, no text, no messages; continuing`);
          continue;
        }

        if (!hasToolCalls && finishText) {
          if (!lastBuildPassed) {
            this.log(
              "info",
              `[${this.config.agentRole}] Agent attempted to FINISH with a broken build/type-check. Intercepting.`
            );
            messages.push({
              role: "user",
              content:
                "SYSTEM ENFORCEMENT: You attempted to finish, but your last build/type-check command failed. Fix the errors with writeFile or executeCommand and run the build again successfully before finishing.",
            });
            if (Date.now() - iterationStart > 30000) {
              this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${Date.now() - iterationStart}ms`);
            }
            continue;
          }
          this.log("info", `[${this.config.agentRole}] Received FINISH (text response) from model`);
          break;
        }

        if (!hasToolCalls) {
          this.log("warn", `[${this.config.agentRole}] No tool calls and no text; continuing`);
          if (responseMessages.length > 0) messages.push(...responseMessages);
          if (Date.now() - iterationStart > 30000) {
            this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${Date.now() - iterationStart}ms`);
          }
          continue;
        }

        const toolCalls = step!.toolCalls as Array<{ toolName: string; args: Record<string, unknown> }>;
        const toolResults = step!.toolResults as Array<unknown>;

        const CMD_LOG_MAX = 120;
        for (const tc of toolCalls) {
          const toolName = tc?.toolName ?? (tc as { name?: string })?.name ?? "unknown";
          const params = tc?.args ?? {};
          if (toolName === "executeCommand" && typeof params.command === "string") {
            const cmd = params.command as string;
            const suffix = cmd.length > CMD_LOG_MAX ? cmd.slice(0, CMD_LOG_MAX) + "..." : cmd;
            this.log("info", `[${this.config.agentRole}] 🛠️ Executing: ${toolName}: ${suffix}`);
          } else if (toolName === "writeFile" && (params.filePath ?? params.path)) {
            this.log("info", `[${this.config.agentRole}] 🛠️ Executing: ${toolName}: ${params.filePath ?? params.path}`);
          } else if (toolName === "readFile" && (params.filePath ?? params.path)) {
            this.log("info", `[${this.config.agentRole}] 🛠️ Executing: ${toolName}: ${params.filePath ?? params.path}`);
          } else {
            this.log("info", `[${this.config.agentRole}] 🛠️ Executing: ${toolName}`);
          }
        }

        for (let i = 0; i < toolCalls.length; i++) {
          const tc = toolCalls[i] as { toolName?: string; name?: string; args?: Record<string, unknown> };
          const name = tc.toolName ?? tc.name ?? "";
          if (name === "submitForQA") {
            shouldSubmitForQA = true;
          } else if (name === "escalateToTeamLead") {
            shouldEscalate = true;
          }
          toolCallCounts.set(name, (toolCallCounts.get(name) ?? 0) + 1);
        }

        const currentActionKey =
          toolCalls.length > 0
            ? `${toolCalls.map((tc) => tc.toolName).join(",")}:${JSON.stringify(toolCalls.map((tc) => tc.args))}`
            : null;
        if (currentActionKey) {
          if (currentActionKey === lastActionKey) {
            sameActionCount += 1;
          } else {
            lastActionKey = currentActionKey;
            sameActionCount = 1;
          }
          if (sameActionCount >= 3) {
            messages.push(...responseMessages);
            messages.push({
              role: "user",
              content:
                "ERROR: You are stuck repeating the same action. Change your approach, use searchWeb, or fix the code.",
            });
            if (Date.now() - iterationStart > 30000) {
              this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${Date.now() - iterationStart}ms`);
            }
            continue;
          }
        }

        for (let i = 0; i < toolResults.length; i++) {
          const tr = toolResults[i] as Record<string, unknown> | undefined;
          const tc = toolCalls[i] as { toolName?: string; name?: string; args?: Record<string, unknown> };
          const name = tc.toolName ?? tc.name ?? "";
          if (!tr) continue;
          const explicitSuccess = typeof tr.success === "boolean" ? tr.success : undefined;
          const rawExitCode = typeof tr.exitCode === "number" ? tr.exitCode : undefined;
          const normalizedSuccess =
            explicitSuccess !== false && (rawExitCode == null || rawExitCode === 0);
          const cmd = typeof tc?.args?.command === "string" ? tc.args.command : "";
          if ((name === "executeCommand" || name === "cloudExecuteCommand") && cmd) {
            const lower = cmd.toLowerCase();
            if (
              lower.includes("build") ||
              lower.includes("tsc") ||
              lower.includes("type-check") ||
              lower.includes("typecheck")
            ) {
              lastBuildPassed = normalizedSuccess;
            }
          }
          if (
            normalizedSuccess &&
            name === "writeFile" &&
            typeof tc?.args?.filePath === "string"
          ) {
            const filePath = tc.args.filePath as string;
            if (
              filePath.endsWith(".css") ||
              filePath.endsWith(".scss") ||
              filePath.endsWith(".module.css")
            ) {
              lastSuccessfulCssWrite = {
                filePath,
                content: typeof tc.args.content === "string" ? tc.args.content : "",
              };
            }
          }
          results.push({ toolName: name, result: tr, success: normalizedSuccess });
        }

        consecutiveFailedIterations = toolResults.length > 0 ? 0 : consecutiveFailedIterations + 1;
        if (iteration >= 20 && consecutiveFailedIterations >= 3 && !stuckEscalationTriggered) {
          stuckEscalationTriggered = true;
          messages.push({
            role: "user",
            content:
              "SYSTEM: You have had multiple iterations without a successful tool call. Summarize what blocks you, then make a concrete fix or finish to hand off.",
          });
        }

        messages.push(...responseMessages);

        if (shouldSubmitForQA) {
          this.log("info", `[${this.config.agentRole}] submitForQA called; exiting loop`);
          break;
        }

        if (iteration > 0 && iteration % 8 === 0 && messages.length > MAX_MESSAGES_BEFORE_DIARY) {
          const historyText = messages
            .slice(-20)
            .map((m) => (m.role === "user" ? `User: ${m.content}` : m.role === "assistant" ? `Assistant: ${typeof m.content === "string" ? m.content : "[content]"}` : "Tool: [result]"))
            .join("\n");
          try {
            const summary = await this.compressHistory(
              [{ type: "task", text: initialTaskContext }, { type: "observation", text: historyText }],
              { projectId, projectProvider, projectModel, llmSettings }
            );
            const kept = messages.slice(-DIARY_KEEP_TAIL);
            messages = [
              { role: "user", content: initialTaskContext },
              { role: "user", content: `Summary of previous steps:\n${(summary || "").trim()}\n\nThen the latest messages:` },
              ...kept,
            ];
          } catch (err) {
            this.log(
              "error",
              `Agent Diary (Native FC) failed: ${err instanceof Error ? err.message : String(err)}`
            );
          }
        }

        if (Date.now() - iterationStart > 30000) {
          this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${Date.now() - iterationStart}ms`);
        }
      }
    }

    const durationSec = Math.round((Date.now() - loopStartTime) / 1000);
    const toolCallsSummary = [...toolCallCounts.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([name, count]) => `${name}(${count})`)
      .join(", ") || "none";
    this.log(
      "info",
      `[${this.config.agentRole}] ReAct loop completed:\nTotal iterations: ${lastCompletedIteration}/${MAX_REACT_ITERATIONS}\nParse errors: ${useNativeFC ? 0 : parseRetryCount}\nTool calls: ${toolCallsSummary}\nDuration: ${durationSec}s`
    );

    if (!shouldSubmitForQA) {
      this.log(
        "info",
        `[${this.config.agentRole}] ReAct loop completed without explicit submitForQA; proceeding to verification and QA pipeline`
      );
    } else {
      this.log(
        "info",
        `[${this.config.agentRole}] ReAct loop requested submitForQA; continuing with verification and QA pipeline`
      );
    }

    return {
      results,
      lastBuildPassed,
      shouldSubmitForQA,
      lastSuccessfulCssWrite: lastSuccessfulCssWrite ?? null,
      shouldEscalate,
      lastCompletedIteration,
    };
  }

  private async runVerificationAndHardGate(params: {
    mode: "task" | "ticket";
    projectId: string;
    taskId: string;
    ticketId?: string;
    verificationCriteria: {
      artifacts?: string[];
      automatedCheck?: string;
      manualCheck?: string;
    } | null;
    tools: ReturnType<typeof createAgentTools>;
    parentMessageId: string;
  }): Promise<{
    artifactsOutput: string;
    automatedCheckOutput: string;
    hasVerificationFailures: boolean;
    hardGateFailed: boolean;
    hardGateCompileOutput: string;
    verifiedArtifactPaths: string[];
  }> {
    const {
      mode,
      taskId,
      ticketId,
      verificationCriteria,
      tools,
      parentMessageId,
    } = params;

    let artifactsOutput = "";
    let automatedCheckOutput = "";
    let hasVerificationFailures = false;
    let verifiedArtifactPaths: string[] = [];
    let hardGateFailed = false;
    let hardGateCompileOutput = "";

    const projectDir = getProjectDir(params.projectId);
    let stack: { type: string; buildCommand: string | null } = { type: "unknown", buildCommand: null };
    try {
      stack = await detectStack(projectDir);
    } catch {
      // keep null buildCommand
    }

    let mergeBuildWithVerification = false;
    let buildCommandToRunOnce: string | null = null;

    if (verificationCriteria) {
      const artifacts = Array.isArray(verificationCriteria.artifacts)
        ? verificationCriteria.artifacts
        : [];
      const automatedCheck =
        typeof verificationCriteria.automatedCheck === "string" &&
        verificationCriteria.automatedCheck.trim()
          ? verificationCriteria.automatedCheck.trim()
          : null;

      if (automatedCheck && stack.buildCommand) {
        const normAuto = normalizeBuildCommand(automatedCheck);
        const normBuild = normalizeBuildCommand(stack.buildCommand);
        if (normAuto === normBuild) {
          mergeBuildWithVerification = true;
          buildCommandToRunOnce = stack.buildCommand;
          this.log(
            "info",
            `[${this.config.agentRole}] Verification + Hard Gate: same command (${stack.buildCommand}), will run once.`
          );
        }
      }

      if (artifacts.length > 0 || automatedCheck) {
        this.log(
          "info",
          `[${this.config.agentRole}] 🔍 Verification phase${
            mode === "ticket" ? " (ticket run)" : ""
          }: running automatedCheck and artifact checks...`
        );

        const verificationSteps: Array<{
          command: string;
          reason: string;
          isArtifactCheck?: boolean;
        }> = [];

        let artifactBase = ".";
        if (artifacts.length > 0) {
          if (allArtifactsArePathLike(artifacts)) {
            artifactBase = ".";
          } else {
            const rootCheckResult = await (tools.executeCommand as any).execute({
              command: MANIFEST_AT_ROOT_CHECK_CMD,
              reason: "Check manifest at workspace root",
            });
            if (rootCheckResult?.exitCode === 0) {
              artifactBase = ".";
              this.log("info", "[Verification] Manifest at workspace root; using artifactBase = '.'.");
            } else {
              try {
                const findResult = await (tools.executeCommand as any).execute({
                  command: FIND_MANIFEST_CMD,
                  reason: "Detect project root for artifact checks",
                });
                artifactBase = resolveArtifactBaseFromFindResult(findResult);
                if (isArtifactBaseInBuildDir(artifactBase)) artifactBase = ".";
                if (artifactBase === ".") {
                  this.log("info", "[Verification] No manifest found at depth 1-3 or path invalid; using workspace root for artifact checks.");
                }
              } catch {
                this.log("info", "[Verification] Project root detection failed; using workspace root.");
              }
            }
          }
        }

        const VERIFY_EOF = "AI_VERIFY_SCRIPT_END_7f3a";
        if (automatedCheck && !mergeBuildWithVerification) {
          verificationSteps.push({
            command: `cat << '${VERIFY_EOF}' > .ai-temp-check.sh\n${automatedCheck}\n${VERIFY_EOF}\nsh .ai-temp-check.sh`,
            reason: "Run automatedCheck",
          });
        }

        for (const artifact of artifacts) {
          const path = getVerificationPath(artifactBase, artifact);
          verificationSteps.push({
            command: `ls -la "${path}"`,
            reason: `Verify artifact: ${artifact}`,
            isArtifactCheck: true,
          });
          verificationSteps.push({
            command: `head -n 200 "${path}"`,
            reason: `Artifact content: ${artifact}`,
            isArtifactCheck: true,
          });
        }
        verifiedArtifactPaths = artifacts.map((a) => getVerificationPath(artifactBase, a));

        const artifactOutputs: string[] = [];
        const checkOutputs: string[] = [];

        for (const vstep of verificationSteps) {
          const cmd = vstep.command;
          const reason = vstep.reason;

          this.log(
            "info",
            `[${this.config.agentRole}][Verification] Running: ${cmd}`
          );

          try {
            const execTool = tools.executeCommand;
            const result = await (execTool as any).execute({ command: cmd, reason });
            const out = typeof result?.stdout === "string" ? result.stdout : "";
            const err = typeof result?.stderr === "string" ? result.stderr : "";
            const success = result?.success === true;
            const exitCode = result?.exitCode;

            if (!success || exitCode !== 0) {
              hasVerificationFailures = true;
            }

            const output = [
              `--- ${reason} ---`,
              `$ ${cmd}`,
              `exit ${exitCode ?? "?"}`,
              out,
              err ? `stderr:\n${err}` : "",
            ]
              .filter(Boolean)
              .join("\n");

            if (vstep.isArtifactCheck) {
              artifactOutputs.push(output);
            } else {
              checkOutputs.push(output);
            }
          } catch (verr) {
            hasVerificationFailures = true;
            const output = [
              `--- ${reason} ---`,
              `$ ${cmd}`,
              `FAILED: ${verr instanceof Error ? verr.message : String(verr)}`,
            ].join("\n");

            if (vstep.isArtifactCheck) {
              artifactOutputs.push(output);
            } else {
              checkOutputs.push(output);
            }
          }
        }

        artifactsOutput = artifactOutputs.join("\n\n");
        if (mergeBuildWithVerification && buildCommandToRunOnce) {
          this.log(
            "info",
            `[${this.config.agentRole}][Verification] Running (shared with Hard Gate): ${buildCommandToRunOnce}`
          );
          try {
            const execTool = tools.executeCommand as any;
            const result = await execTool.execute({
              command: buildCommandToRunOnce,
              reason: "Run automatedCheck + Hard Gate (once)",
            });
            const out = typeof result?.stdout === "string" ? result.stdout : "";
            const err = typeof result?.stderr === "string" ? result.stderr : "";
            const success = result?.success === true;
            const exitCode = result?.exitCode;
            if (!success || (typeof exitCode === "number" && exitCode !== 0)) {
              hasVerificationFailures = true;
              hardGateFailed = true;
              const combined = [out, err].filter(Boolean).join("\n");
              const headLen = 2000;
              const tailLen = 2000;
              const body =
                combined.length <= headLen + tailLen
                  ? combined
                  : combined.slice(0, headLen) + "\n[MIDDLE TRUNCATED]\n" + combined.slice(-tailLen);
              hardGateCompileOutput = HARD_GATE_STOP_HEADER + body;
              this.log(
                "error",
                `[${this.config.agentRole}] Hard Gate failed (${buildCommandToRunOnce}): ${hardGateCompileOutput.slice(0, 500)}`
              );
            }
            const output = [
              "--- Run automatedCheck + Hard Gate ---",
              `$ ${buildCommandToRunOnce}`,
              `exit ${exitCode ?? "?"}`,
              out,
              err ? `stderr:\n${err}` : "",
            ]
              .filter(Boolean)
              .join("\n");
            automatedCheckOutput = (checkOutputs.length ? checkOutputs.join("\n\n") + "\n\n" : "") + output;
          } catch (runErr) {
            const runErrMsg = runErr instanceof Error ? runErr.message : String(runErr);
            if (isHardGateInfrastructureError(runErrMsg)) {
              this.log(
                "info",
                `[${this.config.agentRole}] Hard Gate skipped: infrastructure error`
              );
              const output = [
                "--- Run automatedCheck + Hard Gate ---",
                `$ ${buildCommandToRunOnce}`,
                "Skipped: infrastructure error (docker/network).",
              ].join("\n");
              automatedCheckOutput = (checkOutputs.length ? checkOutputs.join("\n\n") + "\n\n" : "") + output;
            } else {
              hasVerificationFailures = true;
              hardGateFailed = true;
              hardGateCompileOutput =
                HARD_GATE_STOP_HEADER + "BUILD EXECUTOR ERROR: " + runErrMsg;
              const output = [
                "--- Run automatedCheck + Hard Gate ---",
                `$ ${buildCommandToRunOnce}`,
                `FAILED: ${hardGateCompileOutput}`,
              ].join("\n");
              automatedCheckOutput = (checkOutputs.length ? checkOutputs.join("\n\n") + "\n\n" : "") + output;
            }
          }
        } else {
          automatedCheckOutput = checkOutputs.join("\n\n");
        }
      }
    }

    if (!mergeBuildWithVerification) {
      try {
        this.log("info", `[${this.config.agentRole}] Detected stack: ${stack.type}`);
        let hardGateCommand: string | null = stack.buildCommand;
        if (!hardGateCommand && stack.type !== "unknown") {
          hardGateCommand = await getFallbackHardGateCommand(stack.type, projectDir);
        }

        if (hardGateCommand) {
          const cleanCommand = hardGateCommand.replace(/2>&1/g, "").trim();
          this.log(
            "info",
            `[${this.config.agentRole}] Hard Gate${
              mode === "ticket" ? " (ticket)" : ""
            }: running ${cleanCommand}...`
          );
          const execTool = tools.executeCommand as any;
          const gateResult = await execTool.execute({
            command: cleanCommand,
            reason:
              mode === "ticket"
                ? "Hard Gate: Pre-QA compile check (ticket run)"
                : "Hard Gate: Pre-QA compile check",
          });
          const exitCode =
            typeof gateResult?.exitCode === "number" ? gateResult.exitCode : undefined;
          const success = gateResult?.success === true;
          const stdout =
            typeof gateResult?.stdout === "string" ? gateResult.stdout : "";
          const stderr =
            typeof gateResult?.stderr === "string" ? gateResult.stderr : "";
          if (!success || (typeof exitCode === "number" && exitCode !== 0)) {
            hardGateFailed = true;
            const combined = [stdout, stderr].filter(Boolean).join("\n");
            const headLen = 2000;
            const tailLen = 2000;
            const body =
              combined.length <= headLen + tailLen
                ? combined
                : combined.slice(0, headLen) + "\n[MIDDLE TRUNCATED]\n" + combined.slice(-tailLen);
            hardGateCompileOutput = HARD_GATE_STOP_HEADER + body;
            const truncated = hardGateCompileOutput;
            const label =
              mode === "ticket"
                ? `Hard Gate failed (ticket, ${cleanCommand})`
                : `Hard Gate failed (${cleanCommand})`;
            this.log(
              "error",
              `[${this.config.agentRole}] ${label}: ${truncated}`
            );
          }
        } else {
          this.log(
            "info",
            `[${this.config.agentRole}] Hard Gate${
              mode === "ticket" ? " (ticket)" : ""
            } skipped (no build command for stack: ${stack.type})`
          );
        }
      } catch (gateErr) {
        this.log(
          "info",
          `[${this.config.agentRole}] Hard Gate${
            mode === "ticket" ? " (ticket)" : ""
          } skipped: ${gateErr instanceof Error ? gateErr.message : String(gateErr)}`
        );
      }
    }

    return {
      artifactsOutput,
      automatedCheckOutput,
      hasVerificationFailures,
      hardGateFailed,
      hardGateCompileOutput,
      verifiedArtifactPaths,
    };
  }

  private async handleTaskRequest(message: AgentMessage): Promise<Record<string, unknown>> {
    const { taskId } = message.payload as { taskId: string };

    this.log("info", `[${this.config.agentRole}] Processing task: ${taskId}`);

    try {
      const task = await prisma.task.findUnique({
        where: { id: taskId },
        include: {
          plan: {
            include: { project: true },
          },
        },
      });

      if (!task) {
        throw new Error(`Task ${taskId} not found`);
      }

      if (!task.plan?.project) {
        throw new Error(`Task plan has no project`);
      }

      this.log(
        "info",
        `\n👷 [TASK_EXECUTOR] -> @TEAM: "Принял задачу '${task.title}'. Ушел генерировать план и кодить."`
      );

      let instructions = task.generatedPrompt?.trim();

      if (!instructions) {
        this.log(
          "info",
          `[${this.config.agentRole}] Generating detailed prompt for task ${taskId}...`
        );
        instructions = await generateTaskPrompt(taskId, false, false);
      }

      const project = task.plan.project;
      const projectId = project.id;

      const projectContext = await getCompactProjectContext(projectId);

      const tools = createAgentTools(
        projectId,
        this.config.sessionId,
        this.config.mode ?? "local",
        this.log.bind(this)
      );

      let codeMapSection = "Code map unavailable or empty for this project.";
      try {
        const codeMapResult = await (tools.getCodeMap as any).execute({});
        if (codeMapResult) {
          if (typeof codeMapResult.codeMap === "string" && codeMapResult.codeMap.trim()) {
            codeMapSection = codeMapResult.codeMap;
          } else if (typeof codeMapResult.error === "string" && codeMapResult.error.trim()) {
            codeMapSection = `Code map unavailable: ${codeMapResult.error}`;
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        codeMapSection = `Code map unavailable due to error: ${msg}`;
      }

      const workDir =
        this.config.mode === "cloud"
          ? CONTAINER_WORKSPACE_ROOT
          : getProjectDir(projectId);
      const workspaceStateBlock = [
        "# Current Workspace State",
        `Working Directory: ${workDir}`,
        "",
        "File Tree / Code Map:",
        "All paths below are relative to the project root (Working Directory).",
        codeMapSection,
      ].join("\n");

      const vcForContext = (task as any).verificationCriteria as
        | {
            artifacts?: string[];
            automatedCheck?: string;
            manualCheck?: string;
          }
        | null
        | undefined;

      const acceptanceLines = vcForContext
        ? [
            "",
            "Acceptance criteria (what QA will check):",
            JSON.stringify(vcForContext, null, 2),
          ]
        : [];

      const projectTypeLine =
        (task.plan as any)?.projectType && typeof (task.plan as any).projectType === "string"
          ? [`Project type: ${(task.plan as any).projectType}`]
          : [];

      const initialTaskContext = [
        workspaceStateBlock,
        "",
        "---",
        "",
        `Task title: ${task.title}`,
        `Task description: ${task.description ?? "(no description)"}`,
        "",
        "Instructions (from Planner):",
        instructions,
        ...acceptanceLines,
        ...projectTypeLine,
        "",
        "Project context:",
        projectContext,
      ]
        .join("\n")
        .trim();

      let historySegments: ReActHistorySegment[] = [
        {
          type: "task",
          text: initialTaskContext,
        },
      ];

      const results: any[] = [];
      const commandFailureMap = new Map<
        string,
        { lastLog: string; count: number }
      >();
      let shouldSubmitForQA = false;
      let lastBuildPassed = true;
      let parseRetryCount = 0;
      let consecutiveFailedIterations = 0;
      let stuckEscalationTriggered = false;
      let ticketInitializedProjectContext: string | null = null;
      let lastSuccessfulCssWrite: { filePath: string; content: string } | null = null;

      const ticketLoopStartTime = Date.now();
      const ticketToolCallCounts = new Map<string, number>();
      let ticketLastCompletedIteration = 0;
      const ticketResolvedProvider = resolveProvider(project.aiProvider || undefined);
      const ticketUseNativeFC = true;

      if (!ticketUseNativeFC) {
      let ticketIterationWarningShown = false;
      for (let iteration = 1; iteration <= MAX_REACT_ITERATIONS; iteration++) {
        const ticketIterationStart = Date.now();
        ticketLastCompletedIteration = iteration;

        const historyText = buildHistoryText(historySegments);

        this.log(
          "info",
          `[${this.config.agentRole}] ReAct iteration ${iteration}/${MAX_REACT_ITERATIONS} (history length=${historyText.length})`
        );

        if (!ticketIterationWarningShown && iteration === MAX_REACT_ITERATIONS - REACT_ITERATIONS_WARN_THRESHOLD) {
          ticketIterationWarningShown = true;
          historySegments.push({
            type: "observation",
            text: JSON.stringify({
              system: "ITERATION_WARNING",
              message: REACT_ITERATION_WARNING_MESSAGE,
              iterationsRemaining: REACT_ITERATIONS_WARN_THRESHOLD,
            }),
          });
        }

        const step = await this.getNextReActStep(
          historyText,
          projectId,
          project.aiProvider || undefined,
          project.aiModel || undefined,
          ticketInitializedProjectContext
        );

        this.log("info", `[${this.config.agentRole}] 🧠 ${step.thought}`);
        historySegments.push({ type: "thought", text: step.thought });

        if (step.action === "FINISH") {
          if (!lastBuildPassed) {
            this.log(
              "info",
              `[${this.config.agentRole}] Agent attempted to FINISH with a broken build/type-check. Intercepting.`
            );
            const enforcementObservation = {
              system: "FINISH_INTERCEPT",
              message:
                "SYSTEM ENFORCEMENT: You attempted to FINISH, but your last build/type-check command failed. You are TRAPPED in this loop. You MUST read the compiler errors, use writeFile or executeCommand to fix the code, and run the build command again successfully before finishing.",
              lastBuildPassed,
            };
            historySegments.push({
              type: "observation",
              text: JSON.stringify(enforcementObservation),
            });
            historySegments = pruneHistorySegments(historySegments);
            const _tElapsed0 = Date.now() - ticketIterationStart;
            if (_tElapsed0 > 30000) {
              this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_tElapsed0}ms`);
            }
            continue;
          }
          const _tElapsed1 = Date.now() - ticketIterationStart;
          if (_tElapsed1 > 30000) {
            this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_tElapsed1}ms`);
          }
          this.log(
            "info",
            `[${this.config.agentRole}] Received FINISH signal from ReAct loop`
          );
          historySegments.push({ type: "action", text: "FINISH" });
          break;
        }

        const { toolName, params = {} } = step.action;

        if (toolName === "__RETRY_PARSE__") {
          if (parseRetryCount >= 2) {
            const _tElapsed2 = Date.now() - ticketIterationStart;
            if (_tElapsed2 > 30000) {
              this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_tElapsed2}ms`);
            }
            this.log(
              "error",
              `[${this.config.agentRole}] Max ReAct JSON parse retries reached; finishing to avoid infinite loop.`
            );
            const observation = {
              system: "PARSE_RETRY_LIMIT",
              message:
                "SYSTEM: Your responses repeatedly contained invalid or truncated JSON. The executor is finishing this task to avoid an infinite loop.",
              parseRetryCount,
            };
            historySegments.push({
              type: "observation",
              text: JSON.stringify(observation),
            });
            historySegments = pruneHistorySegments(historySegments);
            break;
          }

          parseRetryCount += 1;
          let retryMessage = RETRY_PARSE_MESSAGE;
          if (parseRetryCount >= 2) {
            this.log(
              "warn",
              `[${this.config.agentRole}] Multiple parse errors, agent may be stuck`
            );
            retryMessage += " You seem stuck. Try a different approach or simpler action.";
          }
          const observation = {
            system: "RETRY_PARSE",
            message: retryMessage,
            parseRetryCount,
          };
          historySegments.push({
            type: "observation",
            text: JSON.stringify(observation),
          });
          historySegments = pruneHistorySegments(historySegments);
          const _tElapsed3 = Date.now() - ticketIterationStart;
          if (_tElapsed3 > 30000) {
            this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_tElapsed3}ms`);
          }
          continue;
        }

        const isExecTool =
          toolName === "executeCommand" || toolName === "cloudExecuteCommand";
        const commandParam =
          isExecTool && typeof (params as any)?.command === "string"
            ? (params as any).command
            : undefined;
        const actionSummary = JSON.stringify({ toolName, params }, null, 2).slice(0, 1000);
        historySegments.push({ type: "action", text: actionSummary });

        if (toolName === "submitForQA") {
          shouldSubmitForQA = true;
          const _tElapsed5 = Date.now() - ticketIterationStart;
          if (_tElapsed5 > 30000) {
            this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_tElapsed5}ms`);
          }
          this.log(
            "info",
            `[${this.config.agentRole}] Logical submitForQA action received from ReAct loop`
          );
          const observation = {
            toolName,
            status: "queuedForQA",
            note: "ReAct agent indicated readiness for QA. Proceeding to verification and QA pipeline.",
          };
          historySegments.push({
            type: "observation",
            text: JSON.stringify(observation),
          });
          break;
        }

        if (!(toolName in tools)) {
          if (toolName === "escalateToTeamLead") {
            const _tElapsedEscUnknown = Date.now() - ticketIterationStart;
            if (_tElapsedEscUnknown > 30000) {
              this.log(
                "warn",
                `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_tElapsedEscUnknown}ms`
              );
            }
            continue;
          }
          const errorMsg = `Unknown toolName requested by ReAct agent: ${toolName}`;
          this.log("error", `[${this.config.agentRole}] ${errorMsg}`);
          const observation = {
            toolName,
            success: false,
            error: errorMsg,
          };
          historySegments.push({
            type: "observation",
            text: JSON.stringify(observation),
          });
          historySegments = pruneHistorySegments(historySegments);
          const _tElapsed6 = Date.now() - ticketIterationStart;
          if (_tElapsed6 > 30000) {
            this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_tElapsed6}ms`);
          }
          continue;
        }

        const commandSuffix = commandParam ? `: ${commandParam}` : "";

        this.log(
          "info",
          `[${this.config.agentRole}] 🛠️ Executing: ${toolName}${commandSuffix}`
        );

        ticketToolCallCounts.set(toolName, (ticketToolCallCounts.get(toolName) ?? 0) + 1);

        let iterationSucceeded = false;

        try {
          const tool = tools[toolName as keyof typeof tools];
          const result = await (tool as any).execute(params);

          const anyResult = result as any;
          const rawExitCode =
            typeof anyResult?.exitCode === "number" ? anyResult.exitCode : undefined;
          const explicitSuccess =
            typeof anyResult?.success === "boolean" ? anyResult.success : undefined;

          let normalizedSuccess = true;
          if (explicitSuccess !== undefined || rawExitCode !== undefined) {
            normalizedSuccess =
              explicitSuccess !== false && (rawExitCode == null || rawExitCode === 0);
          }

          if (isExecTool && commandParam) {
            const lowerCmd = commandParam.toLowerCase();
            if (
              lowerCmd.includes("build") ||
              lowerCmd.includes("tsc") ||
              lowerCmd.includes("type-check") ||
              lowerCmd.includes("typecheck")
            ) {
              lastBuildPassed = normalizedSuccess;
            }
          }

          let repeatedFailureNote: string | undefined;

          if (!normalizedSuccess && isExecTool && commandParam) {
            const normalizedLog =
              (typeof anyResult?.stderr === "string" && anyResult.stderr.trim()) ||
              (typeof anyResult?.error === "string" && anyResult.error.trim()) ||
              (typeof anyResult?.stdout === "string" && anyResult.stdout.trim()) ||
              "";

            if (normalizedLog) {
              const existing = commandFailureMap.get(commandParam);
              if (existing && existing.lastLog === normalizedLog) {
                const newCount = existing.count + 1;
                commandFailureMap.set(commandParam, {
                  lastLog: normalizedLog,
                  count: newCount,
                });
                if (newCount >= 3) {
                  repeatedFailureNote = `This command has already failed ${newCount} times with essentially the same logs. You must change the code or configuration before running it again; do not just retry the command without fixes.`;
                }
              } else {
                commandFailureMap.set(commandParam, {
                  lastLog: normalizedLog,
                  count: 1,
                });
              }
            }
          }

          if (normalizedSuccess) {
            iterationSucceeded = true;
            results.push({ toolName, result, success: true });
            this.log(
              "success",
              `[${this.config.agentRole}] ✅ ${toolName} completed`
            );
          } else {
            const stderr =
              typeof anyResult?.stderr === "string" ? anyResult.stderr : "";
            const errorField =
              typeof anyResult?.error === "string" ? anyResult.error : "";
            const snippet = (errorField || stderr).slice(0, 500);
            const exitCodeInfo =
              rawExitCode != null ? ` (exitCode=${rawExitCode})` : "";

            this.log(
              "error",
              `[${this.config.agentRole}] ❌ Tool ${toolName} reported failure${exitCodeInfo}${
                snippet ? `: ${snippet}` : ""
              }`
            );

            results.push({ toolName, result, success: false });
          }

          const stepParams = params as Record<string, unknown>;
          if (
            normalizedSuccess &&
            toolName === "writeFile" &&
            typeof stepParams.filePath === "string"
          ) {
            const filePath = stepParams.filePath;
            if (
              filePath.endsWith(".css") ||
              filePath.endsWith(".scss") ||
              filePath.endsWith(".module.css")
            ) {
              lastSuccessfulCssWrite = {
                filePath,
                content:
                  typeof stepParams.content === "string" ? stepParams.content : "",
              };
            }
          }

          const fullStdout =
            typeof anyResult?.stdout === "string" ? anyResult.stdout : "";
          const fullStderr =
            typeof anyResult?.stderr === "string" ? anyResult.stderr : "";
          const fullError =
            typeof anyResult?.error === "string" ? anyResult.error : "";

          const logSourceBase = [fullStdout, fullStderr].filter(Boolean).join("\n");
          const primaryLogSource = logSourceBase || fullError;

          const maxLogChars = isExecTool ? 4000 : 1000;
          let logSnippet: string | undefined;
          if (primaryLogSource) {
            logSnippet =
              primaryLogSource.length <= maxLogChars
                ? primaryLogSource
                : primaryLogSource.slice(-maxLogChars);
          }

          const observation: {
            toolName: string;
            success: boolean;
            exitCode?: number;
            stdoutSnippet?: string;
            stderrSnippet?: string;
            errorSnippet?: string;
            logSnippet?: string;
            repeatedFailureNote?: string;
            projectPath?: string;
            projectName?: string;
            isRootProject?: boolean;
            message?: string;
          } = {
            toolName,
            success: normalizedSuccess,
            exitCode: rawExitCode,
            stdoutSnippet: fullStdout ? fullStdout.slice(0, 500) : undefined,
            stderrSnippet: fullStderr ? fullStderr.slice(0, 500) : undefined,
            errorSnippet: fullError ? fullError.slice(0, 500) : undefined,
            logSnippet,
          };

          if (repeatedFailureNote) {
            observation.repeatedFailureNote = repeatedFailureNote;
          }

          historySegments.push({
            type: "observation",
            text: JSON.stringify(observation),
          });
        } catch (toolErr) {
          const errMsg = toolErr instanceof Error ? toolErr.message : String(toolErr);
          this.log("error", `[${this.config.agentRole}] ${toolName} failed: ${errMsg}`);
          results.push({ toolName, error: errMsg, success: false });

          if (isExecTool && commandParam) {
            const lowerCmd = commandParam.toLowerCase();
            if (
              lowerCmd.includes("build") ||
              lowerCmd.includes("tsc") ||
              lowerCmd.includes("type-check") ||
              lowerCmd.includes("typecheck")
            ) {
              lastBuildPassed = false;
            }
          }

          const observation = {
            toolName,
            success: false,
            errorSnippet: errMsg.slice(0, 500),
          };
          historySegments.push({
            type: "observation",
            text: JSON.stringify(observation),
          });
        }

      if (iterationSucceeded) {
        consecutiveFailedIterations = 0;
      } else {
        consecutiveFailedIterations += 1;
      }

      if (
        !stuckEscalationTriggered &&
        (iteration >= 20 && consecutiveFailedIterations >= 3)
      ) {
          const stuckObservation = {
            system: "STUCK_LOOP",
            message:
              "SYSTEM: You have had multiple iterations without a successful tool call. You appear stuck. Summarize what you tried and what is blocking you, then either make a concrete fix using writeFile or executeCommand, or FINISH to hand off.",
            iteration,
            consecutiveFailedIterations,
          };
          historySegments.push({
            type: "observation",
            text: JSON.stringify(stuckObservation),
          });
          stuckEscalationTriggered = true;
        }

        const _tElapsedEnd = Date.now() - ticketIterationStart;
        if (_tElapsedEnd > 30000) {
          this.log("warn", `[${this.config.agentRole}] Slow iteration #${iteration}: took ${_tElapsedEnd}ms`);
        }
        historySegments = pruneHistorySegments(historySegments);
      }
      } else {
        const ticketResolvedModel = project.aiModel || "gpt-4o-mini";
        const ticketReactMaxTokens = Math.min((await getLLMSettings()).maxTokens ?? 16384, 16384);
        const ticketSystemPromptFC = buildSystemPromptForNativeFC(ticketInitializedProjectContext);
        let ticketMessages: CoreMessage[] = [{ role: "user", content: initialTaskContext }];
        const ticketLlmSettings = await getLLMSettings();
        let ticketLastActionKey: string | null = null;
        let ticketSameActionCount = 0;
        let ticketIterationWarningShownFC = false;

        for (let iteration = 1; iteration <= MAX_REACT_ITERATIONS; iteration++) {
          const ticketIterationStart = Date.now();
          ticketLastCompletedIteration = iteration;
          this.log(
            "info",
            `[${this.config.agentRole}] ReAct iteration ${iteration}/${MAX_REACT_ITERATIONS} (ticket, messages=${ticketMessages.length})`
          );

          if (!ticketIterationWarningShownFC && iteration === MAX_REACT_ITERATIONS - REACT_ITERATIONS_WARN_THRESHOLD) {
            ticketIterationWarningShownFC = true;
            ticketMessages.push({ role: "user", content: REACT_ITERATION_WARNING_MESSAGE });
          }

          let ticketAiResult;
          try {
            ticketAiResult = await generateText({
              model: getModel(ticketResolvedProvider, ticketResolvedModel),
              system: ticketSystemPromptFC,
              messages: ticketMessages,
              tools,
              maxSteps: 1,
              temperature: ticketLlmSettings.temperature,
              maxTokens: ticketReactMaxTokens,
              abortSignal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
            });
          } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            this.log("error", `[${this.config.agentRole}] Ticket Native FC generateText failed: ${msg}`);
            throw new Error(`Ticket Native FC generateText failed: ${msg}`);
          }

          const ticketStep = ticketAiResult.steps?.[0];
          const ticketResponseMessages = ticketAiResult.response?.messages ?? [];
          const ticketThoughtForLog = (ticketStep?.text ?? ticketAiResult.text ?? "").trim() || "(tool call)";
          this.log("info", `[${this.config.agentRole}] 🧠 ${ticketThoughtForLog}`);

          const ticketHasToolCalls = ticketStep && Array.isArray(ticketStep.toolCalls) && ticketStep.toolCalls.length > 0;
          const ticketFinishText = (ticketStep?.text ?? ticketAiResult.text ?? "").trim();

          if (!ticketStep && !ticketFinishText && ticketResponseMessages.length === 0) {
            continue;
          }

          if (!ticketHasToolCalls && ticketFinishText) {
            if (!lastBuildPassed) {
              ticketMessages.push({
                role: "user",
                content:
                  "SYSTEM ENFORCEMENT: You attempted to finish, but your last build/type-check command failed. Fix the errors and run the build again successfully before finishing.",
              });
              continue;
            }
            break;
          }

          if (!ticketHasToolCalls) {
            if (ticketResponseMessages.length > 0) ticketMessages.push(...ticketResponseMessages);
            continue;
          }

          const ticketToolCalls = ticketStep!.toolCalls as Array<{ toolName: string; args: Record<string, unknown> }>;
          const ticketToolResults = ticketStep!.toolResults as Array<unknown>;

          const ticketCmdLogMax = 120;
          for (const tc of ticketToolCalls) {
            const toolName = tc?.toolName ?? (tc as { name?: string })?.name ?? "unknown";
            const params = tc?.args ?? {};
            if (toolName === "executeCommand" && typeof params.command === "string") {
              const cmd = params.command as string;
              const suffix = cmd.length > ticketCmdLogMax ? cmd.slice(0, ticketCmdLogMax) + "..." : cmd;
              this.log("info", `[${this.config.agentRole}] 🛠️ Executing: ${toolName}: ${suffix}`);
            } else if (toolName === "writeFile" && (params.filePath ?? params.path)) {
              this.log("info", `[${this.config.agentRole}] 🛠️ Executing: ${toolName}: ${params.filePath ?? params.path}`);
            } else if (toolName === "readFile" && (params.filePath ?? params.path)) {
              this.log("info", `[${this.config.agentRole}] 🛠️ Executing: ${toolName}: ${params.filePath ?? params.path}`);
            } else {
              this.log("info", `[${this.config.agentRole}] 🛠️ Executing: ${toolName}`);
            }
          }

          for (let i = 0; i < ticketToolCalls.length; i++) {
            const tc = ticketToolCalls[i];
            const name = tc?.toolName ?? "";
            if (name === "submitForQA") shouldSubmitForQA = true;
            ticketToolCallCounts.set(name, (ticketToolCallCounts.get(name) ?? 0) + 1);
          }

          const ticketActionKey = `${ticketToolCalls.map((tc) => tc.toolName).join(",")}:${JSON.stringify(ticketToolCalls.map((tc) => tc.args))}`;
          if (ticketActionKey === ticketLastActionKey) {
            ticketSameActionCount += 1;
          } else {
            ticketLastActionKey = ticketActionKey;
            ticketSameActionCount = 1;
          }
          if (ticketSameActionCount >= 3) {
            ticketMessages.push(...ticketResponseMessages);
            ticketMessages.push({
              role: "user",
              content: "ERROR: You are stuck repeating the same action. Change your approach or fix the code.",
            });
            continue;
          }

          for (let i = 0; i < ticketToolResults.length; i++) {
            const tr = ticketToolResults[i] as Record<string, unknown> | undefined;
            const tc = ticketToolCalls[i];
            const name = tc?.toolName ?? "";
            if (!tr) continue;
            const explicitSuccess = typeof tr.success === "boolean" ? tr.success : undefined;
            const rawExitCode = typeof tr.exitCode === "number" ? tr.exitCode : undefined;
            const normalizedSuccess = explicitSuccess !== false && (rawExitCode == null || rawExitCode === 0);
            const cmd = typeof tc?.args?.command === "string" ? tc.args.command : "";
            if ((name === "executeCommand" || name === "cloudExecuteCommand") && cmd) {
              const lower = cmd.toLowerCase();
              if (lower.includes("build") || lower.includes("tsc") || lower.includes("type-check") || lower.includes("typecheck")) {
                lastBuildPassed = normalizedSuccess;
              }
            }
            results.push({ toolName: name, result: tr, success: normalizedSuccess });
          }

          ticketMessages.push(...ticketResponseMessages);
          if (shouldSubmitForQA) break;
          if (Date.now() - ticketIterationStart > 30000) {
            this.log("warn", `[${this.config.agentRole}] Slow ticket iteration #${iteration}: took ${Date.now() - ticketIterationStart}ms`);
          }
        }
      }

      const ticketDurationSec = Math.round((Date.now() - ticketLoopStartTime) / 1000);
      const ticketToolCallsSummary = [...ticketToolCallCounts.entries()]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([name, count]) => `${name}(${count})`)
        .join(", ") || "none";
      this.log(
        "info",
        `[${this.config.agentRole}] ReAct loop completed:\nTotal iterations: ${ticketLastCompletedIteration}/${MAX_REACT_ITERATIONS}\nParse errors: ${ticketUseNativeFC ? 0 : parseRetryCount}\nTool calls: ${ticketToolCallsSummary}\nDuration: ${ticketDurationSec}s`
      );

      if (!shouldSubmitForQA) {
        this.log(
          "info",
          `[${this.config.agentRole}] ReAct loop completed without explicit submitForQA; proceeding to verification and QA pipeline`
        );
      } else {
        this.log(
          "info",
          `[${this.config.agentRole}] ReAct loop requested submitForQA; continuing with verification and QA pipeline`
        );
      }

      const projectDir = getProjectDir(projectId);
      if (lastSuccessfulCssWrite && (await isWebProject(projectDir))) {
        try {
          this.log(
            "info",
            `[${this.config.agentRole}] 🎨 Running CSS Agent synchronously before QA`
          );
          const cssAgent = createAgentForRole(
            AgentRole.CSS,
            this.config.sessionId ?? "",
            projectId,
            false,
            this.log.bind(this),
            this.config.mode ?? "local"
          );
          const styleRequestMessage = {
            id: "sync-css-request",
            eventType: MessageType.STYLE_REQUEST,
            payload: {
              taskId,
              filePath: lastSuccessfulCssWrite.filePath,
              content: lastSuccessfulCssWrite.content,
            },
            replyToId: message.id,
          } as unknown as AgentMessage;
          const cssResult = (await cssAgent.processMessage(styleRequestMessage)) as {
            status?: string;
            improvedContent?: string;
            suggestions?: string[];
          };
          if (
            cssResult?.status === "NEEDS_IMPROVEMENT" &&
            typeof cssResult?.improvedContent === "string" &&
            cssResult.improvedContent.trim()
          ) {
            await this.applyCssImprovedContent({
              projectId,
              filePath: lastSuccessfulCssWrite.filePath,
              improvedContent: cssResult.improvedContent,
            });
          }
        } catch (cssErr) {
          this.log(
            "error",
            `[${this.config.agentRole}] Sync CSS agent failed: ${
              cssErr instanceof Error ? cssErr.message : String(cssErr)
            }`
          );
        }
      } else if (lastSuccessfulCssWrite) {
        this.log("info", `[${this.config.agentRole}] [CSS] Skipped - not a web project`);
      }

      // --- Verification phase: run automatedCheck and ls -la for artifacts, if verificationCriteria is present ---
      const vc = task.verificationCriteria as {
        artifacts?: string[];
        automatedCheck?: string;
        manualCheck?: string;
      } | null;

      let artifactsOutput = "";
      let automatedCheckOutput = "";
      let hasVerificationFailures = false;
      let verifiedArtifactPaths: string[] = [];
      let hardGateFailed = false;
      let hardGateCompileOutput = "";
      let mergeBuildWithVerification = false;
      let vcStack: { type: string; buildCommand: string | null } = { type: "unknown", buildCommand: null };

      if (vc) {
        const artifacts = Array.isArray(vc.artifacts) ? vc.artifacts : [];
        const automatedCheck =
          typeof vc.automatedCheck === "string" && vc.automatedCheck.trim()
            ? vc.automatedCheck.trim()
            : null;

        const vcProjectDir = getProjectDir(projectId);
        try {
          vcStack = await detectStack(vcProjectDir);
        } catch {
          // keep null
        }
        let buildCommandToRunOnce: string | null = null;
        if (automatedCheck && vcStack.buildCommand) {
          const normAuto = normalizeBuildCommand(automatedCheck);
          const normBuild = normalizeBuildCommand(vcStack.buildCommand);
          if (normAuto === normBuild) {
            mergeBuildWithVerification = true;
            buildCommandToRunOnce = vcStack.buildCommand;
            this.log(
              "info",
              `[${this.config.agentRole}] Verification + Hard Gate: same command (${vcStack.buildCommand}), will run once.`
            );
          }
        }

        if (artifacts.length > 0 || automatedCheck) {
          this.log(
            "info",
            `[${this.config.agentRole}] 🔍 Verification phase: running automatedCheck and artifact checks...`
          );

          const verificationSteps: Array<{
            command: string;
            reason: string;
            isArtifactCheck?: boolean;
          }> = [];

          // Dynamic project root: prefer manifest at workspace root (artifactBase = "."); else detect via find
          let artifactBase = ".";
          if (artifacts.length > 0) {
            if (allArtifactsArePathLike(artifacts)) {
              artifactBase = ".";
            } else {
              const rootCheckResult = await (tools.executeCommand as any).execute({
                command: MANIFEST_AT_ROOT_CHECK_CMD,
                reason: "Check manifest at workspace root",
              });
              if (rootCheckResult?.exitCode === 0) {
                artifactBase = ".";
                this.log("info", "[Verification] Manifest at workspace root; using artifactBase = '.'.");
              } else {
                try {
                  const findResult = await (tools.executeCommand as any).execute({
                    command: FIND_MANIFEST_CMD,
                    reason: "Detect project root for artifact checks",
                  });
                  artifactBase = resolveArtifactBaseFromFindResult(findResult);
                  if (isArtifactBaseInBuildDir(artifactBase)) artifactBase = ".";
                  if (artifactBase === ".") {
                    this.log("info", "[Verification] No manifest found at depth 1-3 or path invalid; using workspace root for artifact checks.");
                  }
                } catch {
                  this.log("info", "[Verification] Project root detection failed; using workspace root.");
                }
              }
            }
          }

          // FIX 2: Run automatedCheck via script to avoid shell escaping (quotes, !, etc.) — skip if same as Hard Gate (run once below)
          const VERIFY_EOF = "AI_VERIFY_SCRIPT_END_7f3a";
          if (automatedCheck && !mergeBuildWithVerification) {
            verificationSteps.push({
              command: `cat << '${VERIFY_EOF}' > .ai-temp-check.sh\n${automatedCheck}\n${VERIFY_EOF}\nsh .ai-temp-check.sh`,
              reason: "Run automatedCheck",
            });
          }

          for (const artifact of artifacts) {
            const path = getVerificationPath(artifactBase, artifact);
            verificationSteps.push({
              command: `ls -la "${path}"`,
              reason: `Verify artifact: ${artifact}`,
              isArtifactCheck: true,
            });
            // FIX 1: Show file content so QA can verify required text (limit lines)
            verificationSteps.push({
              command: `head -n 200 "${path}"`,
              reason: `Artifact content: ${artifact}`,
              isArtifactCheck: true,
            });
          }
          verifiedArtifactPaths = artifacts.map((a) => getVerificationPath(artifactBase, a));

          const artifactOutputs: string[] = [];
          const checkOutputs: string[] = [];

          for (const vstep of verificationSteps) {
            const cmd = vstep.command;
            const reason = vstep.reason;

            this.log("info", `[${this.config.agentRole}][Verification] Running: ${cmd}`);

            try {
              const execTool = tools.executeCommand;
              const result = await (execTool as any).execute({ command: cmd, reason });
              const out = typeof result?.stdout === "string" ? result.stdout : "";
              const err = typeof result?.stderr === "string" ? result.stderr : "";
              const success = result?.success === true;
              const exitCode = result?.exitCode;

              if (!success || exitCode !== 0) {
                hasVerificationFailures = true;
              }

              const output = [
                `--- ${reason} ---`,
                `$ ${cmd}`,
                `exit ${exitCode ?? "?"}`,
                out,
                err ? `stderr:\n${err}` : "",
              ]
                .filter(Boolean)
                .join("\n");

              if (vstep.isArtifactCheck) {
                artifactOutputs.push(output);
              } else {
                checkOutputs.push(output);
              }
            } catch (verr) {
              hasVerificationFailures = true;
              const output = [
                `--- ${reason} ---`,
                `$ ${cmd}`,
                `FAILED: ${verr instanceof Error ? verr.message : String(verr)}`,
              ].join("\n");

              if (vstep.isArtifactCheck) {
                artifactOutputs.push(output);
              } else {
                checkOutputs.push(output);
              }
            }
          }

          artifactsOutput = artifactOutputs.join("\n\n");
          if (mergeBuildWithVerification && buildCommandToRunOnce) {
            this.log(
              "info",
              `[${this.config.agentRole}][Verification] Running (shared with Hard Gate): ${buildCommandToRunOnce}`
            );
            try {
              const execTool = tools.executeCommand as any;
              const result = await execTool.execute({
                command: buildCommandToRunOnce,
                reason: "Run automatedCheck + Hard Gate (once)",
              });
              const out = typeof result?.stdout === "string" ? result.stdout : "";
              const err = typeof result?.stderr === "string" ? result.stderr : "";
              const success = result?.success === true;
              const exitCode = result?.exitCode;
              if (!success || (typeof exitCode === "number" && exitCode !== 0)) {
                hasVerificationFailures = true;
                hardGateFailed = true;
                const combined = [out, err].filter(Boolean).join("\n");
                const headLen = 2000;
                const tailLen = 2000;
                hardGateCompileOutput =
                  combined.length <= headLen + tailLen
                    ? combined
                    : combined.slice(0, headLen) + "\n[MIDDLE TRUNCATED]\n" + combined.slice(-tailLen);
                this.log(
                  "error",
                  `[${this.config.agentRole}] Hard Gate failed (${buildCommandToRunOnce}): ${hardGateCompileOutput.slice(0, 500)}`
                );
              }
              const output = [
                "--- Run automatedCheck + Hard Gate ---",
                `$ ${buildCommandToRunOnce}`,
                `exit ${exitCode ?? "?"}`,
                out,
                err ? `stderr:\n${err}` : "",
              ]
                .filter(Boolean)
                .join("\n");
              automatedCheckOutput = (checkOutputs.length ? checkOutputs.join("\n\n") + "\n\n" : "") + output;
            } catch (runErr) {
              const runErrMsg = runErr instanceof Error ? runErr.message : String(runErr);
              if (isHardGateInfrastructureError(runErrMsg)) {
                this.log(
                  "info",
                  `[${this.config.agentRole}] Hard Gate skipped: infrastructure error`
                );
                const output = [
                  "--- Run automatedCheck + Hard Gate ---",
                  `$ ${buildCommandToRunOnce}`,
                  "Skipped: infrastructure error (docker/network).",
                ].join("\n");
                automatedCheckOutput = (checkOutputs.length ? checkOutputs.join("\n\n") + "\n\n" : "") + output;
              } else {
                hasVerificationFailures = true;
                hardGateFailed = true;
                hardGateCompileOutput = "BUILD EXECUTOR ERROR: " + runErrMsg;
                const output = [
                  "--- Run automatedCheck + Hard Gate ---",
                  `$ ${buildCommandToRunOnce}`,
                  `FAILED: ${hardGateCompileOutput}`,
                ].join("\n");
                automatedCheckOutput = (checkOutputs.length ? checkOutputs.join("\n\n") + "\n\n" : "") + output;
              }
            }
          } else {
            automatedCheckOutput = checkOutputs.join("\n\n");
          }
        }
      }

      // --- Hard Gate: Pre-QA compile check (type-check or build) — skipped when already run with verification ---
      if (!mergeBuildWithVerification) {
        try {
          const projectDirForGate = getProjectDir(projectId);
          const stack = vc ? vcStack : await detectStack(projectDirForGate);
          this.log("info", `[${this.config.agentRole}] Detected stack: ${stack.type}`);
          let hardGateCommand: string | null = stack.buildCommand;
          if (!hardGateCommand && stack.type !== "unknown") {
            hardGateCommand = await getFallbackHardGateCommand(stack.type, projectDirForGate);
          }
          if (hardGateCommand) {
            const command = hardGateCommand.replace(/2>&1/g, "").trim();
            this.log(
              "info",
              `[${this.config.agentRole}] Hard Gate: running ${command}...`
            );
            const execTool = tools.executeCommand as any;
            const gateResult = await execTool.execute({
              command,
              reason: "Hard Gate: Pre-QA compile check",
            });
            const exitCode =
              typeof gateResult?.exitCode === "number" ? gateResult.exitCode : undefined;
            const success = gateResult?.success === true;
            const stdout =
              typeof gateResult?.stdout === "string" ? gateResult.stdout : "";
            const stderr =
              typeof gateResult?.stderr === "string" ? gateResult.stderr : "";
            if (!success || (typeof exitCode === "number" && exitCode !== 0)) {
              hardGateFailed = true;
              const combined = [stdout, stderr].filter(Boolean).join("\n");
              const headLen = 2000;
              const tailLen = 2000;
              const body =
                combined.length <= headLen + tailLen
                  ? combined
                  : combined.slice(0, headLen) + "\n[MIDDLE TRUNCATED]\n" + combined.slice(-tailLen);
              hardGateCompileOutput = HARD_GATE_STOP_HEADER + body;
              const truncated = hardGateCompileOutput;
              this.log(
                "error",
                `[${this.config.agentRole}] Hard Gate failed (${command}): ${truncated}`
              );
            }
          } else {
            this.log(
              "info",
              `[${this.config.agentRole}] Hard Gate skipped (no build command for stack: ${stack.type})`
            );
          }
        } catch (gateErr) {
          // detectStack or execute error: skip Hard Gate and proceed
          this.log(
            "info",
            `[${this.config.agentRole}] Hard Gate skipped: ${
              gateErr instanceof Error ? gateErr.message : String(gateErr)
            }`
          );
        }
      }

      if (hardGateFailed) {
        const reasoning =
          `Hard Gate: Pre-QA compile check failed. The project did not pass type-check or build. Fix the compiler errors and retry.\n\n` +
          (hardGateCompileOutput
            ? `Compiler output:\n${hardGateCompileOutput.slice(0, 8000)}`
            : "No output captured.");
        try {
          await prisma.comment.create({
            data: {
              taskId,
              content:
                "### EXECUTION ERRORS ###\n" +
                (hardGateCompileOutput || "No output captured."),
              authorRole: "DEVOPS",
              isSystem: false,
            },
          });
        } catch (commentErr) {
          this.log(
            "error",
            `[${this.config.agentRole}] Failed to create Hard Gate DEVOPS comment: ${
              commentErr instanceof Error ? commentErr.message : String(commentErr)
            }`
          );
        }
        return {
          success: false,
          taskId,
          error: "Hard Gate: build/type-check failed",
          compileOutput: hardGateCompileOutput,
        };
      }

      let reportText = results
        .map((r: any) => {
          if (r.success) {
            const result = r.result as any;
            if (result?.stdout || result?.stderr || result?.error) {
              const output = result.stdout || result.stderr || result.error || "";
              return `- ${r.toolName}: ${output.slice(0, 200)}`;
            }
            return `- ${r.toolName}: completed`;
          }
          return `- ${r.toolName}: failed - ${r.error}`;
        })
        .join("\n");

      const executionErrorLines: string[] = [];
      for (const r of results as any[]) {
        const result: any = r.result;
        const exitCode =
          typeof result?.exitCode === "number" ? result.exitCode : undefined;
        const failed =
          r.success === false || (typeof exitCode === "number" && exitCode !== 0);

        if (!failed) continue;

        const stderr =
          typeof result?.stderr === "string" ? result.stderr : undefined;
        const errorField =
          typeof result?.error === "string" ? result.error : undefined;
        const topLevelError =
          typeof r.error === "string" ? r.error : undefined;

        const rawSnippet = stderr || errorField || topLevelError || "";
        const snippet = rawSnippet
          ? rawSnippet.slice(0, 500)
          : "Unknown error (no stderr or message provided).";

        executionErrorLines.push(
          `- Tool: ${r.toolName}\n  Error: ${snippet}`
        );
      }

      if (executionErrorLines.length > 0) {
        const executionErrorsBlock =
          `\n\n### EXECUTION ERRORS ###\n` +
          `The following commands failed during execution:\n` +
          executionErrorLines.join("\n");
        reportText = reportText
          ? reportText + executionErrorsBlock
          : executionErrorsBlock.trimStart();
      }

      const toolsUsed = Array.from(
        new Set((results as any[]).map((r) => r.toolName).filter(Boolean))
      ).join(", ");
      const hadExecutionFailures = executionErrorLines.length > 0;
      const lastBuildLabel = lastBuildPassed
        ? "passed (or was not required)"
        : "failed";

      const handoffSummaryLines = [
        "Summary:",
        `- Tools used: ${toolsUsed || "none"}`,
        `- Last build/type-check (if any): ${lastBuildLabel}`,
        `- Execution failures: ${
          hadExecutionFailures ? "yes (see EXECUTION ERRORS below)" : "no"
        }`,
      ];
      const handoffSummary = handoffSummaryLines.join("\n");

      if (artifactsOutput || automatedCheckOutput) {
        let verificationSection = "\n\n=== VERIFICATION EVIDENCE ===\n\n";
        if (verifiedArtifactPaths.length > 0) {
          verificationSection +=
            "Verified artifact paths (relative to project root): " +
            verifiedArtifactPaths.join(", ") +
            "\n\nПроверенные пути артефактов:\n" +
            verifiedArtifactPaths.map((p) => "  - " + p).join("\n") +
            "\n\n";
        }
        if (artifactsOutput) {
          verificationSection += "Artifacts Check:\n" + artifactsOutput + "\n\n";
        }
        if (automatedCheckOutput) {
          verificationSection += "Automated Check:\n" + automatedCheckOutput + "\n\n";
        }
        if (hasVerificationFailures) {
          verificationSection +=
            "⚠️ Some verification commands failed. See output above for details.\n";
        } else {
          verificationSection += "✅ All verification commands succeeded.\n";
        }
        reportText = reportText
          ? reportText + "\n\n" + verificationSection
          : verificationSection;
      }

      // FIX 3: Explicit headless exception so QA applies HEADLESS EXCEPTION when Artifacts + AutomatedCheck are satisfied
      const headlessBlock = `

### Manual Check
[HEADLESS MODE TRIGGERED]
В headless-окружении (Docker/Server) ручная проверка (открыть браузер, сделать скриншот) физически невозможна. Проверка отложена на пользователя. QA-агент обязан применить HEADLESS EXCEPTION, если Artifacts и AutomatedCheck пройдены успешно.`;
      reportText = `=== IMPLEMENTATION REPORT ===\n\n${handoffSummary}\n\n${
        reportText || "No report generated."
      }${headlessBlock}`;

      try {
        await prisma.comment.create({
          data: {
            taskId,
            content: reportText,
            authorRole: "DEVOPS",
            isSystem: false,
          },
        });
        this.log(
          "info",
          `[${this.config.agentRole}] Created implementation report comment for task ${taskId} (length=${reportText.length})`
        );
      } catch (commentErr) {
        this.log(
          "error",
          `[${this.config.agentRole}] Failed to create report comment: ${
            commentErr instanceof Error ? commentErr.message : String(commentErr)
          }`
        );
      }

      this.log(
        "info",
        `\n📤 [TASK_EXECUTOR] -> @QA: "Я закончил писать код. Отправил тебе на ревью (QA_REQUEST)."`
      );
      await this.sendMessage(
        AgentRole.QA,
        MessageType.QA_REQUEST,
        {
          taskId,
          report: reportText || "No report generated.",
        },
        message.id
      );

      this.log(
        "info",
        `[${this.config.agentRole}] Sent QA_REQUEST for task ${taskId}`
      );

      return {
        success: true,
        results,
        taskId,
        taskTitle: task.title,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.log(
        "error",
        `[${this.config.agentRole}] ReAct task execution failed for task ${taskId}: ${msg}`
      );

      try {
        await prisma.task.update({
          where: { id: taskId },
          data: { status: "REJECTED" },
        });
      } catch (updateErr) {
        this.log(
          "error",
          `[${this.config.agentRole}] Failed to mark task ${taskId} as REJECTED: ${
            updateErr instanceof Error ? updateErr.message : String(updateErr)
          }`
        );
      }

      try {
        await prisma.comment.create({
          data: {
            taskId,
            content: `**TASK_EXECUTOR CRASH**\n\n${msg}`,
            authorRole: "DEVOPS",
            isSystem: true,
          },
        });
      } catch (commentErr) {
        this.log(
          "error",
          `[${this.config.agentRole}] Failed to create rejection comment for task ${taskId}: ${
            commentErr instanceof Error ? commentErr.message : String(commentErr)
          }`
        );
      }

      // Ensure Dispatcher does not deadlock by synthesizing a QA_RESPONSE
      // so TeamLead can create or update a fixing ticket.
      try {
        await this.sendMessage(
          AgentRole.TEAMLEAD,
          MessageType.QA_RESPONSE,
          {
            taskId,
            finalStatus: "REJECTED",
            reasoning: `System crash in TASK_EXECUTOR: ${msg}`,
          },
          message.id
        );
        this.log(
          "info",
          `[${this.config.agentRole}] Sent synthetic QA_RESPONSE(REJECTED) to TEAMLEAD for crashed task ${taskId}`
        );
      } catch (sendErr) {
        this.log(
          "error",
          `[${this.config.agentRole}] Failed to send synthetic QA_RESPONSE for crashed task ${taskId}: ${
            sendErr instanceof Error ? sendErr.message : String(sendErr)
          }`
        );
      }

      return {
        success: false,
        taskId,
        error: msg,
      };
    }
  }

  /** Пишет в комментарии к задаче причину отклонения тикета (статус тикета уже REJECTED). */
  private async createTicketRejectionComment(
    taskId: string,
    ticketId: string,
    reason: string
  ): Promise<void> {
    try {
      await prisma.comment.create({
        data: {
          taskId,
          content: `**Тикет отклонён** (ticket \`${ticketId}\`)\n\n${reason}`,
          authorRole: "DEVOPS",
          isSystem: true,
        },
      });
      this.log("info", `[${this.config.agentRole}] Created rejection comment for ticket ${ticketId} on task ${taskId}`);
    } catch (err) {
      this.log(
        "error",
        `[${this.config.agentRole}] Failed to create ticket rejection comment: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }

  private async handleTicketRequest(
    message: AgentMessage
  ): Promise<Record<string, unknown>> {
    const { ticketId, relatedTaskId } = message.payload as {
      ticketId: string;
      relatedTaskId: string;
    };

    this.log("info", `[${this.config.agentRole}] Processing ticket: ${ticketId}`);

    try {
      const ticket = await prisma.ticket.findUnique({
        where: { id: ticketId },
      });
      if (!ticket) {
        throw new Error(`Ticket ${ticketId} not found`);
      }

      if (!relatedTaskId) {
        throw new Error(`Ticket ${ticketId} has no relatedTaskId`);
      }

      const originalTask = await prisma.task.findUnique({
        where: { id: relatedTaskId },
        include: {
          plan: {
            include: { project: true },
          },
        },
      });

      if (!originalTask || !originalTask.plan?.project) {
        throw new Error(
          `Original task ${relatedTaskId} not found or has no project`
        );
      }

      const project = originalTask.plan.project;
      const projectId = project.id;

      const extraRequirement =
        `Ticket "${ticket.title}": ${ticket.description}\n` +
        `Adapt your instructions to address this requirement from the ticket.`;

      this.log(
        "info",
        `\n🔧 [TASK_EXECUTOR] -> @QA: "Вижу твои замечания по тикету. Извиняюсь, сейчас всё исправлю."`
      );
      this.log(
        "info",
        `[${this.config.agentRole}] Regenerating prompt for task ${relatedTaskId} based on ticket ${ticketId}...`
      );

      let instructions: string;
      try {
        instructions = await generateTaskPrompt(
          relatedTaskId,
          true,
          false,
          extraRequirement
        );
      } catch (promptErr) {
        const msg = promptErr instanceof Error ? promptErr.message : String(promptErr);
        this.log(
          "error",
          `[${this.config.agentRole}] Failed to regenerate prompt: ${msg}`
        );

        await prisma.ticket.update({
          where: { id: ticketId },
          data: { status: "REJECTED" },
        });
        await this.createTicketRejectionComment(
          relatedTaskId,
          ticketId,
          `Не удалось перегенерировать промпт: ${msg}`
        );

        throw new Error(`Failed to regenerate prompt: ${msg}`);
      }

      const tools = createAgentTools(
        projectId,
        this.config.sessionId,
        this.config.mode ?? "local",
        this.log.bind(this)
      );

      const baseProjectContext = await getCompactProjectContext(projectId);

      let codeMapSection = "Code map unavailable or empty for this project.";
      try {
        const codeMapResult = await (tools.getCodeMap as any).execute({});
        if (codeMapResult) {
          if (
            typeof codeMapResult.codeMap === "string" &&
            codeMapResult.codeMap.trim()
          ) {
            codeMapSection = codeMapResult.codeMap;
          } else if (
            typeof codeMapResult.error === "string" &&
            codeMapResult.error.trim()
          ) {
            codeMapSection = `Code map unavailable: ${codeMapResult.error}`;
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        codeMapSection = `Code map unavailable due to error: ${msg}`;
      }

      let relevantCodeSection = "No relevant code chunks found for this ticket/task.";
      try {
        const searchResultsRaw = await (tools.searchCodebase as any).execute({
          query: originalTask.title,
        });
        const searchResults = Array.isArray(searchResultsRaw)
          ? searchResultsRaw
          : [];

        if (searchResults.length > 0) {
          relevantCodeSection = searchResults
            .map((r: any) => {
              const header = r?.filePath
                ? `// File: ${r.filePath}`
                : "// File: (unknown)";
              const content =
                typeof r?.content === "string"
                  ? r.content
                  : JSON.stringify(r, null, 2);
              return `${header}\n${content}`;
            })
            .join("\n---\n");
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        relevantCodeSection = `Relevant code search failed: ${msg}`;
      }

      const ticketWorkDir =
        this.config.mode === "cloud"
          ? CONTAINER_WORKSPACE_ROOT
          : getProjectDir(projectId);
      const existingProjectContext = `
### CURRENT WORKSPACE STATE
Working Directory: ${ticketWorkDir}

### CODE MAP
All paths below are relative to the project root (Working Directory).
${codeMapSection}

### RELEVANT EXISTING CODE
${relevantCodeSection}
`.trim();

      const projectContext = `${baseProjectContext}

${existingProjectContext}`;

      let previousCssFeedbackBlock = "";
      const relatedTaskComments = await prisma.comment.findMany({
        where: { taskId: relatedTaskId, authorRole: "QA" },
        orderBy: { createdAt: "desc" },
        take: 10,
        select: { content: true },
      });
      const cssNeedsImprovementComment = relatedTaskComments.find(
        (c) =>
          c.content.includes("CSS Review") &&
          c.content.includes("Status: NEEDS_IMPROVEMENT")
      );
      if (cssNeedsImprovementComment?.content) {
        const suggestionsMatch = cssNeedsImprovementComment.content.match(
          /Suggestions:\s*([\s\S]*?)(?=\n\n|$)/i
        );
        const suggestions = suggestionsMatch
          ? suggestionsMatch[1]
              .trim()
              .split(/\n/)
              .map((s) => s.trim())
              .filter(Boolean)
          : [];
        const reasoningMatch = cssNeedsImprovementComment.content.match(
          /Reasoning:\s*([\s\S]*?)(?=Suggestions:|$)/i
        );
        const reasoning = reasoningMatch
          ? reasoningMatch[1].trim().slice(0, 500)
          : "";
        if (suggestions.length > 0 || reasoning) {
          previousCssFeedbackBlock =
            "\n\n=== PREVIOUS CSS FEEDBACK (apply these fixes) ===\n" +
            (reasoning ? `Reasoning: ${reasoning}\n` : "") +
            (suggestions.length > 0
              ? "Suggestions:\n" + suggestions.map((s) => `- ${s}`).join("\n")
              : "");
        }
      }

      const initialTaskContext = [
        `Ticket ID: ${ticket.id}`,
        `Ticket title: ${ticket.title}`,
        `Related task ID: ${originalTask.id}`,
        `Related task title: ${originalTask.title}`,
        "",
        "Ticket QA rejection / bug context:",
        ticket.description ?? "(no description)",
        "",
        "Regenerated task instructions (including ticket requirements):",
        instructions,
        previousCssFeedbackBlock,
        "",
        "Project context:",
        projectContext,
      ]
        .filter(Boolean)
        .join("\n")
        .trim();

      const {
        results,
        lastSuccessfulCssWrite: ticketCssWrite,
        shouldEscalate,
        lastCompletedIteration,
        shouldSubmitForQA,
      } = await this.runReActLoop({
        projectId,
        projectProvider: project.aiProvider,
        projectModel: project.aiModel,
        initialTaskContext,
        tools,
        primaryTaskId: relatedTaskId,
        parentMessageId: message.id,
      });

      const reachedIterationLimit =
        lastCompletedIteration >= MAX_REACT_ITERATIONS && !shouldSubmitForQA;

      if (shouldEscalate || reachedIterationLimit) {
        this.log(
          "info",
          `[${this.config.agentRole}] Escalate to TeamLead (ticket): Task execution aborted due to iteration limit or manual escalation.`
        );

        await prisma.ticket.update({
          where: { id: ticketId },
          data: { status: "REJECTED" },
        });

        await this.sendMessage(
          AgentRole.TEAMLEAD,
          MessageType.QA_RESPONSE,
          {
            taskId: relatedTaskId,
            ticketId,
            finalStatus: "REJECTED",
            reasoning:
              "TASK_EXECUTOR escalated the task. Agent got stuck or reached the max iteration limit of 30 without submitting for QA. This requires human intervention or replanning.",
          },
          message.id
        );

        return {
          success: false,
          ticketId,
          taskId: relatedTaskId,
          error: "Task escalated to TeamLead",
        };
      }

      const ticketProjectDir = getProjectDir(projectId);
      if (ticketCssWrite && (await isWebProject(ticketProjectDir))) {
        try {
          this.log(
            "info",
            `[${this.config.agentRole}] 🎨 Running CSS Agent synchronously before QA (ticket)`
          );
          const cssAgent = createAgentForRole(
            AgentRole.CSS,
            this.config.sessionId ?? "",
            projectId,
            false,
            this.log.bind(this),
            this.config.mode ?? "local"
          );
          const styleRequestMessage = {
            id: "sync-css-ticket",
            eventType: MessageType.STYLE_REQUEST,
            payload: {
              taskId: relatedTaskId,
              filePath: ticketCssWrite.filePath,
              content: ticketCssWrite.content,
            },
            replyToId: message.id,
          } as unknown as AgentMessage;
          const cssResult = (await cssAgent.processMessage(styleRequestMessage)) as {
            status?: string;
            improvedContent?: string;
            suggestions?: string[];
          };
          if (
            cssResult?.status === "NEEDS_IMPROVEMENT" &&
            typeof cssResult?.improvedContent === "string" &&
            cssResult.improvedContent.trim()
          ) {
            await this.applyCssImprovedContent({
              projectId,
              filePath: ticketCssWrite.filePath,
              improvedContent: cssResult.improvedContent,
            });
          }
        } catch (cssErr) {
          this.log(
            "error",
            `[${this.config.agentRole}] Sync CSS agent failed (ticket): ${
              cssErr instanceof Error ? cssErr.message : String(cssErr)
            }`
          );
        }
      } else if (ticketCssWrite) {
        this.log("info", `[${this.config.agentRole}] [CSS] Skipped - not a web project`);
      }

      const {
        artifactsOutput,
        automatedCheckOutput,
        hasVerificationFailures,
        hardGateFailed,
        hardGateCompileOutput,
        verifiedArtifactPaths,
      } = await this.runVerificationAndHardGate({
        mode: "ticket",
        projectId,
        taskId: relatedTaskId,
        ticketId,
        verificationCriteria: originalTask.verificationCriteria as {
          artifacts?: string[];
          automatedCheck?: string;
          manualCheck?: string;
        } | null,
        tools,
        parentMessageId: message.id,
      });

      if (hardGateFailed) {
        return {
          success: false,
          ticketId,
          taskId: relatedTaskId,
          error: "Hard Gate: build/type-check failed",
          compileOutput: hardGateCompileOutput,
        };
      }

      const ticketVerificationNote =
        "\n[Ticket run] automatedCheck from the original task was re-run after applying the ticket fix. Verify based on artifacts, automatedCheck output, and report content.";

      let reportText = results
        .map((r: any) => {
          if (r.success) {
            const result = r.result as any;
            if (result?.stdout || result?.stderr || result?.error) {
              const output = result.stdout || result.stderr || result.error || "";
              return `- ${r.toolName}: ${output.slice(0, 200)}`;
            }
            return `- ${r.toolName}: completed`;
          }
          return `- ${r.toolName}: failed - ${r.error}`;
        })
        .join("\n");

      const executionErrorLines: string[] = [];
      for (const r of results as any[]) {
        const result: any = r.result;
        const exitCode =
          typeof result?.exitCode === "number" ? result.exitCode : undefined;
        const failed =
          r.success === false || (typeof exitCode === "number" && exitCode !== 0);

        if (!failed) continue;

        const stderr =
          typeof result?.stderr === "string" ? result.stderr : undefined;
        const errorField =
          typeof result?.error === "string" ? result.error : undefined;
        const topLevelError =
          typeof r.error === "string" ? r.error : undefined;

        const rawSnippet = stderr || errorField || topLevelError || "";
        const snippet = rawSnippet
          ? rawSnippet.slice(0, 500)
          : "Unknown error (no stderr or message provided).";

        executionErrorLines.push(
          `- Tool: ${r.toolName}\n  Error: ${snippet}`
        );
      }

      if (executionErrorLines.length > 0) {
        const executionErrorsBlock =
          `\n\n### EXECUTION ERRORS ###\n` +
          `The following commands failed during execution:\n` +
          executionErrorLines.join("\n");
        reportText = reportText
          ? reportText + executionErrorsBlock
          : executionErrorsBlock.trimStart();
      }

      if (artifactsOutput || automatedCheckOutput) {
        let verificationSection = "\n\n=== VERIFICATION EVIDENCE ===\n\n";
        if (verifiedArtifactPaths.length > 0) {
          verificationSection +=
            "Verified artifact paths (relative to project root): " +
            verifiedArtifactPaths.join(", ") +
            "\n\nПроверенные пути артефактов:\n" +
            verifiedArtifactPaths.map((p) => "  - " + p).join("\n") +
            "\n\n";
        }
        if (artifactsOutput) {
          verificationSection += "Artifacts Check:\n" + artifactsOutput + "\n\n";
        }
        if (automatedCheckOutput) {
          verificationSection += "Automated Check:\n" + automatedCheckOutput + "\n\n";
        }
        verificationSection += ticketVerificationNote + "\n\n";
        if (hasVerificationFailures) {
          verificationSection +=
            "⚠️ Some verification commands failed. See output above for details.\n";
        } else {
          verificationSection += "✅ All verification commands succeeded.\n";
        }
        reportText = reportText
          ? reportText + "\n\n" + verificationSection
          : verificationSection;
      } else {
        reportText = (reportText || "") + ticketVerificationNote;
      }

      const headlessBlockTicket = `

### Manual Check
[HEADLESS MODE TRIGGERED]
В headless-окружении (Docker/Server) ручная проверка (открыть браузер, сделать скриншот) физически невозможна. Проверка отложена на пользователя. QA-агент обязан применить HEADLESS EXCEPTION, если Artifacts и AutomatedCheck пройдены успешно.`;
      reportText = `=== IMPLEMENTATION REPORT (TICKET RUN) ===\n\n${
        reportText || "No report generated."
      }${headlessBlockTicket}`;

      try {
        await prisma.comment.create({
          data: {
            taskId: relatedTaskId,
            content: reportText,
            authorRole: "DEVOPS",
            isSystem: false,
          },
        });
        this.log(
          "info",
          `[${this.config.agentRole}] Created implementation report comment for ticket ${ticketId} (task ${relatedTaskId}, length=${reportText.length})`
        );
      } catch (commentErr) {
        this.log(
          "error",
          `[${this.config.agentRole}] Failed to create report comment: ${
            commentErr instanceof Error ? commentErr.message : String(commentErr)
          }`
        );
      }

      this.log(
        "info",
        `\n📤 [TASK_EXECUTOR] -> @QA: "Я закончил писать код. Отправил тебе на ревью (QA_REQUEST)."`
      );
      // Статус тикета выставит TeamLead по результату QA (единый источник истины).
      await this.sendMessage(
        AgentRole.QA,
        MessageType.QA_REQUEST,
        {
          taskId: relatedTaskId,
          report: reportText || "No report generated.",
          ticketId,
        },
        message.id
      );

      this.log(
        "info",
        `[${this.config.agentRole}] Sent QA_REQUEST for ticket ${ticketId} (task ${relatedTaskId}); ticket status will be set by QA result`
      );

      return {
        success: true,
        results,
        ticketId,
        taskId: relatedTaskId,
        taskTitle: originalTask.title,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.log(
        "error",
        `[${this.config.agentRole}] Ticket execution failed for ticket ${ticketId} (task ${relatedTaskId}): ${msg}`
      );

      try {
        await prisma.comment.create({
          data: {
            taskId: relatedTaskId,
            content: `**TASK_EXECUTOR CRASH (TICKET RUN)**\n\nTicket ${ticketId}\n\n${msg}`,
            authorRole: "DEVOPS",
            isSystem: true,
          },
        });
      } catch (commentErr) {
        this.log(
          "error",
          `[${this.config.agentRole}] Failed to create crash comment for ticket ${ticketId} (task ${relatedTaskId}): ${
            commentErr instanceof Error ? commentErr.message : String(commentErr)
          }`
        );
      }

      try {
        await this.sendMessage(
          AgentRole.TEAMLEAD,
          MessageType.QA_RESPONSE,
          {
            taskId: relatedTaskId,
            finalStatus: "REJECTED",
            ticketId,
            reasoning: `System crash in TASK_EXECUTOR during ticket run: ${msg}`,
          },
          message.id
        );
        this.log(
          "info",
          `[${this.config.agentRole}] Sent synthetic QA_RESPONSE(REJECTED) to TEAMLEAD for crashed ticket ${ticketId} (task ${relatedTaskId})`
        );
      } catch (sendErr) {
        this.log(
          "error",
          `[${this.config.agentRole}] Failed to send synthetic QA_RESPONSE for crashed ticket ${ticketId} (task ${relatedTaskId}): ${
            sendErr instanceof Error ? sendErr.message : String(sendErr)
          }`
        );
      }

      return {
        success: false,
        ticketId,
        taskId: relatedTaskId,
        error: msg,
      };
    }
  }

  private async handleTicketRequestLegacy(message: AgentMessage): Promise<Record<string, unknown>> {
    const { ticketId, relatedTaskId } = message.payload as { ticketId: string; relatedTaskId: string };

    this.log("info", `[${this.config.agentRole}] Processing ticket: ${ticketId}`);

    try {
      const ticket = await prisma.ticket.findUnique({
        where: { id: ticketId },
      });
      if (!ticket) {
        throw new Error(`Ticket ${ticketId} not found`);
      }

      if (!relatedTaskId) {
        throw new Error(`Ticket ${ticketId} has no relatedTaskId`);
      }

      const originalTask = await prisma.task.findUnique({
        where: { id: relatedTaskId },
        include: {
          plan: {
            include: { project: true },
          },
        },
      });

      if (!originalTask || !originalTask.plan?.project) {
        throw new Error(`Original task ${relatedTaskId} not found or has no project`);
      }

      const project = originalTask.plan.project;
      const projectId = project.id;

      const extraRequirement =
        `Ticket "${ticket.title}": ${ticket.description}\n` +
        `Adapt your instructions to address this requirement from the ticket.`;

      this.log("info", `\n🔧 [TASK_EXECUTOR] -> @QA: "Вижу твои замечания по тикету. Извиняюсь, сейчас всё исправлю."`);
      this.log("info", `[${this.config.agentRole}] Regenerating prompt for task ${relatedTaskId} based on ticket ${ticketId}...`);

      let instructions: string;
      try {
        instructions = await generateTaskPrompt(
          relatedTaskId,
          true,
          false,
          extraRequirement
        );
      } catch (promptErr) {
        const msg = promptErr instanceof Error ? promptErr.message : String(promptErr);
        this.log("error", `[${this.config.agentRole}] Failed to regenerate prompt: ${msg}`);

        await prisma.ticket.update({
          where: { id: ticketId },
          data: { status: "REJECTED" },
        });
        await this.createTicketRejectionComment(relatedTaskId, ticketId, `Не удалось перегенерировать промпт: ${msg}`);

        throw new Error(`Failed to regenerate prompt: ${msg}`);
      }

      const tools = createAgentTools(
        projectId,
        this.config.sessionId,
        this.config.mode ?? "local",
        this.log.bind(this)
      );

      const baseProjectContext = await getCompactProjectContext(projectId);

      let codeMapSection = "Code map unavailable or empty for this project.";
      try {
        const codeMapResult = await (tools.getCodeMap as any).execute({});
        if (codeMapResult) {
          if (typeof codeMapResult.codeMap === "string" && codeMapResult.codeMap.trim()) {
            codeMapSection = codeMapResult.codeMap;
          } else if (typeof codeMapResult.error === "string" && codeMapResult.error.trim()) {
            codeMapSection = `Code map unavailable: ${codeMapResult.error}`;
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        codeMapSection = `Code map unavailable due to error: ${msg}`;
      }

      let relevantCodeSection = "No relevant code chunks found for this ticket/task.";
      try {
        const searchResultsRaw = await (tools.searchCodebase as any).execute({
          query: originalTask.title,
        });
        const searchResults = Array.isArray(searchResultsRaw) ? searchResultsRaw : [];

        if (searchResults.length > 0) {
          relevantCodeSection = searchResults
            .map((r: any) => {
              const header = r?.filePath ? `// File: ${r.filePath}` : "// File: (unknown)";
              const content =
                typeof r?.content === "string"
                  ? r.content
                  : JSON.stringify(r, null, 2);
              return `${header}\n${content}`;
            })
            .join("\n---\n");
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        relevantCodeSection = `Relevant code search failed: ${msg}`;
      }

      const planWorkDir =
        this.config.mode === "cloud"
          ? CONTAINER_WORKSPACE_ROOT
          : getProjectDir(projectId);
      const existingProjectContext = `
### CURRENT WORKSPACE STATE
Working Directory: ${planWorkDir}

### CODE MAP
All paths below are relative to the project root (Working Directory).
${codeMapSection}

### RELEVANT EXISTING CODE
${relevantCodeSection}
`.trim();

      const projectContext = `${baseProjectContext}

${existingProjectContext}`;

      const planSystemPrompt = `You are an experienced software engineer. Your job is to produce an execution plan as JSON only.

${projectContext}

Task: ${originalTask.title}
Instructions: ${instructions}

Your output MUST be a JSON object with a "steps" array. Each step is either:
- A thought: { "thought": "Brief explanation of why you are doing next action" }
- A tool call: { "toolName": "<name>", "params": { ... } }

Available toolNames: executeCommand, readFile, writeFile, replaceInFile, searchKnowledge, searchCodebase, getCodeMap, searchWeb.
BEFORE each tool step, add a "thought" step explaining WHY you are doing this.

At the start of a task, always call getCodeMap (with no params or depth 0) to understand the project structure and locate the specific functions or classes you need to work with. Then use readFile or searchCodebase as needed.

CRITICAL RULES FOR FILE MANIPULATION:
1. NO PLACEHOLDERS: You are FORBIDDEN from using placeholders like "// rest of the code", "<!-- existing code -->", or "...".
FILE EDITING RULES:
- For SMALL or PARTIAL edits in existing files, you MUST use replaceInFile. Provide the exact searchString (a sufficiently unique block of code, minimum 3-4 lines to avoid ambiguous matches) and the replaceString.
- DO NOT use writeFile to update existing files unless you are completely rewriting a very small file. writeFile is strictly for CREATING new files. Overwriting a 500-line file to change 1 line will cause critical data loss and token waste.
5. DYNAMIC TOOLING: For complex refactors or regex-heavy replacements, include \`executeCommand\` steps that create and run temporary Node.js or Bash scripts (e.g. write temp-edit.js with a heredoc, then \`node temp-edit.js\`). You are allowed to "write the tools you need" via executeCommand.
6. PATH FORMAT: All paths must be strictly relative to the project root (e.g. src/app/page.tsx, index.html). Do not use \`../\` or absolute paths. The writeFile tool auto-creates missing parent directories, so do not include separate mkdir steps for new files.
7. BACKGROUND SERVERS: NEVER run development servers (e.g., \`npm run dev\`, \`npm start\`) synchronously, as they block execution and trigger timeouts. Always run them in the background (e.g., \`npm run build && (npm run preview & sleep 5) && wget --spider -q http://localhost:4173 || exit 1\` for Vite preview, or use port 3000 for Next.js).
8. PROJECT STRUCTURE: To initialize a new project, use executeCommand with the appropriate init command (e.g. npx create-next-app@latest . --yes, npm create vite@latest . -- --template react-ts). After scaffolding, run npm install from project root. Your generated steps MUST NOT include \`cd <project-name>\`. Every command (npm install, build, etc.) must be executed from the project root. Never assume a nested folder structure. STATELESS: executeCommand (sync) does not remember \`cd\`; chain commands (e.g. \`cd x && npm run build\`) or use the \`cwd\` parameter every time.
9. WORKSPACE SCAFFOLDING RULE: You are operating inside a dedicated project root directory. Whenever you initialize or scaffold a new project—regardless of the language, package manager, or framework (Node, Python, Rust, C++, etc.)—you MUST initialize it directly in the CURRENT directory (e.g., using '.' or the equivalent current-directory flag for your specific tool). NEVER create a nested project subfolder unless the user explicitly instructs you to do so. CRITICAL FALLBACK: If a framework forces you to scaffold into a named subfolder (e.g., creating a my-app directory instead of using .), you MUST IMMEDIATELY move all generated files up to the current root directory using a command like mv my-app/* my-app/.* . 2>/dev/null || true && rm -rf my-app. The package.json MUST be located exactly in the project root before you proceed to any other step.
10. DEPENDENCY INSTALLATION: Before running any install command, collect ALL required dependencies (both runtime and dev) and install them in a single command. Never run multiple sequential install commands when one will do. Applies to all package managers (npm, pip, cargo, go mod, etc.).

Example:
{
  "steps": [
    { "thought": "I need to see what files are in project." },
    { "toolName": "readFile", "params": { "filePath": "package.json" } },
    { "thought": "Now I will run tests." },
    { "toolName": "executeCommand", "params": { "command": "npm test", "reason": "Run tests" } }
  ]
}

### 🚨 CRITICAL: TICKET RETRY PROTOCOL 🚨
You are attempting to fix a task that previously FAILED QA verification. The rejection reason is provided above in the ticket and verification context.
To prevent repeating the same mistake, your JSON execution plan MUST strictly follow this rule:

- Your VERY FIRST element in the "steps" array MUST be a thought-only step (no "toolName" field) that performs a Root Cause Analysis (RCA) of the previous failed attempt.
- In this RCA thought, you MUST explicitly state:
  1. What specific evidence was missing or what failed in the previous attempt (based on the QA rejection / ticket message).
  2. What EXACTLY you will do differently in this new plan to ensure the QA passes this time.
- You MUST NOT include any step with "toolName" (i.e. no tool usage) before this RCA thought step.

Formally, steps[0] MUST be a JSON object of the form:
{ "thought": "Root cause analysis of the previous QA rejection and what I will do differently now." }

Do NOT use any tools until you have outputted this RCA thought as the first step in the "steps" array.

IMPORTANT: Keep your execution plans concise. If a task requires more than 15 steps, only plan the first 10-12 steps and end your plan with a note to "Continue implementation in the next phase". Do NOT attempt to generate 40+ steps in a single response, as it causes API timeouts.

If you cannot create a plan (e.g. task is unclear or out of scope), return exactly: {"steps": []}. Never return null or undefined for "steps"—always an array. Return ONLY valid JSON, no markdown or extra text.`;

      const planUserMessage = `Create an execution plan for this task: ${originalTask.title}\n\nInstructions: ${instructions}`;

      const resolvedProvider = resolveProvider(project.aiProvider || undefined);
      const resolvedModel = project.aiModel || "gpt-4o-mini";
      const llmSettings = await getLLMSettings();
      const planMaxTokens = Math.min(llmSettings.maxTokens ?? 16384, 16384);

      this.log("info", `[${this.config.agentRole}] Generating execution plan for ticket ${ticketId} (task: ${originalTask.title})`);

      let planJsonText: string;
      try {
        if (resolvedProvider === "zai") {
          const { generateTextZai } = await import("@/lib/ai/zai");
          const raw = await callWithRetries(() =>
            generateTextZai({
              systemPrompt: planSystemPrompt,
              userMessage: planUserMessage,
              model: resolvedModel,
              temperature: llmSettings.temperature,
              maxTokens: planMaxTokens,
              signal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
            })
          );
          if (raw == null || typeof raw !== "string") {
            throw new Error("AI returned invalid response");
          }
          planJsonText = raw;
        } else {
          const aiResult = await callWithRetries(() =>
            generateText({
              model: getModel(resolvedProvider, resolvedModel),
              system: planSystemPrompt,
              prompt: planUserMessage,
              temperature: llmSettings.temperature,
              maxTokens: planMaxTokens,
              abortSignal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
            })
          );
          if (aiResult == null) {
            throw new Error("AI returned null");
          }
          planJsonText = aiResult.text ?? "";
          try {
            await trackAIUsage(aiResult, {
              projectId,
              actionType: "execute_task",
              model: resolvedModel,
              executionSessionId: this.config.sessionId,
            });
          } catch (trackErr) {
            this.log("error", `trackAIUsage failed: ${trackErr instanceof Error ? trackErr.message : String(trackErr)}`);
          }
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        this.log("error", `LLM call failed: ${msg}`);

        await prisma.ticket.update({
          where: { id: ticketId },
          data: { status: "REJECTED" },
        });
        await this.createTicketRejectionComment(relatedTaskId, ticketId, `Ошибка вызова LLM: ${msg}`);

        throw new Error(`LLM call failed: ${msg}`);
      }

      if (!planJsonText) {
        await prisma.ticket.update({
          where: { id: ticketId },
          data: { status: "REJECTED" },
        });
        await this.createTicketRejectionComment(relatedTaskId, ticketId, "AI вернул пустой ответ.");
        throw new Error("AI returned empty response");
      }

      let plan: AIExecutionPlan | null = null;
      try {
        const cleaned = planJsonText.replace(/^[\s\S]*?(\{[\s\S]*\})[\s\S]*$/m, "$1").trim();
        if (!cleaned) {
          throw new Error("AI returned invalid plan");
        }
        plan = JSON.parse(cleaned) as AIExecutionPlan;
        if (!plan || !Array.isArray(plan.steps)) {
          throw new Error("Invalid plan structure");
        }
      } catch (parseErr) {
        const msg = parseErr instanceof Error ? parseErr.message : String(parseErr);
        this.log("error", `Failed to parse AI plan: ${msg}`);

        await prisma.ticket.update({
          where: { id: ticketId },
          data: { status: "REJECTED" },
        });
        await this.createTicketRejectionComment(relatedTaskId, ticketId, `Не удалось распарсить план выполнения (JSON): ${msg}`);

        throw new Error(`Failed to parse execution plan JSON: ${msg}`);
      }

      const steps = plan.steps;
      if (steps.length === 0) {
        await prisma.ticket.update({
          where: { id: ticketId },
          data: { status: "REJECTED" },
        });
        await this.createTicketRejectionComment(relatedTaskId, ticketId, "AI не сгенерировал ни одного шага выполнения.");
        throw new Error("AI failed to generate any execution steps");
      }

      this.log("info", `[${this.config.agentRole}] Ticket plan generated. ${steps.length} steps to perform.`);

      const results = [];
      for (const step of steps) {
        if (!step || typeof step !== "object") continue;

        if ("thought" in step && typeof step.thought === "string") {
          this.log("info", `[${this.config.agentRole}] 🧠 ${step.thought}`);
          continue;
        }

        if ("toolName" in step && typeof step.toolName === "string" && step.toolName in tools) {
          const toolName = step.toolName;
          const params = step.params && typeof step.params === "object" ? step.params : {};

          const commandSuffix =
            (toolName === "executeCommand" || toolName === "cloudExecuteCommand") &&
            typeof (params as any)?.command === "string"
              ? `: ${(params as any).command}`
              : "";

          this.log(
            "info",
            `[${this.config.agentRole}] 🛠️ Executing: ${toolName}${commandSuffix}`
          );

          try {
            const tool = tools[toolName as keyof typeof tools];
            const result = await (tool as any).execute(params);

            const anyResult = result as any;
            const rawExitCode =
              typeof anyResult?.exitCode === "number" ? anyResult.exitCode : undefined;
            const explicitSuccess =
              typeof anyResult?.success === "boolean" ? anyResult.success : undefined;

            let normalizedSuccess = true;
            if (explicitSuccess !== undefined || rawExitCode !== undefined) {
              normalizedSuccess =
                explicitSuccess !== false && (rawExitCode == null || rawExitCode === 0);
            }

            if (normalizedSuccess) {
              results.push({ toolName, result, success: true });
              this.log(
                "success",
                `[${this.config.agentRole}] ✅ ${toolName} completed`
              );
            } else {
              const stderr =
                typeof anyResult?.stderr === "string" ? anyResult.stderr : "";
              const errorField =
                typeof anyResult?.error === "string" ? anyResult.error : "";
              const snippet = (errorField || stderr).slice(0, 500);
              const exitCodeInfo =
                rawExitCode != null ? ` (exitCode=${rawExitCode})` : "";

              this.log(
                "error",
                `[${this.config.agentRole}] ❌ Tool ${toolName} reported failure${exitCodeInfo}${
                  snippet ? `: ${snippet}` : ""
                }`
              );

              results.push({ toolName, result, success: false });
            }

            const stepParams = step.params as Record<string, unknown>;
            if (toolName === "writeFile" && typeof stepParams.filePath === "string") {
              const filePath = stepParams.filePath;
              if (filePath.endsWith('.css') || filePath.endsWith('.scss') || filePath.endsWith('.module.css')) {
                this.log("info", `[${this.config.agentRole}] 🎨 CSS file detected, delegating to CSS Agent`);
                await this.sendMessage(
                  AgentRole.CSS,
                  MessageType.STYLE_REQUEST,
                  {
                    taskId: relatedTaskId,
                    filePath,
                    content: typeof stepParams.content === "string" ? stepParams.content : "",
                  },
                  message.id
                );
              }
            }
          } catch (toolErr) {
            const errMsg = toolErr instanceof Error ? toolErr.message : String(toolErr);
            this.log("error", `[${this.config.agentRole}] ${toolName} failed: ${errMsg}`);
            results.push({ toolName, error: errMsg, success: false });
          }
        }
      }

      const vc = originalTask.verificationCriteria as {
        artifacts?: string[];
        automatedCheck?: string;
        manualCheck?: string;
      } | null;

      let artifactsOutput = "";
      let automatedCheckOutput = "";
      let hasVerificationFailures = false;
      let verifiedArtifactPaths: string[] = [];

      if (vc) {
        const artifacts = Array.isArray(vc.artifacts) ? vc.artifacts : [];
        // Для тикета не гоняем automatedCheck исходной задачи: вывод мог измениться (HTML разбит тегами и т.д.).
        const automatedCheck = null;

        if (artifacts.length > 0 || automatedCheck) {
          this.log("info", `[${this.config.agentRole}] 🔍 Verification phase (ticket run: artifact checks only)...`);

          const verificationSteps: Array<{ command: string; reason: string; isArtifactCheck?: boolean }> = [];

          let artifactBase = ".";
          if (artifacts.length > 0) {
            if (allArtifactsArePathLike(artifacts)) {
              artifactBase = ".";
            } else {
              const rootCheckResult = await (tools.executeCommand as any).execute({
                command: MANIFEST_AT_ROOT_CHECK_CMD,
                reason: "Check manifest at workspace root",
              });
              if (rootCheckResult?.exitCode === 0) {
                artifactBase = ".";
                this.log("info", "[Verification] Manifest at workspace root; using artifactBase = '.'.");
              } else {
                try {
                  const findResult = await (tools.executeCommand as any).execute({
                    command: FIND_MANIFEST_CMD,
                    reason: "Detect project root for artifact checks",
                  });
                  artifactBase = resolveArtifactBaseFromFindResult(findResult);
                  if (isArtifactBaseInBuildDir(artifactBase)) artifactBase = ".";
                  if (artifactBase === ".") {
                    this.log("info", "[Verification] No manifest found at depth 1-3 or path invalid; using workspace root for artifact checks.");
                  }
                } catch {
                  this.log("info", "[Verification] Project root detection failed; using workspace root.");
                }
              }
            }
          }

          if (automatedCheck) {
            const VERIFY_EOF = "AI_VERIFY_SCRIPT_END_7f3a";
            verificationSteps.push({
              command: `cat << '${VERIFY_EOF}' > .ai-temp-check.sh\n${automatedCheck}\n${VERIFY_EOF}\nsh .ai-temp-check.sh`,
              reason: "Run automatedCheck",
            });
          }

        for (const artifact of artifacts) {
          const path = getVerificationPath(artifactBase, artifact);
          verificationSteps.push({
            command: `ls -la "${path}"`,
            reason: `Verify artifact: ${artifact}`,
            isArtifactCheck: true,
          });
          verificationSteps.push({
            command: `head -n 200 "${path}"`,
            reason: `Artifact content: ${artifact}`,
            isArtifactCheck: true,
          });
        }
        verifiedArtifactPaths = artifacts.map((a) => getVerificationPath(artifactBase, a));

        const artifactOutputs: string[] = [];
        const checkOutputs: string[] = [];

        for (const vstep of verificationSteps) {
          const cmd = vstep.command;
          const reason = vstep.reason;

          this.log("info", `[${this.config.agentRole}][Verification] Running: ${cmd}`);

            try {
              const execTool = tools.executeCommand;
              const result = await (execTool as any).execute({ command: cmd, reason });
              const out = typeof result?.stdout === "string" ? result.stdout : "";
              const err = typeof result?.stderr === "string" ? result.stderr : "";
              const success = result?.success === true;
              const exitCode = result?.exitCode;

              if (!success || exitCode !== 0) {
                hasVerificationFailures = true;
              }

              const output = [
                `--- ${reason} ---`,
                `$ ${cmd}`,
                `exit ${exitCode ?? "?"}`,
                out,
                err ? `stderr:\n${err}` : "",
              ]
                .filter(Boolean)
                .join("\n");

              if (vstep.isArtifactCheck) {
                artifactOutputs.push(output);
              } else {
                checkOutputs.push(output);
              }
            } catch (verr) {
              hasVerificationFailures = true;
              const output = [
                `--- ${reason} ---`,
                `$ ${cmd}`,
                `FAILED: ${verr instanceof Error ? verr.message : String(verr)}`,
              ].join("\n");

              if (vstep.isArtifactCheck) {
                artifactOutputs.push(output);
              } else {
                checkOutputs.push(output);
              }
            }
          }

          artifactsOutput = artifactOutputs.join("\n\n");
          automatedCheckOutput = checkOutputs.join("\n\n");
        }
      }

      // --- Hard Gate: Pre-QA compile check (ticket run) ---
      let ticketHardGateFailed = false;
      let ticketHardGateCompileOutput = "";
      try {
        const projectDir = getProjectDir(projectId);
        const stack = await detectStack(projectDir);
        this.log("info", `[${this.config.agentRole}] Detected stack: ${stack.type}`);
        if (stack.buildCommand) {
          const command = stack.buildCommand.replace(/2>&1/g, "").trim();
          this.log(
            "info",
            `[${this.config.agentRole}] Hard Gate (ticket): running ${command}...`
          );
          const execTool = tools.executeCommand as any;
          const gateResult = await execTool.execute({
            command,
            reason: "Hard Gate: Pre-QA compile check (ticket run)",
          });
          const exitCode =
            typeof gateResult?.exitCode === "number" ? gateResult.exitCode : undefined;
          const success = gateResult?.success === true;
          const stdout =
            typeof gateResult?.stdout === "string" ? gateResult.stdout : "";
          const stderr =
            typeof gateResult?.stderr === "string" ? gateResult.stderr : "";
          if (!success || (typeof exitCode === "number" && exitCode !== 0)) {
            ticketHardGateFailed = true;
            ticketHardGateCompileOutput = [stdout, stderr].filter(Boolean).join("\n");
            const truncated =
              ticketHardGateCompileOutput.length > 4000
                ? ticketHardGateCompileOutput.slice(0, 4000) + "\n...[truncated]"
                : ticketHardGateCompileOutput;
            this.log(
              "error",
              `[${this.config.agentRole}] Hard Gate failed (ticket, ${command}): ${truncated}`
            );
          }
        } else {
          this.log(
            "info",
            `[${this.config.agentRole}] Hard Gate (ticket) skipped (no build command for stack: ${stack.type})`
          );
        }
      } catch (gateErr) {
        this.log(
          "info",
          `[${this.config.agentRole}] Hard Gate (ticket) skipped: ${
            gateErr instanceof Error ? gateErr.message : String(gateErr)
          }`
        );
      }

      if (ticketHardGateFailed) {
        const reasoning =
          `Hard Gate: Pre-QA compile check failed (ticket run). Fix the compiler errors and retry.\n\n` +
          (ticketHardGateCompileOutput
            ? `Compiler output:\n${ticketHardGateCompileOutput.slice(0, 8000)}`
            : "No output captured.");
        try {
          await prisma.comment.create({
            data: {
              taskId: relatedTaskId,
              content:
                "### EXECUTION ERRORS ###\n" +
                (ticketHardGateCompileOutput || "No output captured."),
              authorRole: "DEVOPS",
              isSystem: false,
            },
          });
        } catch (commentErr) {
          this.log(
            "error",
            `[${this.config.agentRole}] Failed to create Hard Gate DEVOPS comment: ${
              commentErr instanceof Error ? commentErr.message : String(commentErr)
            }`
          );
        }
        try {
          await this.sendMessage(
            AgentRole.TEAMLEAD,
            MessageType.QA_RESPONSE,
            {
              taskId: relatedTaskId,
              finalStatus: "REJECTED",
              ticketId,
              reasoning,
            },
            message.id
          );
          this.log(
            "info",
            `[${this.config.agentRole}] Sent synthetic QA_RESPONSE(REJECTED) to TEAMLEAD for Hard Gate failure (ticket ${ticketId}, task ${relatedTaskId})`
          );
        } catch (sendErr) {
          this.log(
            "error",
            `[${this.config.agentRole}] Failed to send synthetic QA_RESPONSE for Hard Gate (ticket): ${
              sendErr instanceof Error ? sendErr.message : String(sendErr)
            }`
          );
        }
        return {
          success: false,
          ticketId,
          taskId: relatedTaskId,
          error: "Hard Gate: build/type-check failed",
          compileOutput: ticketHardGateCompileOutput,
        };
      }

      const ticketVerificationNote =
        "\n[Ticket run] automatedCheck from the original task was skipped (implementation was modified by the ticket). Verify based on artifacts and report content only.";

      let reportText = results
        .map((r: any) => {
          if (r.success) {
            const result = r.result as any;
            if (result?.stdout || result?.stderr || result?.error) {
              const output = result.stdout || result.stderr || result.error || "";
              return `- ${r.toolName}: ${output.slice(0, 200)}`;
            }
            return `- ${r.toolName}: completed`;
          }
          return `- ${r.toolName}: failed - ${r.error}`;
        })
        .join("\n");

      const executionErrorLines: string[] = [];
      for (const r of results as any[]) {
        const result: any = r.result;
        const exitCode =
          typeof result?.exitCode === "number" ? result.exitCode : undefined;
        const failed =
          r.success === false || (typeof exitCode === "number" && exitCode !== 0);

        if (!failed) continue;

        const stderr =
          typeof result?.stderr === "string" ? result.stderr : undefined;
        const errorField =
          typeof result?.error === "string" ? result.error : undefined;
        const topLevelError =
          typeof r.error === "string" ? r.error : undefined;

        const rawSnippet = stderr || errorField || topLevelError || "";
        const snippet = rawSnippet
          ? rawSnippet.slice(0, 500)
          : "Unknown error (no stderr or message provided).";

        executionErrorLines.push(
          `- Tool: ${r.toolName}\n  Error: ${snippet}`
        );
      }

      if (executionErrorLines.length > 0) {
        const executionErrorsBlock =
          `\n\n### EXECUTION ERRORS ###\n` +
          `The following commands failed during execution:\n` +
          executionErrorLines.join("\n");
        reportText = reportText ? reportText + executionErrorsBlock : executionErrorsBlock.trimStart();
      }

      if (artifactsOutput || automatedCheckOutput) {
        let verificationSection = "\n\n=== VERIFICATION EVIDENCE ===\n\n";
        if (verifiedArtifactPaths.length > 0) {
          verificationSection +=
            "Verified artifact paths (relative to project root): " +
            verifiedArtifactPaths.join(", ") +
            "\n\nПроверенные пути артефактов:\n" +
            verifiedArtifactPaths.map((p) => "  - " + p).join("\n") +
            "\n\n";
        }
        if (artifactsOutput) {
          verificationSection += "Artifacts Check:\n" + artifactsOutput + "\n\n";
        }
        if (automatedCheckOutput) {
          verificationSection += "Automated Check:\n" + automatedCheckOutput + "\n\n";
        }
        verificationSection += ticketVerificationNote + "\n\n";
        if (hasVerificationFailures) {
          verificationSection += "⚠️ Some verification commands failed. See output above for details.\n";
        } else {
          verificationSection += "✅ All verification commands succeeded.\n";
        }
        reportText = reportText ? reportText + "\n\n" + verificationSection : verificationSection;
      } else {
        reportText = (reportText || "") + ticketVerificationNote;
      }

      const headlessBlockTicket = `

### Manual Check
[HEADLESS MODE TRIGGERED]
В headless-окружении (Docker/Server) ручная проверка (открыть браузер, сделать скриншот) физически невозможна. Проверка отложена на пользователя. QA-агент обязан применить HEADLESS EXCEPTION, если Artifacts и AutomatedCheck пройдены успешно.`;
      reportText = `=== IMPLEMENTATION REPORT (TICKET RUN) ===\n\n${reportText || "No report generated."}${headlessBlockTicket}`;

      try {
        await prisma.comment.create({
          data: {
            taskId: relatedTaskId,
            content: reportText,
            authorRole: "DEVOPS",
            isSystem: false,
          },
        });
        this.log(
          "info",
          `[${this.config.agentRole}] Created implementation report comment for ticket ${ticketId} (task ${relatedTaskId}, length=${reportText.length})`
        );
      } catch (commentErr) {
        this.log("error", `[${this.config.agentRole}] Failed to create report comment: ${commentErr instanceof Error ? commentErr.message : String(commentErr)}`);
      }

      this.log("info", `\n📤 [TASK_EXECUTOR] -> @QA: "Я закончил писать код. Отправил тебе на ревью (QA_REQUEST)."`);
      // Статус тикета выставит TeamLead по результату QA (единый источник истины).
      await this.sendMessage(
        AgentRole.QA,
        MessageType.QA_REQUEST,
        {
          taskId: relatedTaskId,
          report: reportText || "No report generated.",
          ticketId,
        },
        message.id
      );

      this.log("info", `[${this.config.agentRole}] Sent QA_REQUEST for ticket ${ticketId} (task ${relatedTaskId}); ticket status will be set by QA result`);

      return {
        success: true,
        results,
        ticketId,
        taskId: relatedTaskId,
        taskTitle: originalTask.title,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.log(
        "error",
        `[${this.config.agentRole}] Ticket execution failed for ticket ${ticketId} (task ${relatedTaskId}): ${msg}`
      );

      try {
        await prisma.comment.create({
          data: {
            taskId: relatedTaskId,
            content: `**TASK_EXECUTOR CRASH (TICKET RUN)**\n\nTicket ${ticketId}\n\n${msg}`,
            authorRole: "DEVOPS",
            isSystem: true,
          },
        });
      } catch (commentErr) {
        this.log(
          "error",
          `[${this.config.agentRole}] Failed to create crash comment for ticket ${ticketId} (task ${relatedTaskId}): ${
            commentErr instanceof Error ? commentErr.message : String(commentErr)
          }`
        );
      }

      try {
        await this.sendMessage(
          AgentRole.TEAMLEAD,
          MessageType.QA_RESPONSE,
          {
            taskId: relatedTaskId,
            finalStatus: "REJECTED",
            ticketId,
            reasoning: `System crash in TASK_EXECUTOR during ticket run: ${msg}`,
          },
          message.id
        );
        this.log(
          "info",
          `[${this.config.agentRole}] Sent synthetic QA_RESPONSE(REJECTED) to TEAMLEAD for crashed ticket ${ticketId} (task ${relatedTaskId})`
        );
      } catch (sendErr) {
        this.log(
          "error",
          `[${this.config.agentRole}] Failed to send synthetic QA_RESPONSE for crashed ticket ${ticketId} (task ${relatedTaskId}): ${
            sendErr instanceof Error ? sendErr.message : String(sendErr)
          }`
        );
      }

      return {
        success: false,
        ticketId,
        taskId: relatedTaskId,
        error: msg,
      };
    }
  }
}

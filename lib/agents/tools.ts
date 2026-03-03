import { tool } from "ai";
import { z } from "zod";
import { existsSync } from "fs";
import { promises as fs } from "fs";
import { dirname, join, normalize, relative, resolve } from "path";
import { searchSimilar, findRelatedFiles, getFileEntities, searchWithContext } from "@/lib/rag/search";
import { prisma } from "@/lib/prisma";
import { ExecutionSessionManager } from "@/lib/execution/session-manager";
import { getProjectDir } from "@/lib/project-workspace";
import { createHash } from "crypto";
import { exec } from "child_process";
import { promisify } from "util";
import { executeInContainer } from "@/lib/execution/container-manager";
import { generateEmbedding } from "@/lib/ai/embeddings";

const execAsync = promisify(exec);

/** Normalizes file path: strips leading "./" and normalizes slashes. Use for readFile/writeFile params. */
function normalizeFilePath(p: string): string {
  if (!p || typeof p !== "string") return p;
  const trimmed = p.trim().replace(/^\.\/+/, "");
  return normalize(trimmed) || trimmed;
}

/** Remove one layer of surrounding single or double quotes from a path (e.g. '/app/backend' -> /app/backend). */
function stripPathQuotes(s: string): string {
  if (typeof s !== "string" || !s.length) return s;
  const t = s.trim();
  if (t.length >= 2 && ((t.startsWith("'") && t.endsWith("'")) || (t.startsWith('"') && t.endsWith('"')))) {
    return t.slice(1, -1);
  }
  return t;
}

function stripMarkdownFences(raw: string): string {
  if (typeof raw !== "string") return "";
  let text = raw;
  if (!text.includes("```")) return text.trim();

  // Remove leading fenced block marker (optional language).
  text = text.replace(/^```[a-z0-9_-]*\n?/i, "");
  // Remove trailing fenced block marker.
  text = text.replace(/\n?```\s*$/i, "");
  // Fallback: strip any remaining leading/trailing bare ```.
  text = text.replace(/^```|```$/g, "");

  return text.trim();
}

export type RunExecuteCommandFn = (
  command: string
) => Promise<{ success?: boolean; exitCode?: number; stdout?: string; stderr?: string; error?: string }>;

/**
 * Builds a shell command that writes content to path using a heredoc with a unique delimiter.
 * Prepends mkdir -p for the parent dir so the command works when run from project root.
 * Safe for paths with spaces and content containing quotes; delimiter avoids EOF inside content.
 */
function buildHeredocWriteCommand(safePath: string, content: string): string {
  const delim = `WRITEFILE_EOF_${Date.now()}`;
  const pathShell = "'" + safePath.replace(/'/g, "'\"'\"'") + "'";
  const dirPart = safePath.includes("/") ? `mkdir -p "$(dirname ${pathShell})" && ` : "";
  return `${dirPart}cat << '${delim}' > ${pathShell}\n${content}\n${delim}`;
}

type ToolStatus = "SUCCESS" | "EMPTY" | "NOT_FOUND" | "ERROR";

export type AgentLogFunction = (
  level: "info" | "error" | "success",
  message: string
) => void;

type CodebaseSearchResult = {
  filePath: string;
  content: string;
  similarity: number;
};

/**
 * Project-scoped semantic search over all embedded file chunks using pgvector.
 * Returns the most similar chunks with similarity in (0, 1], filtered by a 0.4 threshold.
 */
async function searchCodebaseForProject(
  projectId: string,
  query: string,
  limit = 5
): Promise<CodebaseSearchResult[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];

  const embedding = await generateEmbedding(trimmed);
  if (!embedding) {
    // Embeddings are disabled or failed; fail soft by returning no results.
    return [];
  }

  const vectorQueryString = `[${embedding.join(",")}]`;

  type Row = {
    filePath: string;
    content: string;
    similarity: number;
  };

  const rows = await prisma.$queryRawUnsafe<Row[]>(
    `
      SELECT 
        pf.name as "filePath", 
        fe.content, 
        1 - (fe.embedding::vector <=> $1::vector) AS "similarity"
      FROM "FileEmbedding" fe
      JOIN "ProjectFile" pf ON fe."fileId" = pf.id
      WHERE pf."projectId" = $2 AND 1 - (fe.embedding::vector <=> $1::vector) > 0.4
      ORDER BY "similarity" DESC
      LIMIT $3;
    `,
    vectorQueryString,
    projectId,
    limit
  );

  return rows.map((row) => ({
    filePath: row.filePath,
    content: row.content,
    similarity: row.similarity,
  }));
}

/**
 * Creates a searchKnowledge tool for a specific project.
 * This factory function allows dynamic projectId binding.
 */
export function createSearchKnowledgeTool(
  projectId: string,
  logFunction?: AgentLogFunction
) {
  return tool({
    description: "Search project documentation and files for relevant context",
    parameters: z.object({
      query: z.string()
    }),
    execute: async ({ query }) => {
      const results = await searchSimilar(projectId, query, 5);

      if (logFunction) {
        logFunction(
          "info",
          `🧠 searchGlobalInsights for "${query}" found ${results.length} relevant lessons.`
        );
      }

      return results.map(r => ({
        content: r.content,
        similarity: r.similarity
      }));
    }
  });
}

/**
 * Creates a searchCodebase tool for a specific project.
 * Performs semantic search across all embedded file chunks in the project using pgvector.
 */
export function createSearchCodebaseTool(
  projectId: string,
  logFunction?: AgentLogFunction
) {
  return tool({
    description:
      "Semantic search across the entire project codebase. Use this to find relevant logic, functions, or variables when you are unsure where they are located. " +
      "CRITICAL: The 'limit' parameter MUST NOT exceed 5. If you need more results, narrow your query.",
    parameters: z.object({
      query: z.string().describe("Natural language description of what you are searching for"),
      limit: z
        .number()
        .int()
        .positive()
        .max(5)
        .optional()
        .describe("Maximum number of results to return (default 5). MUST NOT exceed 5."),
    }),
    execute: async ({ query, limit }) => {
      const trimmedQuery = (query || "").trim();
      if (!trimmedQuery) {
        if (logFunction) {
          logFunction("info", "🔍 searchCodebase received empty query; returning 0 results.");
        }
        return [];
      }

      const effectiveLimit =
        typeof limit === "number" && Number.isFinite(limit) ? Math.min(limit, 5) : 5;
      const results = await searchCodebaseForProject(
        projectId,
        trimmedQuery,
        effectiveLimit
      );

      if (logFunction) {
        logFunction(
          "info",
          `🔍 searchCodebase for "${query}" found ${results.length} relevant chunks.`
        );
      }

      return results;
    },
  });
}

/**
 * Web search tool using Tavily API.
 * Enables agents to search the internet for fresh documentation and solutions.
 * Canonical name for ReAct: searchWeb.
 */
export const searchWeb = tool({
  description:
    "Search the internet for fresh documentation, error solutions, and current information. Use when you need up-to-date info (e.g. Tailwind v4, new React patterns) or solutions to compiler/setup errors.",
  parameters: z.object({
    query: z.string().describe("Search query (e.g. 'How to setup Tailwind v4 with Vite', 'Fix TS2307 in React')"),
  }),
  execute: async ({ query }) => {
    const apiKey = process.env.TAVILY_API_KEY;
    if (!apiKey) {
      return { success: false, error: "TAVILY_API_KEY is not set." };
    }

    try {
      const response = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: apiKey,
          query: query,
          search_depth: "basic",
          include_answer: true,
          max_results: 3,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        return { success: false, error: `Tavily API error: ${response.status} - ${errorText}` };
      }

      const data = (await response.json()) as {
        answer?: string;
        results?: Array<{ title?: string; content?: string }>;
      };
      const results = Array.isArray(data.results)
        ? data.results.map((r) => ({ title: r.title ?? "", content: r.content ?? "" }))
        : [];
      return {
        success: true,
        answer: typeof data.answer === "string" ? data.answer : undefined,
        results,
      };
    } catch (e) {
      return {
        success: false,
        error: e instanceof Error ? e.message : String(e),
      };
    }
  },
});

// Backwards-compatible alias: older code may import { webSearch } from "./tools".
export const webSearch = searchWeb;

function getMimeTypeForPath(filePath: string): string {
  const ext = (filePath.split(".").pop() || "").toLowerCase();
  const mimeTypes: Record<string, string> = {
    js: "text/javascript",
    ts: "text/typescript",
    tsx: "text/typescript",
    jsx: "text/javascript",
    json: "application/json",
    md: "text/markdown",
    txt: "text/plain",
    html: "text/html",
    css: "text/css",
    scss: "text/x-scss",
    py: "text/x-python",
    go: "text/x-go",
    rs: "text/x-rust",
  };
  return mimeTypes[ext] || "text/plain";
}

/** Path under project root safe for filesystem read (strip file:// and path traversal). Used for DB-resolved paths. */
function projectRelativePathForFs(pathFromDbOrRequest: string): string {
  return (pathFromDbOrRequest || "")
    .replace(/^file:\/\//i, "")
    .replace(/\.\./g, "")
    .replace(/^\/+/, "")
    .trim();
}

/**
 * Returns a path relative to the project root for display in getCodeMap.
 * Strips host-style prefixes (e.g. projects/<projectId>/) and absolute project root
 * so the LLM only sees paths like src/App.tsx, package.json.
 */
function toProjectRelativePath(path: string, projectId: string): string {
  if (!path || typeof path !== "string") return path;
  const normalized = path.replace(/\\/g, "/").trim();
  const prefixRel = `projects/${projectId}/`;
  if (normalized.startsWith(prefixRel)) {
    const after = normalized.slice(prefixRel.length).replace(/^\/+/, "");
    return after.replace(/\.\./g, "").trim() || normalized;
  }
  const projectDir = getProjectDir(projectId).replace(/\\/g, "/");
  const absDir = resolve(projectDir).replace(/\\/g, "/");
  if (normalized.startsWith(absDir + "/") || normalized === absDir) {
    const after = normalized.slice(absDir.length).replace(/^\/+/, "");
    return after.replace(/\.\./g, "").trim() || normalized;
  }
  const noLeading = normalized.replace(/^\/+/, "").replace(/\.\./g, "").trim();
  return noLeading || normalized;
}

/**
 * Builds project-root-relative safe path the same way as writeFile (cloud/local).
 * Use for readFile so paths match writeFile exactly.
 */
function safeRelativePathForRead(relativePath: string): string {
  const normalized = normalizeFilePath(relativePath).replace(/^file:\/\//i, "");
  return normalized.replace(/\.\./g, "").replace(/^\/+/, "").trim();
}

/** Escapes a path for safe use in shell (e.g. cat 'path'). */
function shellEscapePath(p: string): string {
  return "'" + p.replace(/'/g, "'\"'\"'") + "'";
}

/**
 * Try to read file from project workspace filesystem. Returns content or null if not found/invalid.
 * fullPath must be under rootPath (path traversal not allowed).
 * Uses same path logic as writeFile (safeRelativePathForRead).
 */
async function tryReadFileFromProjectFs(
  projectId: string,
  relativePath: string
): Promise<{ content: string; fullPath: string } | null> {
  const rootPath = getProjectDir(projectId);
  const safeRelative = safeRelativePathForRead(relativePath);
  if (!safeRelative) return null;
  const fullPath = join(rootPath, safeRelative);
  const normalizedFull = normalize(fullPath);
  if (!normalizedFull.startsWith(normalize(rootPath))) return null;
  try {
    const content = await fs.readFile(fullPath, "utf-8");
    return { content, fullPath: safeRelative };
  } catch {
    return null;
  }
}

/**
 * Creates a readFile tool for a specific project.
 * Tries filesystem first (same path as writeFile), then DB; in cloud mode can fall back to executeCommand (cat) when FS and DB miss.
 * In cloud mode with executionSessionId, resolves filePath relative to session CWD (same as executeCommand).
 */
export function createReadFileTool(
  projectId: string,
  options?: {
    logFunction?: AgentLogFunction;
    runExecuteCommand?: RunExecuteCommandFn;
    executionSessionId?: string;
  }
) {
  const logFunction = options?.logFunction;
  const runExecuteCommand = options?.runExecuteCommand;
  const executionSessionId = options?.executionSessionId;

  return tool({
    description: "Read the content of a file from the project. Use this when the executor mentions creating or modifying specific files.",
    parameters: z.object({
      filePath: z.string().describe("Path to the file (e.g., 'src/components/Button.tsx' or 'app/api/route.ts')")
    }),
    execute: async ({ filePath }) => {
      try {
        const normalizedPath = normalizeFilePath(filePath);
        const maxAttempts = 3;
        const retryDelayMs = 200;

        // In cloud mode, resolve path relative to session CWD so readFile matches executeCommand's current directory.
        let pathForFs = safeRelativePathForRead(normalizedPath);
        let catPathForFallback = pathForFs;
        if (runExecuteCommand && executionSessionId) {
          const sessionCwd = cloudSessionCwdMap.get(executionSessionId) ?? CONTAINER_WORKSPACE_ROOT;
          const resolved = resolveCloudPath(sessionCwd, normalizedPath);
          if (resolved) {
            pathForFs = resolved.projectRelativePath;
            catPathForFallback = relative(sessionCwd, resolved.containerPath).replace(/\\/g, "/");
          }
        }

        if (logFunction) {
          const projectDir = getProjectDir(projectId);
          const fullPath = join(projectDir, pathForFs);
          logFunction("info", `[readFile] Looking for: ${fullPath}`);
          logFunction("info", `[readFile] Project dir: ${projectDir}`);
          logFunction("info", `[readFile] File exists: ${existsSync(fullPath)}`);
        }

        // Try filesystem first (same path logic as writeFile) so we don't depend on DB for files on disk.
        const fsFirst = await tryReadFileFromProjectFs(projectId, pathForFs);
        if (fsFirst) {
          return {
            success: true,
            status: "SUCCESS" as ToolStatus,
            filePath: fsFirst.fullPath,
            content: fsFirst.content,
            mimeType: null,
          };
        }

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
          const files = await prisma.projectFile.findMany({
            where: {
              projectId: projectId,
              OR: [
                { name: { contains: normalizedPath, mode: "insensitive" } },
                { url: { contains: normalizedPath, mode: "insensitive" } }
              ]
            }
          });

          if (files.length === 0) {
            if (attempt < maxAttempts - 1) {
              await new Promise((r) => setTimeout(r, retryDelayMs));
              continue;
            }
            // Fallback: file may exist on disk only (e.g. created by cloud writeFile) with no DB row.
            const fsResult = await tryReadFileFromProjectFs(projectId, pathForFs);
            if (fsResult) {
              return {
                success: true,
                status: "SUCCESS" as ToolStatus,
                filePath: fsResult.fullPath,
                content: fsResult.content,
                mimeType: null,
              };
            }
            // Cloud mode: try reading via executeCommand (container has project at cwd) when host FS missed.
            if (runExecuteCommand) {
              const catCommand = `cat ${shellEscapePath(catPathForFallback)}`;
              const execResult = await runExecuteCommand(catCommand);
              if (execResult?.success === true && typeof execResult?.stdout === "string" && execResult.stdout.length >= 0) {
              return {
                success: true,
                status: "SUCCESS" as ToolStatus,
                filePath: pathForFs,
                content: execResult.stdout,
                mimeType: null,
              };
              }
            }
            const allFiles = await prisma.projectFile.findMany({
              where: { projectId },
              select: { name: true, url: true }
            });
            return {
              success: false,
              status: "NOT_FOUND" as ToolStatus,
              error: "File not found.",
              requestedPath: normalizedPath,
              suggestion: "Try one of these files:",
              availableFiles: allFiles.map(f => f.name || f.url).slice(0, 10),
              hint:
                "The file does not exist in the indexed project files. Verify the path using the executeCommand tool (e.g. 'ls') or create it using writeFile.",
            };
          }

          const exactMatch = files.find(f =>
            f.name === normalizedPath ||
            f.name?.endsWith(normalizedPath) ||
            f.url?.endsWith(normalizedPath)
          );

          const file = exactMatch || files[0];
          const resolvedPath = file.name || file.url || normalizedPath;
          const pathForFsFromDb = safeRelativePathForRead(resolvedPath);

          if (file.content == null) {
            if (attempt < maxAttempts - 1) {
              await new Promise((r) => setTimeout(r, retryDelayMs));
              continue;
            }
            // Fallback: read from filesystem when DB has no content (e.g. cloud writeFile did not update DB).
            const fsResult = pathForFsFromDb ? await tryReadFileFromProjectFs(projectId, pathForFsFromDb) : null;
            if (fsResult) {
              return {
                success: true,
                status: "SUCCESS" as ToolStatus,
                filePath: resolvedPath,
                content: fsResult.content,
                mimeType: file.mimeType,
              };
            }
            return {
              success: false,
              status: "ERROR" as ToolStatus,
              error: `File content not available for: ${file.name || file.url}`,
              mimeType: file.mimeType,
              hint:
                "The file exists in the index but its content is not available. The project may need to be re-synced, or this file type might not store content.",
            };
          }

          const content = file.content;

          if (typeof content === "string" && content.trim() === "") {
            // Fallback: file may have been written to disk only (e.g. cloud writeFile).
            const fsResult = pathForFsFromDb ? await tryReadFileFromProjectFs(projectId, pathForFsFromDb) : null;
            if (fsResult) {
              return {
                success: true,
                status: "SUCCESS" as ToolStatus,
                filePath: resolvedPath,
                content: fsResult.content,
                mimeType: file.mimeType,
              };
            }
            return {
              success: true,
              status: "EMPTY" as ToolStatus,
              filePath: resolvedPath,
              content: "",
              mimeType: file.mimeType,
              hint: `Hint: file exists but content is empty in DB. Use executeCommand: cat ${resolvedPath} to read directly.`,
            };
          }

          return {
            success: true,
            status: "SUCCESS" as ToolStatus,
            filePath: resolvedPath,
            content,
            mimeType: file.mimeType,
          };
        }

        const allFiles = await prisma.projectFile.findMany({
          where: { projectId },
          select: { name: true, url: true }
        });
        return {
          success: false,
          status: "NOT_FOUND" as ToolStatus,
          error: "File not found.",
          requestedPath: normalizedPath,
          suggestion: "Try one of these files:",
          availableFiles: allFiles.map(f => f.name || f.url).slice(0, 10),
          hint:
            "The file does not exist in the indexed project files. Verify the path using the executeCommand tool (e.g. 'ls') or create it using writeFile.",
        };
      } catch (error) {
        return {
          success: false,
          status: "ERROR" as ToolStatus,
          error: `Failed to read file: ${
            error instanceof Error ? error.message : String(error)
          }`,
          hint:
            "Reading the file failed due to an internal error. Try again, or re-sync the project files if the problem persists.",
        };
      }
    }
  });
}

function createErrorSignature(command: string, errorMessage: string): string {
  const signature = `${command}:${errorMessage}`;
  return createHash('sha256').update(signature).digest('hex').substring(0, 16);
}

async function handleCommandError(projectId: string, command: string, errorMessage: string, executionSessionId?: string): Promise<void> {
  if (!executionSessionId) {
    console.warn(`[Tools] Command error for '${command}' but no executionSessionId was provided.`);
    return;
  }
  const errorSignature = createErrorSignature(command, errorMessage);
  const sessionManager = ExecutionSessionManager.getInstance();

  await sessionManager.incrementRetryCounter(executionSessionId, errorSignature);

  const { shouldPause, reason } = await sessionManager.checkRetryLimit(executionSessionId, errorSignature);

  if (shouldPause) {
    await sessionManager.pauseSession(executionSessionId);
  }
}

const BUILD_LOG_FILE = "/tmp/build-out.log";

const FORBIDDEN_SILENT_ERROR_MESSAGE =
  "FORBIDDEN: --silent, --quiet, and -s flags are NOT ALLOWED.\n" +
  "Reason: You need to see error messages to debug issues.\n" +
  "Action: Remove the flag and run the command again.\n" +
  'Example: "npm install package" NOT "npm install --silent package"';

/**
 * Returns true if the command contains forbidden flags that hide output (--silent, --quiet, -s).
 * -s is only matched as a standalone flag (word boundary), not inside e.g. --save.
 */
export function hasForbiddenSilentFlag(command: string): boolean {
  const c = (command || "").trim();
  if (!c) return false;
  if (c.includes("--silent")) return true;
  if (c.includes("--quiet")) return true;
  // -s as standalone flag: at start, or after space, or at end
  if (c.endsWith("-s")) return true;
  if (/(^|\s)-s(\s|$)/.test(c)) return true;
  return false;
}

/**
 * Wraps build/type-check commands so all output is written to a temp file and then
 * printed, ensuring Docker/exec buffers return full stderr/stdout.
 */
export function wrapBuildCommandForLogCapture(command: string): string {
  const trimmed = (command || "").trim();
  if (!trimmed || trimmed.includes(BUILD_LOG_FILE) || trimmed.includes("build-out.log")) {
    return trimmed;
  }
  const isBuild =
    trimmed.includes("npm run build") ||
    trimmed.includes("npm run type-check") ||
    /(^|\s)(npx\s+)?tsc(\s|$)/.test(trimmed);
  if (!isBuild) return trimmed;
  const cleanCommand = trimmed.replace(/2>&1/g, "").trim();
  return `( ${cleanCommand} > ${BUILD_LOG_FILE} 2>&1 ); EXIT=$?; cat ${BUILD_LOG_FILE}; exit $EXIT`;
}

function isNpmInstallCommand(command: string): boolean {
  const t = (command || "").trim();
  return /^npm\s+(install|i)(\s|$)/.test(t) || t === "npm install" || t === "npm i";
}

function hasERESOLVE(
  stderr?: string | null,
  stdout?: string | null,
  combinedLogs?: string | null
): boolean {
  const combined = [stderr, stdout, combinedLogs].filter(Boolean).join("\n");
  return combined.includes("ERESOLVE");
}

function withLegacyPeerDeps(command: string): string {
  const c = (command || "").trim();
  if (c.includes("--legacy-peer-deps")) return c;
  return c + " --legacy-peer-deps";
}

const ERESOLVE_SUGGESTION =
  "Retry with --legacy-peer-deps flag: npm install <packages> --legacy-peer-deps";

/** Merges stdout and stderr so the agent always receives full combined output (like 2>&1). */
function mergeStdoutStderr(
  stdout: string | null | undefined,
  stderr: string | null | undefined
): string {
  return [stdout, stderr].filter(Boolean).join("\n").trim();
}

/**
 * Creates an executeCommand tool for a specific project.
 * Allows agents to execute shell commands on the client's machine via the sync client.
 * WARNING: This terminal is STATELESS — 'cd folder' does NOT persist to the next call; chain commands (e.g. 'cd folder && npm run build') or use the cwd parameter.
 */
export function createExecuteCommandTool(
  projectId: string,
  executionSessionId?: string,
  _logFunction?: AgentLogFunction
) {
  return tool({
    description:
      "Execute a shell command on the client's machine (e.g., 'npm test', 'npm run build'). " +
      "WARNING: This terminal is STATELESS. Commands like 'cd folder' do NOT persist to the next tool call. You MUST chain commands in a single call (e.g. 'cd folder && npm run build') every time, or use the optional 'cwd' parameter. " +
      "The command is sent to the sync client and must be approved unless --auto-approve is set.",
    parameters: z.object({
      command: z.string().describe("The shell command to execute (e.g., 'npm test', 'cd app && npm run build')"),
      reason: z.string().optional().describe("Optional explanation of why this command is being executed"),
      cwd: z.string().optional().describe("Working directory for the command (project-root-relative, e.g. 'src' or 'packages/app'). If set, command runs as: cd <cwd> && <command>."),
    }),
    execute: async ({ command, reason, cwd }) => {
      try {
        if (hasForbiddenSilentFlag(command)) {
          console.warn("[executeCommand] Blocked command with forbidden --silent/--quiet/-s flag");
          return {
            success: false,
            exitCode: 1,
            stdout: "",
            stderr: FORBIDDEN_SILENT_ERROR_MESSAGE,
            duration: "0ms",
            error: FORBIDDEN_SILENT_ERROR_MESSAGE,
          };
        }
        let effectiveCommand = command;
        if (typeof cwd === "string" && cwd.trim()) {
          const safeCwd = cwd.trim().replace(/\.\./g, "").replace(/^\/+/, "").trim() || ".";
          effectiveCommand = `cd ${safeCwd.includes(" ") ? `"${safeCwd}"` : safeCwd} && ${command}`;
        }
        const finalCommand = wrapBuildCommandForLogCapture(effectiveCommand);
        const response = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3002'}/api/sync/command`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            projectId,
            command: finalCommand,
            reason,
          }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          const stderr = errorText || "Failed to create command (HTTP error from /api/sync/command).";
          await handleCommandError(projectId, command, stderr, executionSessionId);
          const combined = mergeStdoutStderr("", stderr);
          return {
            success: false,
            exitCode: 1,
            stdout: combined,
            stderr,
            duration: "0ms",
            error: `Failed to create command: ${stderr}`
          };
        }

        const data = await response.json();
        const commandId = data.commandId;

        let commandResult = null;
        let attempts = 0;
        const maxAttempts = 120;

        while (attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 2000));

          const checkResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3002'}/api/sync/command?projectId=${projectId}`, {
            method: 'GET',
          });

          if (!checkResponse.ok) {
            attempts++;
            continue;
          }

          const checkData = await checkResponse.json();

          if (checkData.command && checkData.command.id !== commandId) {
            attempts++;
            continue;
          }

          const commandRecord = await prisma.syncCommand.findUnique({
            where: { id: commandId },
          });

          if (!commandRecord) {
            attempts++;
            continue;
          }

          if (commandRecord.status === "COMPLETED" || commandRecord.status === "FAILED" || commandRecord.status === "REJECTED") {
            commandResult = commandRecord;
            break;
          }

          attempts++;
        }

        const createdAt = new Date();

        if (!commandResult) {
          const timeoutMessage =
            "Command execution timed out. The client may not be running or the command was not approved.";
          const combined = mergeStdoutStderr("", timeoutMessage);
          return {
            success: false,
            exitCode: 1,
            stdout: combined,
            stderr: timeoutMessage,
            duration: `${maxAttempts * 2000}ms`,
            error: timeoutMessage
          };
        }

        const rawExitCode =
          typeof commandResult.exitCode === "number"
            ? commandResult.exitCode
            : undefined;
        const exitCode =
          rawExitCode !== undefined
            ? rawExitCode
            : commandResult.status === "COMPLETED"
            ? 0
            : 1;
        const success = commandResult.status === "COMPLETED" && exitCode === 0;

        if (!success) {
          const shouldRetryEresolve =
            isNpmInstallCommand(command) &&
            hasERESOLVE(commandResult.stderr, commandResult.stdout) &&
            !command.includes("--legacy-peer-deps");

          if (shouldRetryEresolve) {
            const retryCommand = withLegacyPeerDeps(command);
            const retryFinal = wrapBuildCommandForLogCapture(retryCommand);
            const retryResponse = await fetch(
              `${process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3002"}/api/sync/command`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  projectId,
                  command: retryFinal,
                  reason: reason ? `${reason} (retry with --legacy-peer-deps)` : undefined,
                }),
              }
            );
            if (retryResponse.ok) {
              const retryData = await retryResponse.json();
              const retryId = retryData.commandId;
              let retryResult: typeof commandResult | null = null;
              let retryAttempts = 0;
              while (retryAttempts < maxAttempts) {
                await new Promise((r) => setTimeout(r, 2000));
                const checkRes = await fetch(
                  `${process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3002"}/api/sync/command?projectId=${projectId}`
                );
                if (!checkRes.ok) {
                  retryAttempts++;
                  continue;
                }
                const check = await checkRes.json();
                if (check.command && check.command.id !== retryId) {
                  retryAttempts++;
                  continue;
                }
                const record = await prisma.syncCommand.findUnique({
                  where: { id: retryId },
                });
                if (!record) {
                  retryAttempts++;
                  continue;
                }
                if (
                  record.status === "COMPLETED" ||
                  record.status === "FAILED" ||
                  record.status === "REJECTED"
                ) {
                  retryResult = record;
                  break;
                }
                retryAttempts++;
              }
              if (retryResult) {
                const retryExitCode =
                  typeof retryResult.exitCode === "number"
                    ? retryResult.exitCode
                    : retryResult.status === "COMPLETED"
                      ? 0
                      : 1;
                const retrySuccess =
                  retryResult.status === "COMPLETED" && retryExitCode === 0;
                if (!retrySuccess && retryResult.stderr) {
                  await handleCommandError(
                    projectId,
                    retryCommand,
                    retryResult.stderr || `Exit code: ${retryExitCode}`,
                    executionSessionId
                  );
                }
                const retryDuration = `${retryResult.updatedAt.getTime() - createdAt.getTime()}ms`;
                const retryMerged = mergeStdoutStderr(retryResult.stdout, retryResult.stderr);
                return {
                  success: retrySuccess,
                  exitCode: retryExitCode,
                  stdout: retryMerged,
                  stderr: retryResult.stderr,
                  duration: retryDuration,
                  note: "Retried with --legacy-peer-deps due to ERESOLVE",
                  ...(retrySuccess
                    ? {}
                    : {
                        error:
                          retryResult.stderr ||
                          `Command failed with exit code ${retryExitCode}`,
                        ...(hasERESOLVE(retryResult.stderr, retryResult.stdout)
                          ? { suggestion: ERESOLVE_SUGGESTION }
                          : {}),
                      }),
                };
              }
            }
          }

          if (commandResult.stderr || exitCode !== 0) {
            await handleCommandError(
              projectId,
              command,
              commandResult.stderr || `Exit code: ${exitCode}`,
              executionSessionId
            );
          }

          const merged = mergeStdoutStderr(commandResult.stdout, commandResult.stderr);
          const failedResult = {
            success: false as const,
            exitCode,
            stdout: merged,
            stderr: commandResult.stderr,
            duration: `${commandResult.updatedAt.getTime() - createdAt.getTime()}ms`,
            error:
              commandResult.stderr ||
              `Command failed with exit code ${exitCode}`,
            ...(hasERESOLVE(commandResult.stderr, commandResult.stdout)
              ? { suggestion: ERESOLVE_SUGGESTION }
              : {}),
          };
          return failedResult;
        }

        const mergedSuccess = mergeStdoutStderr(commandResult.stdout, commandResult.stderr);
        return {
          success: true,
          exitCode,
          stdout: mergedSuccess,
          stderr: commandResult.stderr,
          duration: `${commandResult.updatedAt.getTime() - createdAt.getTime()}ms`,
        };
      } catch (error) {
        const err = error as any;
        const stdout = typeof err?.stdout === "string" ? err.stdout : "";
        const stderrRaw =
          typeof err?.stderr === "string"
            ? err.stderr
            : err instanceof Error && typeof err.message === "string"
            ? err.message
            : String(err);
        const combinedLogs = `Stdout: ${stdout}\nStderr: ${stderrRaw}`.trim();
        const errorMessage =
          err instanceof Error && typeof err.message === "string"
            ? err.message
            : "Failed to execute command due to an unexpected error.";

        await handleCommandError(projectId, command, stderrRaw, executionSessionId);
        const mergedCatch = mergeStdoutStderr(stdout, stderrRaw);
        return {
          success: false,
          exitCode: 1,
          stdout: mergedCatch || combinedLogs,
          stderr: stderrRaw,
          duration: "0ms",
          error:
            combinedLogs.length > 0
              ? `Failed to execute command: ${errorMessage}\nLogs:\n${combinedLogs}`
              : `Failed to execute command: ${errorMessage}`
        };
      }
    }
  });
}

const EXECUTE_COMMAND_LOG_SNIPPET_LEN = 200;

/** Container workspace root (matches docker -w and mount in container-manager). */
const CONTAINER_WORKSPACE_ROOT = "/app";

/** Matches scaffolding commands that may create a named subfolder (create-vite, create-next-app, cargo new <name>, etc.). */
const SCAFFOLD_COMMAND_PATTERN =
  /(?:npm\s+create|npx\s+create)\s+(?:vite|next-app|react-app|svelte|vue)(?:\s|@|$)|create-vite|create-next-app|cargo\s+new\s+(?!\.\s*$)[^\s]+|django-admin\s+startproject\s+(?!\.\s*$)[^\s]+|rails\s+new\s+[^\s]+/i;

/** Shell script run from /app after a scaffold command: if no manifest in ., move single manifest-containing subdir contents to root. */
const FLATTEN_AFTER_SCAFFOLD_CMD =
  `cd ${CONTAINER_WORKSPACE_ROOT} && ( test -f package.json || test -f Cargo.toml || test -f go.mod || test -f pyproject.toml ) && exit 0; set -- */; [ -d "\$1" ] || exit 0; sub="\${1%/}"; ( test -f "\$sub/package.json" || test -f "\$sub/Cargo.toml" || test -f "\$sub/go.mod" || test -f "\$sub/pyproject.toml" ) && ( mv "\$sub"/* . 2>/dev/null; mv "\$sub"/.[!.]* . 2>/dev/null; rm -rf "\$sub"; echo "[Flatten] Moved project from subfolder to root." )`;

function isScaffoldCommand(command: string): boolean {
  return SCAFFOLD_COMMAND_PATTERN.test(command);
}

/** Per-session current working directory in cloud execute. Key: executionSessionId. */
const cloudSessionCwdMap = new Map<string, string>();

/** Convert container absolute path to project-relative (strip CONTAINER_WORKSPACE_ROOT). */
function containerPathToProjectRelative(containerPath: string): string {
  const normalized = containerPath.replace(/\\/g, "/").trim();
  if (!normalized.startsWith(CONTAINER_WORKSPACE_ROOT + "/") && normalized !== CONTAINER_WORKSPACE_ROOT) {
    return normalized.replace(/^\/+/, "");
  }
  return normalized.slice(CONTAINER_WORKSPACE_ROOT.length).replace(/^\/+/, "") || "";
}

/**
 * Resolve agent filePath relative to session CWD to container path and project-relative path.
 * Returns null if the resolved path escapes outside CONTAINER_WORKSPACE_ROOT.
 */
function resolveCloudPath(
  sessionCwd: string,
  filePath: string
): { containerPath: string; projectRelativePath: string } | null {
  const trimmed = (filePath || "").trim();
  const parsedPath = stripPathQuotes(trimmed);
  if (!parsedPath) return null;
  const containerPath = normalize(resolve(sessionCwd, parsedPath));
  if (containerPath !== CONTAINER_WORKSPACE_ROOT && !containerPath.startsWith(CONTAINER_WORKSPACE_ROOT + "/")) {
    return null;
  }
  const projectRelativePath = containerPathToProjectRelative(containerPath);
  return { containerPath, projectRelativePath };
}

/**
 * Normalize CWD so host-style paths like /app/projects/<id> (invalid inside container) become /app.
 * Inside the container the project root is always CONTAINER_WORKSPACE_ROOT; there is no /app/projects/.
 * Clamp so CWD never escapes /app (e.g. path.resolve('/app', '..') -> / gets reset to /app).
 */
function normalizeContainerCwd(cwd: string): string {
  const trimmed = cwd.trim();
  if (trimmed === CONTAINER_WORKSPACE_ROOT) return CONTAINER_WORKSPACE_ROOT;
  if (trimmed.startsWith(`${CONTAINER_WORKSPACE_ROOT}/projects/`)) return CONTAINER_WORKSPACE_ROOT;
  if (trimmed !== CONTAINER_WORKSPACE_ROOT && !trimmed.startsWith(CONTAINER_WORKSPACE_ROOT + "/")) {
    return CONTAINER_WORKSPACE_ROOT;
  }
  return trimmed || CONTAINER_WORKSPACE_ROOT;
}

function escapeShellPath(p: string): string {
  return "'" + p.replace(/'/g, "'\\''") + "'";
}

function resolveContainerCwd(current: string, pathArg: string): string {
  const parsedPath = stripPathQuotes(pathArg.trim());
  if (!parsedPath) return current;
  const resolved = resolve(current, parsedPath);
  return resolved || current;
}

/**
 * If command starts with "cd <path>" (optional "&& rest"), returns { newCwd, effectiveCommand }.
 * Otherwise returns null (no cd parsed).
 */
function parseLeadingCd(command: string, currentCwd: string): { newCwd: string; effectiveCommand: string } | null {
  const m = /^\s*cd\s+([^\s;&|]+)(?:\s*(&&|\;)\s*([\s\S]*))?/.exec(command);
  if (!m) return null;
  const pathArg = m[1].trim();
  const parsedPath = stripPathQuotes(pathArg);
  const rest = (m[3] ?? "").trim();
  const newCwd = resolveContainerCwd(currentCwd, parsedPath);
  return { newCwd, effectiveCommand: rest || "true" };
}

/**
 * Creates a cloudExecuteCommand tool for a specific project.
 * Executes shell commands inside the Docker container; working directory is /app and persists per session (or use cwd to set it).
 */
export function createCloudExecuteCommandTool(
  projectId: string,
  executionSessionId?: string,
  logFunction?: AgentLogFunction
) {
  return tool({
    description:
      "Execute a shell command inside the cloud workspace (Docker container). Working directory is /app and persists across calls for this session. " +
      "Optional 'cwd' sets the directory for this (and subsequent) commands. Use for tests, builds, or tools in cloud mode.",
    parameters: z.object({
      command: z.string().describe("The shell command to execute (e.g., 'npm test', 'npm run build')"),
      reason: z.string().optional().describe("Optional explanation of why this command is being executed"),
      cwd: z.string().optional().describe("Working directory for this command (project-root-relative, e.g. 'src'). If set, runs from that directory and remembers it for the session."),
    }),
    execute: async ({ command, reason, cwd: cwdParam }) => {
      if (logFunction) {
        logFunction("info", `[executeCommand] cloud: command=${String(command).slice(0, 80)}... hasSession=${Boolean(executionSessionId)}`);
      }
      if (hasForbiddenSilentFlag(command)) {
        console.warn("[executeCommand] Blocked command with forbidden --silent/--quiet/-s flag");
        return {
          success: false,
          exitCode: 1,
          stdout: "",
          stderr: FORBIDDEN_SILENT_ERROR_MESSAGE,
          duration: "0ms",
          error: FORBIDDEN_SILENT_ERROR_MESSAGE,
        };
      }
      const startTime = Date.now();

      if (!executionSessionId) {
        const msg = "Cloud executeCommand requires a valid executionSessionId.";
        await handleCommandError(projectId, command, msg, executionSessionId);
        return {
          success: false,
          exitCode: 1,
          stdout: "",
          stderr: msg,
          duration: "0ms",
          error: msg,
        };
      }

      let sessionCwd = normalizeContainerCwd(cloudSessionCwdMap.get(executionSessionId) ?? CONTAINER_WORKSPACE_ROOT);
      if (typeof cwdParam === "string" && cwdParam.trim()) {
        sessionCwd = normalizeContainerCwd(resolveContainerCwd(sessionCwd, cwdParam.trim()));
        cloudSessionCwdMap.set(executionSessionId, sessionCwd);
      }
      const cdParsed = parseLeadingCd(command, sessionCwd);
      let effectiveCommand: string;
      if (cdParsed) {
        sessionCwd = normalizeContainerCwd(cdParsed.newCwd);
        cloudSessionCwdMap.set(executionSessionId, sessionCwd);
        effectiveCommand = cdParsed.effectiveCommand;
      } else {
        effectiveCommand = command;
      }
      const commandWithCwd = `cd ${escapeShellPath(sessionCwd)} && ${effectiveCommand}`;
      const finalCommand = wrapBuildCommandForLogCapture(commandWithCwd);
      try {
        const result = await executeInContainer(executionSessionId, finalCommand);
        const duration = `${Date.now() - startTime}ms`;
        const success = result.exitCode === 0;
        const effectiveExitCode = result.exitCode ?? (success ? 0 : 1);
        const primaryLogs =
          (typeof result.combinedLogs === "string" &&
            result.combinedLogs.trim()) ||
          result.stderr ||
          result.stdout ||
          "";
        if (logFunction) {
          const out = (result.stdout ?? "").slice(0, EXECUTE_COMMAND_LOG_SNIPPET_LEN);
          const err = (result.stderr ?? "").slice(0, EXECUTE_COMMAND_LOG_SNIPPET_LEN);
          if (out) logFunction("info", `[executeCommand] stdout: ${out}${(result.stdout?.length ?? 0) > EXECUTE_COMMAND_LOG_SNIPPET_LEN ? "..." : ""}`);
          if (err) logFunction("info", `[executeCommand] stderr: ${err}${(result.stderr?.length ?? 0) > EXECUTE_COMMAND_LOG_SNIPPET_LEN ? "..." : ""}`);
        }

        if (!success) {
          const shouldRetryEresolve =
            isNpmInstallCommand(command) &&
            hasERESOLVE(result.stderr, result.stdout, result.combinedLogs) &&
            !command.includes("--legacy-peer-deps");

          if (shouldRetryEresolve) {
            const retryCommand = withLegacyPeerDeps(command);
            let retrySessionCwd = normalizeContainerCwd(cloudSessionCwdMap.get(executionSessionId) ?? CONTAINER_WORKSPACE_ROOT);
            const retryCdParsed = parseLeadingCd(retryCommand, retrySessionCwd);
            const retryCwd = retryCdParsed ? normalizeContainerCwd(retryCdParsed.newCwd) : retrySessionCwd;
            if (retryCdParsed) cloudSessionCwdMap.set(executionSessionId, retryCwd);
            const retryEffective = retryCdParsed ? retryCdParsed.effectiveCommand : retryCommand;
            const retryCommandWithCwd = `cd ${escapeShellPath(retryCwd)} && ${retryEffective}`;
            const retryFinal = wrapBuildCommandForLogCapture(retryCommandWithCwd);
            try {
              const retryResult = await executeInContainer(
                executionSessionId,
                retryFinal
              );
              const retryDuration = `${Date.now() - startTime}ms`;
              const retrySuccess = retryResult.exitCode === 0;
              const retryExitCode =
                retryResult.exitCode ?? (retrySuccess ? 0 : 1);
              if (!retrySuccess) {
                await handleCommandError(
                  projectId,
                  retryCommand,
                  retryResult.stderr ||
                    retryResult.stdout ||
                    `Exit code ${retryExitCode}`,
                  executionSessionId
                );
              }
              const retryMerged =
                retryResult.combinedLogs?.trim() ||
                mergeStdoutStderr(retryResult.stdout, retryResult.stderr);
              return {
                success: retrySuccess,
                exitCode: retryExitCode,
                stdout: retryMerged,
                stderr: retryResult.stderr,
                duration: retryDuration,
                combinedLogs: retryResult.combinedLogs,
                note: "Retried with --legacy-peer-deps due to ERESOLVE",
                ...(retrySuccess
                  ? {}
                  : {
                      error:
                        retryResult.stderr ||
                        retryResult.stdout ||
                        `Cloud command failed with exit code ${retryExitCode} in ${retryDuration}`,
                      ...(hasERESOLVE(
                        retryResult.stderr,
                        retryResult.stdout,
                        retryResult.combinedLogs
                      )
                        ? { suggestion: ERESOLVE_SUGGESTION }
                        : {}),
                    }),
              };
            } catch {
              // fall through to normal failed return below
            }
          }

          await handleCommandError(
            projectId,
            command,
            primaryLogs || `Exit code: ${effectiveExitCode}`,
            executionSessionId
          );
        }

        let mergedStdout =
          result.combinedLogs?.trim() ||
          mergeStdoutStderr(result.stdout, result.stderr);

        if (success && isScaffoldCommand(command)) {
          try {
            const flattenResult = await executeInContainer(
              executionSessionId,
              FLATTEN_AFTER_SCAFFOLD_CMD
            );
            if (flattenResult.exitCode === 0 && flattenResult.stdout?.includes("[Flatten]")) {
              mergedStdout = mergedStdout + "\n\n" + (flattenResult.stdout?.trim() ?? "");
              if (logFunction) {
                logFunction("info", "[executeCommand] Post-scaffold flatten: moved subfolder contents to project root.");
              }
            }
          } catch {
            // Non-fatal: flatten best-effort
          }
        }

        return {
          success,
          exitCode: effectiveExitCode,
          stdout: mergedStdout,
          stderr: result.stderr,
          duration,
          combinedLogs: result.combinedLogs,
          ...(success
            ? {}
            : {
                error:
                  primaryLogs ||
                  `Cloud command failed with exit code ${effectiveExitCode} in ${duration}`,
                ...(hasERESOLVE(
                  result.stderr,
                  result.stdout,
                  result.combinedLogs
                )
                  ? { suggestion: ERESOLVE_SUGGESTION }
                  : {}),
              }),
        };
      } catch (error: any) {
        const rawStdout =
          typeof error?.stdout === "string" ? error.stdout : "";
        const rawStderr =
          typeof error?.stderr === "string" ? error.stderr : "";
        const baseMessage =
          typeof error?.message === "string"
            ? error.message
            : "Cloud command failed with unexpected error.";

        const logsParts: string[] = [];
        if (rawStdout) {
          logsParts.push(`Stdout:\n${rawStdout}`);
        }
        if (rawStderr) {
          logsParts.push(`Stderr:\n${rawStderr}`);
        }
        const logsSection = logsParts.join("\n\n");

        const errMsg =
          logsSection.length > 0
            ? `${baseMessage}\nLogs:\n${logsSection}`
            : baseMessage;

        await handleCommandError(
          projectId,
          command,
          errMsg,
          executionSessionId
        );

        const duration = `${Date.now() - startTime}ms`;
        const catchMerged = mergeStdoutStderr(rawStdout, rawStderr) || errMsg;

        return {
          success: false,
          exitCode: 1,
          stdout: catchMerged,
          stderr: errMsg,
          duration,
          error: errMsg,
        };
      }
    },
  });
}

/**
 * Creates a cloud writeFile tool that writes directly to the project workspace.
 * Always uses heredoc via runExecuteCommand to avoid truncation.
 * Parent directories are always created (host + heredoc in container) so the LLM never needs to mkdir first.
 */
export function createCloudWriteFileTool(
  projectId: string,
  executionSessionId?: string,
  runExecuteCommand?: RunExecuteCommandFn
) {
  return tool({
    description:
      "Create or update a file in the cloud project workspace (/app/projects/[projectId]). Use this for creating new files or updating existing ones in cloud mode.",
    parameters: z.object({
      filePath: z.string().describe("Path to file to write (e.g., 'index.html', 'src/App.tsx')"),
      content: z.string().describe("The content to write to file"),
      reason: z.string().optional().describe("Optional explanation of why this file is being created or modified"),
    }),
    execute: async ({ filePath, content }) => {
      try {
        if (typeof filePath !== "string" || !filePath.trim()) {
          return {
            success: false,
            error: "writeFile (cloud) requires a non-empty string filePath",
          };
        }
        if (typeof content !== "string") {
          return {
            success: false,
            error: "writeFile (cloud) requires 'content' to be a string",
          };
        }

        const normalizedContent = stripMarkdownFences(content ?? "");

        const normalized = normalizeFilePath(filePath);
        if (!normalized || typeof normalized !== "string") {
          return {
            success: false,
            error: "writeFile (cloud) received an invalid filePath",
          };
        }

        // Resolve path relative to session CWD (same as executeCommand) so writes match where the agent is.
        const sessionCwd = executionSessionId
          ? (cloudSessionCwdMap.get(executionSessionId) ?? CONTAINER_WORKSPACE_ROOT)
          : CONTAINER_WORKSPACE_ROOT;
        const resolved = resolveCloudPath(sessionCwd, normalized);
        if (!resolved) {
          return {
            success: false,
            error: "writeFile (cloud) path resolved outside workspace; use project-root-relative paths.",
          };
        }
        const { containerPath, projectRelativePath } = resolved;
        const rootPath = resolve(getProjectDir(projectId));
        const fullPath = resolve(rootPath, projectRelativePath);

        if (!fullPath.startsWith(rootPath)) {
          return { success: false, error: "Invalid path: only project-root-relative paths are allowed; path traversal is not allowed." };
        }

        if (typeof runExecuteCommand !== "function") {
          return {
            success: false,
            error: "writeFile (cloud) requires runExecuteCommand to be available.",
          };
        }

        // Create parent dirs on host so the path exists before heredoc runs in container (same mount).
        const parentDir = dirname(fullPath);
        await fs.mkdir(parentDir, { recursive: true });
        const pathRelativeToSession = relative(sessionCwd, containerPath).replace(/\\/g, "/");
        const heredocCmd = `cd ${escapeShellPath(sessionCwd)} && ${buildHeredocWriteCommand(pathRelativeToSession, normalizedContent)}`;
        const execResult = await runExecuteCommand(heredocCmd);
        const success = execResult?.success === true;
        if (!success) {
          return {
            success: false,
            error: execResult?.error ?? execResult?.stderr ?? "Heredoc write failed",
            exitCode: execResult?.exitCode,
            stdout: execResult?.stdout,
            stderr: execResult?.stderr,
          };
        }
        // Keep DB in sync so readFile sees content (store project-relative path for getCodeMap consistency).
        const existing = await prisma.projectFile.findFirst({
          where: { projectId, name: projectRelativePath },
        });
        if (existing) {
          await prisma.projectFile.update({
            where: { id: existing.id },
            data: { content: normalizedContent },
          });
        } else {
          await prisma.projectFile.create({
            data: {
              projectId,
              name: projectRelativePath,
              url: `file://${projectRelativePath}`,
              mimeType: getMimeTypeForPath(projectRelativePath),
              content: normalizedContent,
            },
          });
        }
        const bytesWritten = Buffer.byteLength(normalizedContent, "utf-8");
        const linesWritten = normalizedContent.split("\n").length;
        return {
          success: true,
          filePath: projectRelativePath,
          message: `File ${projectRelativePath} written successfully`,
          bytesWritten,
          linesWritten,
        };
      } catch (error) {
        return {
          success: false,
          error: `Failed to write file: ${error instanceof Error ? error.message : String(error)}`,
        };
      }
    },
  });
}

/**
 * Creates a getCodeMap tool for a specific project.
 * Returns a high-level structural map: file paths and their code entities (classes, functions, etc.) without code body.
 */
export function createGetCodeMapTool(
  projectId: string,
  logFunction?: AgentLogFunction
) {
  return tool({
    description:
      "Get a high-level structural map of the project: file paths and their code entities (classes, functions, exported constants). Use this at the start of a task to locate which files contain the logic you need. " +
      "CRITICAL: The 'depth' parameter must be between 0 and 10 (inclusive). Omit or 0 = all files; 1–10 = max path depth in segments.",
    parameters: z.object({
      depth: z
        .number()
        .int()
        .min(0)
        .max(10)
        .optional()
        .describe("Optional. 0 or omit = all files; 1–10 = only files with at most that many path segments. MUST be between 0 and 10."),
    }),
    execute: async ({ depth }) => {
      try {
        const files = await prisma.projectFile.findMany({
          where: { projectId },
          orderBy: { name: "asc" },
          select: { id: true, name: true },
        });

        let filesToShow = files;
        if (depth != null && depth > 0) {
          const maxSegments = depth;
          filesToShow = files.filter((f) => {
            const rel = toProjectRelativePath(f.name || "", projectId);
            const segments = rel.split("/").filter(Boolean).length;
            return segments <= maxSegments;
          });
        }

        const lines: string[] = ["# Project code map", ""];
        let entityCount = 0;

        for (const file of filesToShow) {
          const relPath = toProjectRelativePath(file.name || "", projectId);
          const entities = await prisma.codeEntity.findMany({
            where: { fileId: file.id },
            orderBy: [{ type: "asc" }, { startLine: "asc" }],
            select: { name: true, type: true },
          });
          lines.push(`## ${relPath}`);
          if (entities.length === 0) {
            lines.push("- (no entities indexed)");
          } else {
            for (const e of entities) {
              lines.push(`- ${e.type}: ${e.name}`);
              entityCount += 1;
            }
          }
          lines.push("");
        }

        const codeMap = lines.join("\n").trim();
        const isEmptyMap =
          !codeMap ||
          codeMap === "# Project code map" ||
          filesToShow.length === 0;

        if (isEmptyMap) {
          if (logFunction) {
            logFunction(
              "info",
              "🗺️ getCodeMap returned map for 0 files and 0 entities."
            );
          }

          return {
            success: true,
            status: "EMPTY" as ToolStatus,
            codeMap: "# Project code map\n\nNo files or entities indexed yet. Sync project files first.",
            fileCount: 0,
            entityCount: 0,
            hint:
              "The code map is empty. The directory might be empty, files may not be synced yet, or certain files are ignored. Use the executeCommand tool with 'ls -la' to verify which files exist on disk.",
          };
        }

        if (logFunction) {
          logFunction(
            "info",
            `🗺️ getCodeMap returned map for ${filesToShow.length} files and ${entityCount} entities.`
          );
        }

        return {
          success: true,
          status: "SUCCESS" as ToolStatus,
          codeMap,
          fileCount: filesToShow.length,
          entityCount,
        };
      } catch (error) {
        return {
          success: false,
          status: "ERROR" as ToolStatus,
          error: `Failed to build code map: ${
            error instanceof Error ? error.message : String(error)
          }`,
          hint:
            "Building the code map failed due to an internal error. Try syncing the project files again, then re-run getCodeMap.",
        };
      }
    },
  });
}

/**
 * Creates a findRelatedFiles tool for a specific project.
 * Allows agents to find files that depend on or are used by a given file.
 */
export function createFindRelatedFilesTool() {
  return tool({
    description: "Find files that are related to a specific file through code dependencies (imports, exports, function calls). This is useful when modifying a file to understand which other files might be affected.",
    parameters: z.object({
      filePath: z.string().describe("Path to the file (e.g., 'src/components/Button.tsx' or 'app/api/route.ts')")
    }),
    execute: async ({ filePath }) => {
      try {
        const files = await prisma.projectFile.findMany({
          where: {
            OR: [
              { name: { contains: filePath, mode: "insensitive" } },
              { url: { contains: filePath, mode: "insensitive" } }
            ]
          }
        });

        if (files.length === 0) {
          return {
            error: `File not found: ${filePath}`,
            suggestion: "The file may not exist in the project or hasn't been synced yet."
          };
        }

        const exactMatch = files.find(f => 
          f.name === filePath || 
          f.name.endsWith(filePath) || 
          f.url.endsWith(filePath)
        );

        const file = exactMatch || files[0];

        const relatedFiles = await findRelatedFiles(file.id);

        if (relatedFiles.length === 0) {
          return {
            filePath: file.name,
            message: "No related files found. This file may not have any dependencies or may not be imported by other files.",
            entities: await getFileEntities(file.id)
          };
        }

        return {
          filePath: file.name,
          relatedFiles: relatedFiles.map(rf => ({
            fileName: rf.fileName,
            relationship: rf.relationship,
            entityName: rf.entityName
          })),
          entities: await getFileEntities(file.id)
        };
      } catch (error) {
        return {
          error: `Failed to find related files: ${error instanceof Error ? error.message : String(error)}`
        };
      }
    }
  });
}

/**
 * Creates a writeFile tool for a specific project.
 * When runExecuteCommand is provided, uses heredoc (includes mkdir -p) so parent dirs are created; otherwise sync WRITE_FILE (sync-client.js does mkdirSync recursive).
 */
export function createWriteFileTool(
  projectId: string,
  executionSessionId?: string,
  runExecuteCommand?: RunExecuteCommandFn
) {
  return tool({
    description: "Create or update a file on the client's machine via sync client. The file will be written to disk after user approval (unless auto-approve is enabled). Use this for creating new files or updating existing ones.",
    parameters: z.object({
      filePath: z.string().describe("Path to file to write (e.g., 'src/components/Button.tsx' or 'app/api/route.ts')"),
      content: z.string().describe("The content to write to file"),
      reason: z.string().optional().describe("Optional explanation of why this file is being created or modified")
    }),
    execute: async ({ filePath, content, reason }) => {
      try {
        if (typeof filePath !== "string" || !filePath.trim()) {
          return {
            success: false,
            error: "writeFile requires a non-empty string filePath",
          };
        }
        if (typeof content !== "string") {
          return {
            success: false,
            error: "writeFile requires 'content' to be a string",
          };
        }

        const normalizedContent = stripMarkdownFences(content ?? "");

        const normalizedPath = normalizeFilePath(filePath);
        const safePath = normalizedPath.replace(/\.\./g, "").replace(/^\/+/, "").trim();
        if (!safePath) {
          return {
            success: false,
            error: "writeFile requires a valid project-root-relative path (no '..' or absolute paths).",
          };
        }

        if (typeof runExecuteCommand === "function") {
          // Heredoc includes mkdir -p for parent dir; runs in project root.
          const heredocCmd = buildHeredocWriteCommand(safePath, normalizedContent);
          const execResult = await runExecuteCommand(heredocCmd);
          const success = execResult?.success === true;
          const bytesWritten = Buffer.byteLength(normalizedContent, "utf-8");
          const linesWritten = normalizedContent.split("\n").length;
          return {
            success,
            filePath: safePath,
            exitCode: execResult?.exitCode,
            stdout: execResult?.stdout,
            stderr: execResult?.stderr,
            message: success ? `File ${safePath} written successfully` : `Failed to write ${safePath}`,
            ...(success ? { bytesWritten, linesWritten } : { error: execResult?.error ?? execResult?.stderr }),
          };
        }

        const response = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3002'}/api/sync/command`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            projectId,
            command: `Write file: ${safePath}`,
            reason,
            type: 'WRITE_FILE',
            filePath: safePath,
            fileContent: normalizedContent,
          }),
        });

        if (!response.ok) {
          const error = await response.text();
          await handleCommandError(projectId, `Write file: ${safePath}`, error, executionSessionId);
          return {
            success: false,
            error: `Failed to create write file command: ${error}`
          };
        }

        const data = await response.json();
        const commandId = data.commandId;

        let commandResult = null;
        let attempts = 0;
        const maxAttempts = 120;

        while (attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 2000));

          const checkResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3002'}/api/sync/command?projectId=${projectId}`, {
            method: 'GET',
          });

          if (!checkResponse.ok) {
            attempts++;
            continue;
          }

          const checkData = await checkResponse.json();

          if (checkData.command && checkData.command.id !== commandId) {
            attempts++;
            continue;
          }

          const commandRecord = await prisma.syncCommand.findUnique({
            where: { id: commandId },
          });

          if (!commandRecord) {
            attempts++;
            continue;
          }

          if (commandRecord.status === "COMPLETED" || commandRecord.status === "FAILED" || commandRecord.status === "REJECTED") {
            commandResult = commandRecord;
            break;
          }

          attempts++;
        }

        if (!commandResult) {
          return {
            success: false,
            error: "File write timed out. The client may not be running or command was not approved."
          };
        }

        const success = commandResult.status === "COMPLETED";

        if (!success && commandResult.stderr) {
          await handleCommandError(projectId, `Write file: ${safePath}`, commandResult.stderr, executionSessionId);
        }

        const bytesWritten = Buffer.byteLength(normalizedContent, "utf-8");
        const linesWritten = normalizedContent.split("\n").length;
        return {
          success,
          filePath: safePath,
          exitCode: commandResult.exitCode ?? undefined,
          stdout: commandResult.stdout,
          stderr: commandResult.stderr,
          message: success ? `File ${safePath} written successfully` : `Failed to write ${safePath}`,
          ...(success ? { bytesWritten, linesWritten } : {}),
        };
      } catch (error) {
        const normalizedForError =
          typeof filePath === "string" && filePath.trim()
            ? normalizeFilePath(filePath)
            : "(unknown-path)";
        await handleCommandError(
          projectId,
          `Write file: ${normalizedForError}`,
          (error as Error).message,
          executionSessionId
        );
        return {
          success: false,
          error: `Failed to write file: ${error instanceof Error ? error.message : String(error)}`
        };
      }
    }
  });
}

/**
 * Creates a replaceInFile tool for a specific project.
 * Performs surgical, search-and-replace edits inside a single file without rewriting unrelated content.
 * Uses the same path semantics as writeFile/readFile and delegates the final write to the appropriate writeFile tool.
 */
export function createReplaceInFileTool(
  projectId: string,
  mode: "local" | "cloud" = "local",
  options?: {
    logFunction?: AgentLogFunction;
    runExecuteCommand?: RunExecuteCommandFn;
    executionSessionId?: string;
  }
) {
  const useCloud = mode === "cloud";
  const logFunction = options?.logFunction;
  const runExecuteCommand = options?.runExecuteCommand;
  const executionSessionId = options?.executionSessionId;

  return tool({
    description:
      "Surgically replace occurrences of a specific text block inside a file. " +
      "Use this for SMALL or PARTIAL edits in existing files instead of rewriting the whole file with writeFile.",
    parameters: z.object({
      path: z
        .string()
        .describe(
          "Path to the file to edit (project-root-relative, e.g. 'src/components/Button.tsx' or 'app/api/route.ts')"
        ),
      searchString: z
        .string()
        .describe(
          "Exact text to replace. Must match the current file content exactly, including whitespace and indentation."
        ),
      replaceString: z
        .string()
        .describe("Replacement text that will be inserted in place of every occurrence of searchString."),
    }),
    execute: async ({ path, searchString, replaceString }) => {
      try {
        if (typeof path !== "string" || !path.trim()) {
          return {
            success: false,
            error: "replaceInFile requires a non-empty string path",
          };
        }
        if (typeof searchString !== "string" || searchString.length === 0) {
          return {
            success: false,
            error: "replaceInFile requires a non-empty searchString",
          };
        }

        const normalizedPath = normalizeFilePath(path);
        if (!normalizedPath || typeof normalizedPath !== "string") {
          return {
            success: false,
            error: "replaceInFile received an invalid path",
          };
        }

        let projectRelativePath: string;

        if (useCloud) {
          if (!executionSessionId) {
            return {
              success: false,
              error: "replaceInFile (cloud) requires a valid executionSessionId.",
            };
          }
          const sessionCwd =
            cloudSessionCwdMap.get(executionSessionId) ?? CONTAINER_WORKSPACE_ROOT;
          const resolved = resolveCloudPath(sessionCwd, normalizedPath);
          if (!resolved) {
            return {
              success: false,
              error:
                "replaceInFile (cloud) path resolved outside workspace; use project-root-relative paths under /app.",
            };
          }
          projectRelativePath = resolved.projectRelativePath;
        } else {
          projectRelativePath = normalizedPath.replace(/\.\./g, "").replace(/^\/+/, "").trim();
        }

        if (!projectRelativePath) {
          return {
            success: false,
            error:
              "replaceInFile requires a valid project-root-relative path (no '..' or absolute paths).",
          };
        }

        // Read current file content from the project filesystem.
        const fsResult = await tryReadFileFromProjectFs(projectId, projectRelativePath);
        if (!fsResult) {
          if (logFunction) {
            logFunction(
              "error",
              `[replaceInFile] File not found for path: ${projectRelativePath}`
            );
          }
          return {
            success: false,
            error: `File not found: ${projectRelativePath}`,
          };
        }

        const content = fsResult.content;
        if (!content.includes(searchString)) {
          return {
            success: false,
            error:
              "Error: searchString not found in file. Ensure you copied the exact text, including whitespace and indentation.",
            filePath: projectRelativePath,
          };
        }

        const parts = content.split(searchString);
        const occurrences = parts.length - 1;
        const newContent = parts.join(replaceString);

        let warning: string | undefined;
        if (occurrences > 1) {
          warning = "Warning: Replaced multiple occurrences";
          if (logFunction) {
            logFunction(
              "info",
              `[replaceInFile] ${warning} of searchString in ${projectRelativePath} (count=${occurrences})`
            );
          }
        }

        // Delegate final write to the appropriate writeFile tool so behavior matches other tools.
        const writeTool = useCloud
          ? createCloudWriteFileTool(projectId, executionSessionId, runExecuteCommand)
          : createWriteFileTool(projectId, executionSessionId, runExecuteCommand);

        const writeResult = await (writeTool as any).execute(
          { filePath: projectRelativePath, content: newContent },
          { toolCallId: "replace-in-file-write", messages: [] }
        );

        const writeAny = writeResult as any;
        const rawExitCode =
          typeof writeAny?.exitCode === "number" ? writeAny.exitCode : undefined;
        const explicitSuccess =
          typeof writeAny?.success === "boolean" ? writeAny.success : undefined;
        const writeSuccess =
          explicitSuccess !== false && (rawExitCode == null || rawExitCode === 0);

        if (!writeSuccess) {
          const errorMessage =
            writeAny?.error ||
            writeAny?.stderr ||
            `Failed to write updated content to ${projectRelativePath}`;
          if (logFunction) {
            logFunction(
              "error",
              `[replaceInFile] writeFile failed for ${projectRelativePath}: ${errorMessage}`
            );
          }
          return {
            success: false,
            error: errorMessage,
            filePath: projectRelativePath,
            occurrences,
            exitCode: rawExitCode,
            stdout: writeAny?.stdout,
            stderr: writeAny?.stderr,
          };
        }

        const message = `Successfully replaced ${occurrences} occurrence(s) in ${projectRelativePath}`;
        if (logFunction) {
          logFunction("success", `[replaceInFile] ${message}`);
        }

        return {
          success: true,
          filePath: projectRelativePath,
          occurrences,
          message,
          ...(warning ? { warning } : {}),
        };
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        if (logFunction) {
          logFunction("error", `[replaceInFile] Failed for path=${path}: ${msg}`);
        }
        return {
          success: false,
          error: `replaceInFile failed: ${msg}`,
        };
      }
    },
  });
}

/**
 * Creates a tool registry for agent operations.
 * Centralizes all tools used across different agent APIs.
 * When mode is "cloud", writeFile and executeCommand run in the server workspace (Docker/project dir); otherwise they use the sync client.
 */
export function createAgentTools(
  projectId: string,
  executionSessionId?: string,
  mode: "local" | "cloud" = "local",
  logFunction?: AgentLogFunction
) {
  const useCloud = mode === "cloud";
  const execTool = useCloud
    ? createCloudExecuteCommandTool(projectId, executionSessionId, logFunction)
    : createExecuteCommandTool(projectId, executionSessionId, logFunction);
  const runExecuteCommand: RunExecuteCommandFn = (command: string) =>
    execTool.execute(
      { command },
      { toolCallId: "run-execute-command", messages: [] }
    ) as unknown as Promise<{
      success?: boolean;
      exitCode?: number;
      stdout?: string;
      stderr?: string;
      error?: string;
    }>;
  const readFileTool = createReadFileTool(projectId, {
    logFunction,
    runExecuteCommand: useCloud ? runExecuteCommand : undefined,
    executionSessionId: useCloud ? executionSessionId : undefined,
  });
  const writeFileTool = useCloud
    ? createCloudWriteFileTool(projectId, executionSessionId, runExecuteCommand)
    : createWriteFileTool(projectId, executionSessionId, runExecuteCommand);
  const replaceInFileTool = createReplaceInFileTool(projectId, mode, {
    logFunction,
    runExecuteCommand: useCloud ? runExecuteCommand : undefined,
    executionSessionId: useCloud ? executionSessionId : undefined,
  });
  const baseTools = {
    searchKnowledge: createSearchKnowledgeTool(projectId, logFunction),
    searchCodebase: createSearchCodebaseTool(projectId, logFunction),
    getCodeMap: createGetCodeMapTool(projectId, logFunction),
    searchWeb,
    readFile: readFileTool,
    executeCommand: execTool,
    writeFile: writeFileTool,
    replaceInFile: replaceInFileTool,
    findRelatedFiles: createFindRelatedFilesTool(),
    submitForQA: tool({
      description:
        "Submit completed work for QA verification. Call this when implementation is ready and you want to hand off for verification.",
      parameters: z.object({
        summary: z.string().optional().describe("Brief summary of what was implemented"),
      }),
      execute: async () => ({ status: "queuedForQA" }),
    }),
  };
  const toolsWithDuration: Record<string, (typeof baseTools)[keyof typeof baseTools]> = {};
  for (const [toolName, t] of Object.entries(baseTools)) {
    const raw = t as unknown as { execute: (args: unknown, options?: unknown) => Promise<unknown> };
    const originalExecute = raw.execute;
    raw.execute = async (args: unknown, options?: unknown) => {
      const start = Date.now();
      try {
        const result = await originalExecute(args, options);
        const durationMs = Date.now() - start;
        const res = result as Record<string, unknown> | null | undefined;
        const success = res != null && res.success !== false && (res.exitCode == null || res.exitCode === 0);
        if (logFunction) {
          logFunction("info", `[Tools] ${success ? "Done" : "Failed"}: ${toolName} (${durationMs}ms)`);
        }
        return result;
      } catch (err) {
        const durationMs = Date.now() - start;
        if (logFunction) {
          logFunction("info", `[Tools] Failed: ${toolName} (${durationMs}ms) - ${err instanceof Error ? err.message : String(err)}`);
        }
        throw err;
      }
    };
    toolsWithDuration[toolName] = t;
  }
  return toolsWithDuration;
}

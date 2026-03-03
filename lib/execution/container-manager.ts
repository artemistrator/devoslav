import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);
const DOCKER_IMAGE = "node:20"; // Standard base image for JS/TS projects with build tools

function getDockerImage(projectType?: string | null): string {
  switch (projectType) {
    case "backend":
    case "python":
      return "python:3.11-slim";
    case "frontend":
    case "static":
    case "fullstack":
    case "script":
    default:
      return DOCKER_IMAGE;
  }
}
const DOCKER_EXEC_MAX_BUFFER = 10 * 1024 * 1024; // 10MB so build/TS compiler output is not truncated

const HOST_PROJECTS_DIR =
  process.env.HOST_PROJECTS_DIR || "/app/projects";
const HOST_UID = process.env.HOST_UID || "1000";
const HOST_GID = process.env.HOST_GID || "1000";

/** Container working directory (docker -w). Use this in prompts so the LLM sees the path inside the container, not the host path. */
export const CONTAINER_WORKSPACE_ROOT = "/app";

/** Cache paths under /app to persist via named volumes (Node/Next.js). Keyed by projectId.
 *  node_modules is NOT included: named volumes are created as root, but the container runs as HOST_UID:HOST_GID, causing EACCES on npm install. */
const NODE_CACHE_PATHS = [".next/cache"] as const;

function resolveHostProjectPath(internalPath: string): string {
  // Internal paths are expected like /app/projects/<projectId>[...]
  if (!internalPath.startsWith("/app/projects/")) {
    // Fallback: treat as already a host path
    return internalPath;
  }

  const relative = internalPath.replace("/app/projects/", "");
  return `${HOST_PROJECTS_DIR}/${relative}`;
}

/** Sanitize projectId for use in Docker volume names (only [a-zA-Z0-9_-]). */
function sanitizeProjectIdForVolume(projectId: string): string {
  return projectId.replace(/[^a-zA-Z0-9_-]/g, "-");
}

export function getContainerName(sessionId: string): string {
  // Safe container name
  return `ai-orch-session-${sessionId.replace(/[^a-zA-Z0-9_.-]/g, "-")}`;
}

export async function ensureContainer(
  sessionId: string,
  projectPath: string,
  projectId: string,
  projectType?: string | null
): Promise<void> {
  const containerName = getContainerName(sessionId);
  const hostProjectPath = resolveHostProjectPath(projectPath);
  const safeId = sanitizeProjectIdForVolume(projectId);
  try {
    // Check if running
    await execAsync(`docker inspect -f '{{.State.Running}}' ${containerName}`, { maxBuffer: DOCKER_EXEC_MAX_BUFFER });
    console.log(
      `[ContainerManager] Container ${containerName} already running.`
    );
    return;
  } catch (error) {
    console.log(
      `[ContainerManager] Creating container ${containerName} with host path ${hostProjectPath}...`
    );
    const image = getDockerImage(projectType);
    console.log(
      `[ContainerManager] Using image: ${image} for projectType: ${
        projectType ?? "unknown"
      }`
    );
    // Named volumes for cache paths so they persist across container restarts (same projectId). node_modules lives in the bind-mounted project dir to avoid root-owned volume EACCES.
    const volumeMounts = NODE_CACHE_PATHS.map(
      (subPath) =>
        `-v ai-orch-${safeId}-${subPath.replace("/", "-")}:${CONTAINER_WORKSPACE_ROOT}/${subPath}`
    );
    // Run a detached container that stays alive, mounting the project directory
    // Use a writable HOME and npm cache to avoid permission issues inside the container.
    const command = [
      "docker run -d",
      `--name ${containerName}`,
      `-u ${HOST_UID}:${HOST_GID}`,
      `-e HOME=/tmp`,
      `-e NPM_CONFIG_CACHE=/tmp/.npm`,
      `-e USER=node`,
      `-w ${CONTAINER_WORKSPACE_ROOT}`,
      `-v "${hostProjectPath}:${CONTAINER_WORKSPACE_ROOT}"`,
      ...volumeMounts,
      image,
      "tail -f /dev/null",
    ].join(" ");

    await execAsync(command, { maxBuffer: DOCKER_EXEC_MAX_BUFFER });
    console.log(
      `[ContainerManager] Container ${containerName} created successfully.`
    );
  }
}

const DEFAULT_COMMAND_TIMEOUT_MS = 600000; // 10 minutes

export async function executeInContainer(
  sessionId: string,
  command: string,
  timeoutMs: number = DEFAULT_COMMAND_TIMEOUT_MS
): Promise<{ stdout: string; stderr: string; exitCode: number | null; combinedLogs?: string }> {
  const containerName = getContainerName(sessionId);
  // Safely escape single quotes for the sh -c command
  const escapedCommand = command.replace(/'/g, "'\\''");
  const dockerCommand = `docker exec ${containerName} sh -c 'export HOME=/tmp && export NPM_CONFIG_CACHE=/tmp/.npm && mkdir -p /tmp/.npm && ${escapedCommand}'`;

  const timeoutMinutes = Math.round(timeoutMs / 60000);
  const timeoutMessage = `Execution timed out after ${timeoutMinutes} minute${timeoutMinutes === 1 ? "" : "s"}.`;

  try {
    const { stdout, stderr } = await new Promise<{ stdout: string; stderr: string }>(
      (resolve, reject) => {
        const ac = new AbortController();
        const timeoutId = setTimeout(() => ac.abort(), timeoutMs);
        exec(
          dockerCommand,
          // Use a generous buffer so large build/test logs are fully captured.
          { signal: ac.signal, maxBuffer: 10 * 1024 * 1024 },
          (error, stdout, stderr) => {
            clearTimeout(timeoutId);
            if (error) {
              reject(error);
              return;
            }
            resolve({ stdout: stdout ?? "", stderr: stderr ?? "" });
          }
        );
      }
    );
    const combinedLogs = [stdout ?? "", stderr ?? ""]
      .filter((s) => s && s.length > 0)
      .join("\n");
    return { stdout, stderr, exitCode: 0, combinedLogs };
  } catch (error: any) {
    if (error?.name === "AbortError" || error?.code === "ABORT_ERR") {
      throw new Error(timeoutMessage);
    }
    const stdout =
      typeof error?.stdout === "string" ? error.stdout : "";
    const stderr =
      typeof error?.stderr === "string"
        ? error.stderr
        : typeof error?.message === "string"
        ? error.message
        : String(error);
    const exitCode =
      typeof error?.code === "number" ? error.code : 1;
    const combinedLogsRaw = [stdout, stderr]
      .filter((s) => s && s.length > 0)
      .join("\n");
    const combinedLogs =
      combinedLogsRaw || stderr || "Command failed with no output.";

    return {
      stdout,
      stderr,
      exitCode,
      combinedLogs,
    };
  }
}

export async function destroyContainer(sessionId: string): Promise<void> {
  const containerName = getContainerName(sessionId);
  try {
    console.log(`[ContainerManager] Destroying container ${containerName}...`);
    await execAsync(`docker rm -f ${containerName}`, { maxBuffer: DOCKER_EXEC_MAX_BUFFER });
    console.log(`[ContainerManager] Container ${containerName} destroyed.`);
  } catch (error) {
    console.log(
      `[ContainerManager] Container ${containerName} not found or already destroyed.`
    );
  }
}

const ORPHAN_CONTAINER_PREFIX = "ai-orch-session-";

/**
 * Finds all containers whose names start with ai-orch-session- and forcefully
 * removes them. Intended to be called once on application startup.
 */
export async function cleanupOrphanedContainers(): Promise<void> {
  try {
    const { stdout } = await execAsync(
      "docker ps -a --format '{{.Names}}'",
      { maxBuffer: DOCKER_EXEC_MAX_BUFFER }
    );
    const names = (stdout ?? "")
      .split("\n")
      .map((s) => s.trim())
      .filter((s) => s.startsWith(ORPHAN_CONTAINER_PREFIX));

    if (names.length === 0) {
      console.log("[ContainerManager] No orphaned session containers found.");
      return;
    }

    console.log(
      `[ContainerManager] Cleaning up ${names.length} orphaned container(s): ${names.join(", ")}`
    );

    const results = await Promise.allSettled(
      names.map((name) => execAsync(`docker rm -f ${name}`, { maxBuffer: DOCKER_EXEC_MAX_BUFFER }))
    );

    for (let i = 0; i < names.length; i++) {
      const name = names[i];
      const result = results[i];
      if (result.status === "fulfilled") {
        console.log(`[ContainerManager] Removed orphan container: ${name}`);
      } else {
        console.warn(
          `[ContainerManager] Failed to remove orphan container ${name}:`,
          (result as PromiseRejectedResult).reason
        );
      }
    }
  } catch (error) {
    console.error(
      "[ContainerManager] cleanupOrphanedContainers failed (non-fatal):",
      error
    );
    // Do not rethrow so startup can continue
  }
}


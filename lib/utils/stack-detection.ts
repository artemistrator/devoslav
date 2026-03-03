import { promises as fs } from "fs";
import { join } from "path";

export interface StackDetectionResult {
  type: string;
  buildCommand: string | null;
}

/**
 * Detects project stack from manifest files in projectDir and returns
 * the appropriate build command for Hard Gate (or null if none).
 */
export async function detectStack(projectDir: string): Promise<StackDetectionResult> {
  let files: string[];
  try {
    files = await fs.readdir(projectDir);
  } catch {
    return { type: "unknown", buildCommand: null };
  }

  if (files.includes("package.json")) {
    try {
      const content = await fs.readFile(join(projectDir, "package.json"), "utf-8");
      const pkg = JSON.parse(content) as { scripts?: Record<string, string> };
      const scripts = pkg?.scripts ?? {};
      const hasTypeCheck = typeof scripts["type-check"] === "string";
      const hasBuild = typeof scripts["build"] === "string";
      const hasTsconfig = files.includes("tsconfig.json");

      let buildCommand: string | null = null;
      if (hasTypeCheck) {
        buildCommand = "$(which npm 2>/dev/null || echo npm) run type-check";
      } else if (hasBuild) {
        buildCommand = "$(which npm 2>/dev/null || echo npm) run build";
      } else if (hasTsconfig) {
        // Fallback for TS projects without explicit build/type-check scripts
        buildCommand = "$(which npx 2>/dev/null || echo npx) tsc --noEmit";
      }

      return { type: "nodejs", buildCommand };
    } catch {
      return { type: "nodejs", buildCommand: null };
    }
  }

  if (files.includes("Cargo.toml")) {
    return { type: "rust", buildCommand: "$(which cargo 2>/dev/null || echo cargo) build" };
  }
  if (files.includes("go.mod")) {
    return { type: "go", buildCommand: "$(which go 2>/dev/null || echo go) build ./..." };
  }
  const hasRequirements = files.includes("requirements.txt");
  const hasPyproject = files.includes("pyproject.toml");
  const hasMainPy = files.includes("main.py");

  if (hasRequirements || hasPyproject || hasMainPy) {
    return {
      type: "python",
      buildCommand:
        "$(which python3 2>/dev/null || which python 2>/dev/null || echo python3) -m compileall -q .",
    };
  }
  if (files.includes("pom.xml")) {
    return {
      type: "java",
      buildCommand:
        "$(which mvn 2>/dev/null || which ./mvnw 2>/dev/null || echo mvn) compile",
    };
  }

  // Static HTML project: no known manifests but index.html present at root
  const hasAnyManifest =
    files.includes("package.json") ||
    files.includes("Cargo.toml") ||
    files.includes("go.mod") ||
    hasRequirements ||
    hasPyproject ||
    files.includes("pom.xml");

  if (!hasAnyManifest && files.includes("index.html")) {
    return {
      type: "static",
      buildCommand: "$(which npx 2>/dev/null || echo npx) htmlhint *.html",
    };
  }

  return { type: "unknown", buildCommand: null };
}

/**
 * Returns true if the project is a web frontend (has React, Vue, Svelte, Next, Nuxt, Astro, or Solid in deps).
 * Used to run CSS Agent only for web projects.
 */
export async function isWebProject(projectDir: string): Promise<boolean> {
  const stack = await detectStack(projectDir);
  if (stack.type !== "nodejs") return false;
  try {
    const pkg = JSON.parse(
      await fs.readFile(join(projectDir, "package.json"), "utf-8")
    ) as { dependencies?: Record<string, string>; devDependencies?: Record<string, string> };
    const deps = { ...pkg.dependencies, ...pkg.devDependencies };
    const webFrameworks = ["react", "vue", "svelte", "next", "nuxt", "astro", "solid"];
    return webFrameworks.some((fw) =>
      Object.keys(deps ?? {}).some((d) => d.includes(fw))
    );
  } catch {
    return false;
  }
}

/**
 * Shared helpers for artifact verification path resolution.
 * Ensures we never use build/output dirs (.next, node_modules, dist, target, etc.) as project root or as artifact paths.
 */

/** Build/output dir segments: never use as artifactBase. Covers Node, Next.js, Python, Go, Rust. */
export const BUILD_DIR_SEGMENTS = [
  "/node_modules/",
  "/.next/",
  "/dist/",
  "/.git/",
  "/target/",       // Rust
  "/__pycache__/",  // Python
  "/vendor/",       // Go
];
const MANIFEST_NAMES = ["package.json", "Cargo.toml", "go.mod", "pyproject.toml", "pom.xml", "build.gradle"];

/**
 * Command to run from workspace root: exits 0 if any project manifest exists in current directory.
 * Use this first; when it succeeds, use artifactBase = "." so verification paths are root-relative.
 */
export const MANIFEST_AT_ROOT_CHECK_CMD =
  "test -f package.json || test -f Cargo.toml || test -f go.mod || test -f pyproject.toml || test -f pom.xml || test -f build.gradle";

/** Find command that excludes build and VCS dirs; returns first manifest path. */
export const FIND_MANIFEST_CMD =
  `find . -maxdepth 3 -not -path "*/node_modules/*" -not -path "*/.next/*" -not -path "*/dist/*" -not -path "*/.git/*" -not -path "*/target/*" -not -path "*/__pycache__/*" -not -path "*/vendor/*" \\( -name "package.json" -o -name "Cargo.toml" -o -name "go.mod" -o -name "pyproject.toml" -o -name "pom.xml" -o -name "build.gradle" \\) 2>/dev/null | head -1`;

export interface FindResultLike {
  success?: boolean;
  stdout?: string;
}

/**
 * Resolves artifactBase from the result of running FIND_MANIFEST_CMD.
 * Only uses stdout when the command succeeded and the first line is a valid manifest path outside build dirs.
 */
export function resolveArtifactBaseFromFindResult(result: FindResultLike | null | undefined): string {
  if (result?.success !== true) return ".";
  const out = typeof result.stdout === "string" ? result.stdout : "";
  const firstLine = out.split("\n")[0]?.trim() ?? "";
  if (!firstLine || !firstLine.startsWith(".")) return ".";

  const looksLikeManifest = MANIFEST_NAMES.some((name) => firstLine.endsWith(name));
  const outsideBuildDirs = !BUILD_DIR_SEGMENTS.some((seg) => firstLine.includes(seg));
  if (!looksLikeManifest || !outsideBuildDirs) return ".";

  const dir = firstLine.includes("/") ? firstLine.replace(/\/[^/]+$/, "") : ".";
  const base = dir || ".";
  if (base !== "." && !base.startsWith("./")) return "./" + base;
  return base;
}

/**
 * Returns true when every artifact looks like a root-relative path (contains "/").
 * When true, caller should use artifactBase = "." and skip find for artifact path resolution.
 */
export function allArtifactsArePathLike(artifacts: string[]): boolean {
  if (!Array.isArray(artifacts) || artifacts.length === 0) return false;
  return artifacts.every((a) => typeof a === "string" && a.trim().includes("/"));
}

/** Dir names that indicate build/output; artifactBase must not be under these. */
const BUILD_DIR_NAMES = [".next", "node_modules", "dist", ".git", "target", "__pycache__", "vendor"];

/**
 * Returns true if the given artifactBase is under a build/output dir and must not be used (fallback to ".").
 */
export function isArtifactBaseInBuildDir(artifactBase: string): boolean {
  if (!artifactBase || artifactBase === ".") return false;
  // Strip only "./" prefix so ".next/types" stays ".next/types" for matching
  const normalized = artifactBase.replace(/^\.\//, "").replace(/\/+$/, "");
  return BUILD_DIR_NAMES.some(
    (name) => normalized === name || normalized.startsWith(name + "/") || normalized.endsWith("/" + name)
  );
}

/**
 * Returns the path to use for ls -la / head -n.
 * If the resolved path would land under a build dir, uses the artifact as path (workspace root) so we verify source files.
 */
export function getVerificationPath(artifactBase: string, artifact: string): string {
  const path = artifactBase === "." ? artifact : `${artifactBase}/${artifact}`;
  // Normalize with leading slash so ".next/types/src/..." matches "/.next/"
  const pathForCheck = path.startsWith("/") ? path : "/" + path;
  const isBuildPath = BUILD_DIR_SEGMENTS.some((seg) => pathForCheck.includes(seg));
  if (isBuildPath) return artifact;
  return path;
}

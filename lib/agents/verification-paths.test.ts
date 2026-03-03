import { describe, it, expect } from "vitest";
import {
  resolveArtifactBaseFromFindResult,
  allArtifactsArePathLike,
  isArtifactBaseInBuildDir,
  getVerificationPath,
  FIND_MANIFEST_CMD,
  MANIFEST_AT_ROOT_CHECK_CMD,
  BUILD_DIR_SEGMENTS,
} from "./verification-paths";

describe("verification-paths", () => {
  describe("resolveArtifactBaseFromFindResult", () => {
    it("returns . when find returns ./.next/types/package.json (build dir)", () => {
      const result = resolveArtifactBaseFromFindResult({
        success: true,
        stdout: "./.next/types/package.json",
      });
      expect(result).toBe(".");
    });

    it("returns . when find returns ./node_modules/foo/package.json", () => {
      const result = resolveArtifactBaseFromFindResult({
        success: true,
        stdout: "./node_modules/foo/package.json",
      });
      expect(result).toBe(".");
    });

    it("returns . when success is false", () => {
      expect(resolveArtifactBaseFromFindResult({ success: false, stdout: "./package.json" })).toBe(".");
      expect(resolveArtifactBaseFromFindResult(null)).toBe(".");
      expect(resolveArtifactBaseFromFindResult(undefined)).toBe(".");
    });

    it("returns . when stdout is empty or invalid", () => {
      expect(resolveArtifactBaseFromFindResult({ success: true, stdout: "" })).toBe(".");
      expect(resolveArtifactBaseFromFindResult({ success: true, stdout: "package.json" })).toBe(".");
    });

    it("returns . for root package.json", () => {
      const result = resolveArtifactBaseFromFindResult({
        success: true,
        stdout: "./package.json",
      });
      expect(result).toBe(".");
    });

    it("returns subdir for packages/app/package.json", () => {
      const result = resolveArtifactBaseFromFindResult({
        success: true,
        stdout: "./packages/app/package.json",
      });
      expect(result).toBe("./packages/app");
    });
  });

  describe("allArtifactsArePathLike", () => {
    it("returns true when all artifacts contain /", () => {
      expect(allArtifactsArePathLike(["src/lib/db.ts", "src/store/useStore.ts"])).toBe(true);
      expect(allArtifactsArePathLike(["lib/foo.ts"])).toBe(true);
    });

    it("returns false when at least one artifact has no /", () => {
      expect(allArtifactsArePathLike(["store.ts"])).toBe(false);
      expect(allArtifactsArePathLike(["src/lib/db.ts", "store.ts"])).toBe(false);
    });

    it("returns false for empty or non-array", () => {
      expect(allArtifactsArePathLike([])).toBe(false);
      expect(allArtifactsArePathLike(null as any)).toBe(false);
      expect(allArtifactsArePathLike(undefined as any)).toBe(false);
    });
  });

  describe("isArtifactBaseInBuildDir", () => {
    it("returns true for .next/types and similar build dirs", () => {
      expect(isArtifactBaseInBuildDir(".next/types")).toBe(true);
      expect(isArtifactBaseInBuildDir("./.next/types")).toBe(true);
      expect(isArtifactBaseInBuildDir("node_modules")).toBe(true);
      expect(isArtifactBaseInBuildDir("dist")).toBe(true);
      expect(isArtifactBaseInBuildDir("target")).toBe(true);
      expect(isArtifactBaseInBuildDir("__pycache__")).toBe(true);
      expect(isArtifactBaseInBuildDir("vendor")).toBe(true);
    });

    it("returns false for . and empty", () => {
      expect(isArtifactBaseInBuildDir(".")).toBe(false);
      expect(isArtifactBaseInBuildDir("")).toBe(false);
    });

    it("returns false for source-like paths", () => {
      expect(isArtifactBaseInBuildDir("packages/app")).toBe(false);
      expect(isArtifactBaseInBuildDir("src")).toBe(false);
    });
  });

  describe("getVerificationPath", () => {
    it("returns artifact as-is when artifactBase is .", () => {
      expect(getVerificationPath(".", "src/lib/db.ts")).toBe("src/lib/db.ts");
      expect(getVerificationPath(".", "src/store/useStore.ts")).toBe("src/store/useStore.ts");
    });

    it("returns artifact only when combined path would be under build dir", () => {
      expect(getVerificationPath(".next/types", "src/lib/db.ts")).toBe("src/lib/db.ts");
    });

    it("returns base/artifact when base is valid source dir", () => {
      expect(getVerificationPath("packages/app", "src/index.ts")).toBe("packages/app/src/index.ts");
    });
  });

  describe("MANIFEST_AT_ROOT_CHECK_CMD", () => {
    it("checks for package.json and other manifests in current dir", () => {
      expect(MANIFEST_AT_ROOT_CHECK_CMD).toContain("test -f package.json");
      expect(MANIFEST_AT_ROOT_CHECK_CMD).toContain("Cargo.toml");
      expect(MANIFEST_AT_ROOT_CHECK_CMD).toContain("go.mod");
      expect(MANIFEST_AT_ROOT_CHECK_CMD).toContain("pyproject.toml");
    });
  });

  describe("FIND_MANIFEST_CMD", () => {
    it("excludes .next, node_modules, dist, target, __pycache__, vendor", () => {
      expect(FIND_MANIFEST_CMD).toContain("-not -path \"*/.next/*\"");
      expect(FIND_MANIFEST_CMD).toContain("-not -path \"*/node_modules/*\"");
      expect(FIND_MANIFEST_CMD).toContain("-not -path \"*/dist/*\"");
      expect(FIND_MANIFEST_CMD).toContain("-not -path \"*/target/*\"");
      expect(FIND_MANIFEST_CMD).toContain("-not -path \"*/__pycache__/*\"");
      expect(FIND_MANIFEST_CMD).toContain("-not -path \"*/vendor/*\"");
    });
  });

  describe("BUILD_DIR_SEGMENTS", () => {
    it("includes expected build dirs for Node, Next, Python, Go, Rust", () => {
      expect(BUILD_DIR_SEGMENTS).toContain("/.next/");
      expect(BUILD_DIR_SEGMENTS).toContain("/node_modules/");
      expect(BUILD_DIR_SEGMENTS).toContain("/dist/");
      expect(BUILD_DIR_SEGMENTS).toContain("/target/");
      expect(BUILD_DIR_SEGMENTS).toContain("/__pycache__/");
      expect(BUILD_DIR_SEGMENTS).toContain("/vendor/");
    });
  });
});

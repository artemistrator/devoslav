import fs from "fs/promises";
import path from "path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadProjectVibe } from "./parser";

vi.mock("@/lib/project-workspace", () => ({
  getProjectDir: (projectId: string) =>
    path.join(process.cwd(), "__test-fixtures__", "vibe", projectId),
}));

describe("loadProjectVibe", () => {
  const fixturesRoot = path.join(process.cwd(), "__test-fixtures__", "vibe");

  beforeEach(async () => {
    await fs.mkdir(fixturesRoot, { recursive: true });
  });

  afterEach(async () => {
    try {
      await fs.rm(fixturesRoot, { recursive: true, force: true });
    } catch {
      // ignore
    }
  });

  it("returns null when no vibe files exist", async () => {
    const result = await loadProjectVibe("no-vibe");
    expect(result).toBeNull();
  });

  it("loads configuration from vibe.yaml when present", async () => {
    const projectDir = path.join(fixturesRoot, "yaml-project");
    await fs.mkdir(projectDir, { recursive: true });
    const yamlContent = `
version: "1.0"
architecture:
  preferred_pattern: "hexagonal"
  forbidden_patterns:
    - "god-object"
code_style:
  error_handling: "never-throw-any"
`;
    await fs.writeFile(path.join(projectDir, "vibe.yaml"), yamlContent, "utf8");

    const result = await loadProjectVibe("yaml-project");

    expect(result).not.toBeNull();
    expect(result?.version).toBe("1.0");
    expect(result?.architecture?.preferred_pattern).toBe("hexagonal");
    expect(result?.architecture?.forbidden_patterns).toContain("god-object");
    expect(result?.code_style?.error_handling).toBe("never-throw-any");
  });

  it("falls back to vibe.json when YAML is missing", async () => {
    const projectDir = path.join(fixturesRoot, "json-project");
    await fs.mkdir(projectDir, { recursive: true });
    const jsonContent = {
      architecture: {
        preferred_pattern: "clean-architecture",
      },
      qa_rules: {
        mandatory_evidence: ["tests", "screenshots"],
      },
    };
    await fs.writeFile(
      path.join(projectDir, "vibe.json"),
      JSON.stringify(jsonContent),
      "utf8"
    );

    const result = await loadProjectVibe("json-project");

    expect(result).not.toBeNull();
    expect(result?.architecture?.preferred_pattern).toBe("clean-architecture");
    expect(result?.qa_rules?.mandatory_evidence).toContain("tests");
  });

  it("handles unexpected YAML content without throwing", async () => {
    const projectDir = path.join(fixturesRoot, "invalid-project");
    await fs.mkdir(projectDir, { recursive: true });
    await fs.writeFile(
      path.join(projectDir, "vibe.yaml"),
      ":::this is not valid yaml:::",
      "utf8"
    );

    const result = await loadProjectVibe("invalid-project");

    expect(result).not.toBeNull();
  });
});


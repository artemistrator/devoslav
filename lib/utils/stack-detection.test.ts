import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { detectStack, isWebProject } from "./stack-detection";

describe("detectStack", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "stack-detection-"));
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("returns nodejs with npm run build when package.json has scripts.build", async () => {
    writeFileSync(
      join(tmpDir, "package.json"),
      JSON.stringify({ scripts: { build: "vite build" } })
    );
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("nodejs");
    expect(result.buildCommand).toBe("$(which npm 2>/dev/null || echo npm) run build");
  });

  it("returns nodejs with npm run type-check when package.json has scripts.type-check", async () => {
    writeFileSync(
      join(tmpDir, "package.json"),
      JSON.stringify({ scripts: { "type-check": "tsc --noEmit" } })
    );
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("nodejs");
    expect(result.buildCommand).toBe("$(which npm 2>/dev/null || echo npm) run type-check");
  });

  it("prefers type-check over build for nodejs", async () => {
    writeFileSync(
      join(tmpDir, "package.json"),
      JSON.stringify({
        scripts: { build: "vite build", "type-check": "tsc --noEmit" },
      })
    );
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("nodejs");
    expect(result.buildCommand).toBe("$(which npm 2>/dev/null || echo npm) run type-check");
  });

  it("returns nodejs with tsc --noEmit when package.json has no build or type-check but tsconfig.json exists", async () => {
    writeFileSync(join(tmpDir, "package.json"), JSON.stringify({ name: "app" }));
    writeFileSync(join(tmpDir, "tsconfig.json"), JSON.stringify({ compilerOptions: {} }));
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("nodejs");
    expect(result.buildCommand).toBe("$(which npx 2>/dev/null || echo npx) tsc --noEmit");
  });

  it("returns rust with cargo build for Cargo.toml", async () => {
    writeFileSync(join(tmpDir, "Cargo.toml"), '[package]\nname = "foo"');
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("rust");
    expect(result.buildCommand).toBe("$(which cargo 2>/dev/null || echo cargo) build");
  });

  it("returns go with go build ./... for go.mod", async () => {
    writeFileSync(join(tmpDir, "go.mod"), "module example.com/foo");
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("go");
    expect(result.buildCommand).toBe("$(which go 2>/dev/null || echo go) build ./...");
  });

  it("returns python with compileall for requirements.txt", async () => {
    writeFileSync(join(tmpDir, "requirements.txt"), "requests\n");
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("python");
    expect(result.buildCommand).toBe(
      "$(which python3 2>/dev/null || which python 2>/dev/null || echo python3) -m compileall -q ."
    );
  });

  it("returns python with compileall for pyproject.toml", async () => {
    writeFileSync(join(tmpDir, "pyproject.toml"), "[project]\nname = \"foo\"");
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("python");
    expect(result.buildCommand).toBe(
      "$(which python3 2>/dev/null || which python 2>/dev/null || echo python3) -m compileall -q ."
    );
  });

  it("returns python with compileall for main.py", async () => {
    writeFileSync(join(tmpDir, "main.py"), "print('hello')");
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("python");
    expect(result.buildCommand).toBe(
      "$(which python3 2>/dev/null || which python 2>/dev/null || echo python3) -m compileall -q ."
    );
  });

  it("returns java with mvn compile for pom.xml", async () => {
    writeFileSync(
      join(tmpDir, "pom.xml"),
      '<?xml version="1.0"?><project></project>'
    );
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("java");
    expect(result.buildCommand).toBe(
      "$(which mvn 2>/dev/null || which ./mvnw 2>/dev/null || echo mvn) compile"
    );
  });

  it("returns static with htmlhint for index.html when no manifests exist", async () => {
    writeFileSync(join(tmpDir, "index.html"), "<!DOCTYPE html><html><body>Hello</body></html>");
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("static");
    expect(result.buildCommand).toBe("$(which npx 2>/dev/null || echo npx) htmlhint *.html");
  });

  it("returns unknown when directory is empty", async () => {
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("unknown");
    expect(result.buildCommand).toBeNull();
  });

  it("prioritizes package.json over Cargo.toml when both exist", async () => {
    writeFileSync(
      join(tmpDir, "package.json"),
      JSON.stringify({ scripts: { build: "vite build" } })
    );
    writeFileSync(join(tmpDir, "Cargo.toml"), '[package]\nname = "foo"');
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("nodejs");
    expect(result.buildCommand).toBe("$(which npm 2>/dev/null || echo npm) run build");
  });

  it("returns unknown for non-existent directory", async () => {
    const result = await detectStack(join(tmpDir, "nonexistent"));
    expect(result.type).toBe("unknown");
    expect(result.buildCommand).toBeNull();
  });

  it("returns nodejs with buildCommand null when package.json is invalid JSON", async () => {
    writeFileSync(join(tmpDir, "package.json"), "not json");
    const result = await detectStack(tmpDir);
    expect(result.type).toBe("nodejs");
    expect(result.buildCommand).toBeNull();
  });
});

describe("isWebProject", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "isWebProject-"));
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("returns true when dependencies include react", async () => {
    writeFileSync(
      join(tmpDir, "package.json"),
      JSON.stringify({ dependencies: { react: "^18.0.0" } })
    );
    expect(await isWebProject(tmpDir)).toBe(true);
  });

  it("returns true when devDependencies include vue", async () => {
    writeFileSync(
      join(tmpDir, "package.json"),
      JSON.stringify({ devDependencies: { vue: "^3.0.0" } })
    );
    expect(await isWebProject(tmpDir)).toBe(true);
  });

  it("returns true when dependencies include next", async () => {
    writeFileSync(
      join(tmpDir, "package.json"),
      JSON.stringify({ dependencies: { next: "14.0.0" } })
    );
    expect(await isWebProject(tmpDir)).toBe(true);
  });

  it("returns true when dependencies include svelte", async () => {
    writeFileSync(
      join(tmpDir, "package.json"),
      JSON.stringify({ dependencies: { svelte: "^4.0.0" } })
    );
    expect(await isWebProject(tmpDir)).toBe(true);
  });

  it("returns false when nodejs has no web framework (express/lodash only)", async () => {
    writeFileSync(
      join(tmpDir, "package.json"),
      JSON.stringify({ dependencies: { express: "^4.0.0", lodash: "^4.0.0" } })
    );
    expect(await isWebProject(tmpDir)).toBe(false);
  });

  it("returns false for rust project (Cargo.toml)", async () => {
    writeFileSync(join(tmpDir, "Cargo.toml"), '[package]\nname = "foo"');
    expect(await isWebProject(tmpDir)).toBe(false);
  });

  it("returns false when directory is empty (no package.json)", async () => {
    expect(await isWebProject(tmpDir)).toBe(false);
  });

  it("returns false when package.json is invalid JSON", async () => {
    writeFileSync(join(tmpDir, "package.json"), "not json");
    expect(await isWebProject(tmpDir)).toBe(false);
  });
});

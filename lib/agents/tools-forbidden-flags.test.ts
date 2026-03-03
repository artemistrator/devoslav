import { describe, it, expect } from "vitest";
import { hasForbiddenSilentFlag } from "./tools";

describe("hasForbiddenSilentFlag", () => {
  it("returns true for --silent", () => {
    expect(hasForbiddenSilentFlag("npm install --silent")).toBe(true);
    expect(hasForbiddenSilentFlag("npm install --silent package")).toBe(true);
    expect(hasForbiddenSilentFlag("npm install -D canvas --silent")).toBe(true);
  });

  it("returns true for --quiet", () => {
    expect(hasForbiddenSilentFlag("npm install --quiet")).toBe(true);
    expect(hasForbiddenSilentFlag("npm run build --quiet")).toBe(true);
  });

  it("returns true for -s as standalone flag", () => {
    expect(hasForbiddenSilentFlag("npm install -s")).toBe(true);
    expect(hasForbiddenSilentFlag("npm install -s package")).toBe(true);
    expect(hasForbiddenSilentFlag("npm install -s ")).toBe(true);
    expect(hasForbiddenSilentFlag("npm run build -s")).toBe(true);
  });

  it("returns true for -s at end of command", () => {
    expect(hasForbiddenSilentFlag("npm install -s")).toBe(true);
  });

  it("returns false for commands without forbidden flags", () => {
    expect(hasForbiddenSilentFlag("npm install")).toBe(false);
    expect(hasForbiddenSilentFlag("npm run build")).toBe(false);
    expect(hasForbiddenSilentFlag("npm test")).toBe(false);
  });

  it("does not treat --save as -s flag", () => {
    expect(hasForbiddenSilentFlag("npm install --save package")).toBe(false);
  });

  it("returns false for empty or whitespace", () => {
    expect(hasForbiddenSilentFlag("")).toBe(false);
    expect(hasForbiddenSilentFlag("   ")).toBe(false);
  });
});

import { describe, it, expect } from "vitest";
import { checkOpenAIConnection, checkZaiConnection } from "./provider-connectivity";

const OPENAI_READY = !!process.env.OPENAI_API_KEY?.trim();
const ZAI_READY = !!process.env.ZAI_API_KEY?.trim();

describe("Provider connectivity (integration)", () => {
  it.skipIf(!OPENAI_READY)(
    "OpenAI: connection succeeds when OPENAI_API_KEY is set",
    async () => {
      const result = await checkOpenAIConnection();
      expect(result.ok, !result.ok ? (result as { message: string }).message : undefined).toBe(true);
    },
    25000
  );

  it("OpenAI: when key missing, returns ok: false with message", async () => {
    const orig = process.env.OPENAI_API_KEY;
    process.env.OPENAI_API_KEY = "";
    try {
      const result = await checkOpenAIConnection();
      expect(result.ok).toBe(false);
      expect("message" in result && result.message).toContain("OPENAI_API_KEY");
    } finally {
      process.env.OPENAI_API_KEY = orig;
    }
  });

  it.skipIf(!ZAI_READY)(
    "Z.ai: connection succeeds when ZAI_API_KEY is set",
    async () => {
      const result = await checkZaiConnection();
      expect(result.ok, !result.ok ? (result as { message: string }).message : undefined).toBe(true);
    },
    25000
  );

  it("Z.ai: when key missing, returns ok: false with message", async () => {
    const orig = process.env.ZAI_API_KEY;
    process.env.ZAI_API_KEY = "";
    try {
      const result = await checkZaiConnection();
      expect(result.ok).toBe(false);
      expect("message" in result && result.message).toContain("ZAI_API_KEY");
    } finally {
      process.env.ZAI_API_KEY = orig;
    }
  });
});

/**
 * Minimal connectivity checks for AI providers.
 * Used by /api/debug/provider-connectivity and integration tests.
 */

import { generateText } from "ai";
import { getModel, getProviderApiKey } from "@/lib/ai/providers";
import { generateTextZai } from "@/lib/ai/zai";

export type ConnectivityResult = { ok: true } | { ok: false; message: string };

const PING_PROMPT = "Reply with exactly: OK";
const MAX_TOKENS = 10;
const TIMEOUT_MS = 20000;

/** Check OpenAI (or OpenRouter if OPENAI_BASE_URL points there). Uses same code path as generate-tasks. */
export async function checkOpenAIConnection(): Promise<ConnectivityResult> {
  const apiKey = getProviderApiKey("openai");
  if (!apiKey?.trim()) {
    return { ok: false, message: "OPENAI_API_KEY not set" };
  }
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const model = getModel("openai", process.env.OPENAI_DEFAULT_MODEL || "gpt-4o-mini");
    await generateText({
      model,
      prompt: PING_PROMPT,
      maxTokens: MAX_TOKENS,
      abortSignal: controller.signal,
    });
    clearTimeout(t);
    return { ok: true };
  } catch (err) {
    clearTimeout(t);
    const msg = err instanceof Error ? err.message : String(err);
    return { ok: false, message: msg };
  }
}

/** Check Z.ai. Uses same code path as generate-tasks (generateTextZai). */
export async function checkZaiConnection(): Promise<ConnectivityResult> {
  const apiKey = process.env.ZAI_API_KEY?.trim();
  if (!apiKey) {
    return { ok: false, message: "ZAI_API_KEY not set" };
  }
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    await generateTextZai({
      systemPrompt: "Reply with exactly: OK",
      userMessage: PING_PROMPT,
      model: process.env.ZAI_DEFAULT_MODEL || "glm-4.7",
      temperature: 0,
      maxTokens: MAX_TOKENS,
      signal: controller.signal,
    });
    clearTimeout(t);
    return { ok: true };
  } catch (err) {
    clearTimeout(t);
    const msg = err instanceof Error ? err.message : String(err);
    return { ok: false, message: msg };
  }
}

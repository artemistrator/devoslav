export const LLM_TIMEOUT_MS = 90_000;

// Default timeout for regular/lightweight LLM calls (QA, small helpers, etc.)
export const DEFAULT_LLM_TIMEOUT_MS = LLM_TIMEOUT_MS;

// Extended timeout for heavy operations (plan generation, prompt regeneration, long ReAct loops).
export const HEAVY_LLM_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Create an AbortSignal for LLM calls with a hard timeout.
 *
 * Uses AbortSignal.timeout when available (Node 18.17+ / modern runtimes),
 * and falls back to a manual AbortController-based timeout otherwise.
 */
export function createLLMAbortSignal(timeoutMs: number = DEFAULT_LLM_TIMEOUT_MS): AbortSignal {
  const anyAbortSignal = AbortSignal as any;

  if (typeof anyAbortSignal?.timeout === "function") {
    return anyAbortSignal.timeout(timeoutMs) as AbortSignal;
  }

  const controller = new AbortController();
  setTimeout(() => controller.abort(), timeoutMs);
  return controller.signal;
}


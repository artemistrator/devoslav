export type GenerateTextZaiOptions = {
  systemPrompt: string;
  userMessage: string;
  model: string;
  temperature?: number;
  maxTokens?: number;
  /** Optional abort signal (e.g. for timeout in connectivity checks). */
  signal?: AbortSignal;
};

/**
 * Call Z.ai chat completions with thinking disabled so the reply is in content (not only reasoning_content).
 * Uses ZAI_API_KEY and ZAI_BASE_URL from env.
 */
export async function generateTextZai(options: GenerateTextZaiOptions): Promise<string> {
  const {
    systemPrompt,
    userMessage,
    model,
    temperature = 0.2,
    maxTokens = 16384,
    signal,
  } = options;

  const apiKey = process.env.ZAI_API_KEY;
  const baseURL = (
    process.env.ZAI_BASE_URL || "https://api.z.ai/api/coding/paas/v4"
  ).replace(/\/$/, "");

  if (!apiKey) {
    throw new Error("Missing ZAI_API_KEY");
  }

  const url = `${baseURL}/chat/completions`;
  const safeMaxTokens = Math.min(maxTokens, 16384);
  const res = await fetch(url, {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage },
      ],
      temperature,
      max_tokens: safeMaxTokens,
      thinking: { type: "disabled" },
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText || `Z.ai API ${res.status}`);
  }

  const data = (await res.json()) as {
    choices?: Array<{
      message?: { content?: string | null; reasoning_content?: string | null };
    }>;
  };

  const choice = data.choices?.[0]?.message;
  const content = choice?.content?.trim();
  const reasoning = choice?.reasoning_content?.trim();
  return (content || reasoning || "").trim();
}

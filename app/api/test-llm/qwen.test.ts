import { describe, it, expect } from "vitest";
import { POST } from "./route";

describe("Qwen test-llm ping→pong", () => {
  const hasQwenKey = !!process.env.QWEN_API_KEY;

  (hasQwenKey ? it : it.skip)(
    "returns pong for ping with Qwen 3.5 Plus",
    async () => {
      const body = {
        message: "ping",
        provider: "qwen",
        model: "qwen/qwen3.5-plus",
      };

      const request = new Request("http://localhost/api/test-llm", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      const response = await POST(request);
      const json = (await response.json()) as { reply?: string; error?: string };

      expect(json.error).toBeUndefined();
      expect(typeof json.reply).toBe("string");
      expect(json.reply?.trim().toLowerCase()).toBe("pong");
    },
    60_000,
  );
});


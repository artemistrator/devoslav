import { NextResponse } from "next/server";
import { checkOpenAIConnection, checkZaiConnection } from "@/lib/ai/provider-connectivity";

export const maxDuration = 45;

/**
 * GET /api/debug/provider-connectivity
 * Checks connectivity to OpenAI and Z.ai (when keys are set).
 * Use to verify "fetch failed" / 500 causes: network, key, or provider down.
 */
export async function GET() {
  const [openai, zai] = await Promise.all([
    checkOpenAIConnection(),
    checkZaiConnection(),
  ]);
  return NextResponse.json({
    openai: openai.ok ? { ok: true } : { ok: false, message: openai.message },
    zai: zai.ok ? { ok: true } : { ok: false, message: zai.message },
  });
}

const MAX_STRING_PREVIEW_CHARS = 4000;
// For \"vision\"-style tools (readFile, getCodeMap, cat-like commands), we
// deliberately allow up to ~100KB of content/stdout to pass through so agents
// can see full files and build logs, while still capping everything else more
// aggressively.
const MAX_VISION_FIELD_CHARS = 100_000;
const MAX_EXECUTION_LOG_PAYLOAD_CHARS = 16_000;

function truncateString(value: string, max: number = MAX_STRING_PREVIEW_CHARS): string {
  if (value.length <= max) return value;
  const omitted = value.length - max;
  return `${value.slice(0, max)}\n...[truncated ${omitted} chars]`;
}

function sanitizeHistory(value: unknown): unknown {
  if (Array.isArray(value)) {
    const totalCount = value.length;
    const start = Math.max(totalCount - 3, 0);
    const tail = value.slice(start);
    return {
      _truncated: true,
      totalCount,
      lastSteps: tail.map((step) =>
        typeof step === "string" ? truncateString(step) : step
      ),
    };
  }

  if (typeof value === "string") {
    return {
      _truncated: true,
      totalLength: value.length,
      lastChunk: truncateString(value),
    };
  }

  return value;
}

function sanitizeLargeField(value: unknown): unknown {
  if (typeof value === "string") {
    return truncateString(value);
  }
  return value;
}

function sanitizeValueInternal(
  value: unknown,
  depth: number,
  context?: { toolName?: string; command?: string }
): unknown {
  if (value == null) return value;

  if (typeof value === "string") {
    return truncateString(value);
  }

  if (typeof value !== "object") {
    return value;
  }

  if (depth > 4) {
    // Avoid very deep recursion; summarize.
    return { _truncated: true, note: "max depth reached" };
  }

  if (Array.isArray(value)) {
    return value.map((item) => sanitizeValueInternal(item, depth + 1, context));
  }

  const input = value as Record<string, unknown>;
  const output: Record<string, unknown> = {};

  const toolNameFromSelf =
    typeof (input as any).toolName === "string"
      ? String((input as any).toolName)
      : undefined;
  const commandFromSelf =
    typeof (input as any).command === "string"
      ? String((input as any).command)
      : undefined;

  const effectiveToolName = toolNameFromSelf ?? context?.toolName;
  const effectiveCommand = commandFromSelf ?? context?.command;

  const isReadFile = effectiveToolName === "readFile";
  const isGetCodeMap = effectiveToolName === "getCodeMap";
  const isCatLike =
    (effectiveToolName === "executeCommand" ||
      effectiveToolName === "cloudExecuteCommand") &&
    typeof effectiveCommand === "string" &&
    /\bcat\b/.test(effectiveCommand);

  for (const [key, v] of Object.entries(input)) {
    if (key === "history") {
      output[key] = sanitizeHistory(v);
      continue;
    }

    if (key === "content") {
      if (isReadFile && typeof v === "string") {
        output[key] = truncateString(v, MAX_VISION_FIELD_CHARS);
      } else {
        output[key] = sanitizeLargeField(v);
      }
      continue;
    }

    if (key === "codeMap") {
      if (isGetCodeMap && typeof v === "string") {
        output[key] = truncateString(v, MAX_VISION_FIELD_CHARS);
      } else {
        output[key] = sanitizeLargeField(v);
      }
      continue;
    }

    if (key === "stdout" && isCatLike) {
      if (typeof v === "string") {
        output[key] = truncateString(v, MAX_VISION_FIELD_CHARS);
      } else {
        output[key] = v;
      }
      continue;
    }

    if (key === "fullFile" || key === "diff") {
      output[key] = sanitizeLargeField(v);
      continue;
    }

    output[key] = sanitizeValueInternal(v, depth + 1, {
      toolName: effectiveToolName,
      command: effectiveCommand,
    });
  }

  return output;
}

/**
 * Makes an arbitrary payload safe for logging and transport to the frontend.
 * - Truncates very large strings.
 * - Collapses `history` arrays to only the last few steps with metadata.
 * - Truncates `codeMap` / file-content-like fields.
 * - Enforces a global serialized-size cap by falling back to a summarized shape.
 */
export function makeExecutionPayloadLogSafe<T = unknown>(payload: T): T {
  try {
    // Detect whether this payload is coming from a vision-heavy tool where we
    // want to preserve up to MAX_VISION_FIELD_CHARS of content/stdout and
    // avoid collapsing the entire payload into a tiny summary object.
    let isVisionPayload = false;

    if (payload && typeof payload === "object" && !Array.isArray(payload)) {
      const input = payload as any;
      const toolName =
        typeof input.toolName === "string" ? String(input.toolName) : undefined;
      const command =
        typeof input.command === "string" ? String(input.command) : undefined;

      const isReadFile = toolName === "readFile";
      const isGetCodeMap = toolName === "getCodeMap";
      const isCatLike =
        (toolName === "executeCommand" || toolName === "cloudExecuteCommand") &&
        typeof command === "string" &&
        /\bcat\b/.test(command);

      isVisionPayload = isReadFile || isGetCodeMap || isCatLike;
    }

    const sanitized = sanitizeValueInternal(payload, 0) as T;

    try {
      const serialized = JSON.stringify(sanitized);

      // For vision-heavy tools (readFile, getCodeMap, cat-like commands), we
      // rely on per-field caps (MAX_VISION_FIELD_CHARS) and deliberately skip
      // the global payload-size summarization so the agent can see enough
      // file content/stdout in the UI.
      if (isVisionPayload) {
        return sanitized;
      }

      if (serialized.length <= MAX_EXECUTION_LOG_PAYLOAD_CHARS) {
        return sanitized;
      }

      const originalKeys =
        payload && typeof payload === "object" && !Array.isArray(payload)
          ? Object.keys(payload as Record<string, unknown>)
          : [];

      return {
        _truncated: true,
        reason: "payload too large for execution log",
        approxSize: serialized.length,
        keys: originalKeys,
        note: "vision payload truncated; consider narrowing readFile/getCodeMap scope or using smaller cat outputs.",
      } as unknown as T;
    } catch {
      // If JSON.stringify itself fails, fall back to minimal summary.
      return {
        _truncated: true,
        reason: "failed to serialize payload for size check",
      } as unknown as T;
    }
  } catch {
    return {
      _truncated: true,
      reason: "failed to sanitize payload",
    } as unknown as T;
  }
}

export { MAX_EXECUTION_LOG_PAYLOAD_CHARS, MAX_STRING_PREVIEW_CHARS };



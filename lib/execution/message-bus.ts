import { prisma } from "@/lib/prisma";
import { AgentRole, MessageType, MessageStatus, AgentMessage } from "@prisma/client";
import { appendSessionLog } from "@/lib/execution/file-logger";

function sanitizeForPostgres(value: unknown): unknown {
  if (typeof value === "string") {
    return value.replace(/\u0000/g, "");
  }
  if (Array.isArray(value)) {
    return value.map(sanitizeForPostgres);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([k, v]) => [
        k,
        sanitizeForPostgres(v),
      ])
    );
  }
  return value;
}

export interface NewMessage {
  sessionId: string;
  sourceAgent: AgentRole;
  targetAgent: AgentRole;
  eventType: MessageType;
  payload: Record<string, unknown>;
  correlationId?: string;
  replyToId?: string;
}

export class MessageBus {
  async postMessage(message: NewMessage): Promise<AgentMessage> {
    const correlationId = message.correlationId || this.generateCorrelationId();

    const taskRef =
      (message.payload as Record<string, unknown>)?.taskTitle ??
      (message.payload as Record<string, unknown>)?.taskId ??
      "—";
    appendSessionLog(
      message.sessionId,
      "info",
      `\n📨 [${message.sourceAgent}] -> @${message.targetAgent}: Отправил ${message.eventType} (Task: ${taskRef})`
    );

    return await prisma.agentMessage.create({
      data: {
        sessionId: message.sessionId,
        sourceAgent: message.sourceAgent,
        targetAgent: message.targetAgent,
        eventType: message.eventType,
        payload: message.payload as any,
        correlationId,
        replyToId: message.replyToId,
        status: MessageStatus.PENDING,
      },
    });
  }

  async getPendingMessagesFor(
    sessionId: string,
    agentType: AgentRole
  ): Promise<AgentMessage[]> {
    return await prisma.agentMessage.findMany({
      where: {
        sessionId,
        targetAgent: agentType,
        status: MessageStatus.PENDING,
      },
      orderBy: { createdAt: "asc" },
    });
  }

  async markAsProcessing(messageId: string): Promise<void> {
    await prisma.agentMessage.update({
      where: { id: messageId },
      data: { status: MessageStatus.PROCESSING },
    });
  }

  async markAsProcessed(
    messageId: string,
    responsePayload?: Record<string, unknown>
  ): Promise<void> {
    await prisma.agentMessage.update({
      where: { id: messageId },
      data: {
        status: MessageStatus.PROCESSED,
        processedAt: new Date(),
        ...(responsePayload && {
          payload: sanitizeForPostgres(responsePayload) as any,
        }),
      },
    });
  }

  async markAsFailed(messageId: string, error: string): Promise<void> {
    // Self-healing retry semantics for transient LLM / network errors live here so that
    // all agents benefit in a consistent way.
    const MAX_RETRIES = 3;

    // Helper: classify whether an error string looks transient (LLM 5xx, fetch failures, etc.)
    const isTransientError = (message: string): boolean => {
      const lower = message.toLowerCase();
      return (
        lower.includes("llm call failed") ||
        lower.includes("css analysis failed") ||
        lower.includes("operation failed") ||
        lower.includes("fetch failed") ||
        lower.includes("timeout") ||
        lower.includes("timed out") ||
        lower.includes("socket hang up") ||
        lower.includes("econnreset") ||
        lower.includes("ecanceled") ||
        lower.includes("503") ||
        lower.includes("502") ||
        lower.includes("500") ||
        lower.includes("429")
      );
    };

    let existing: AgentMessage | null = null;
    try {
      existing = await prisma.agentMessage.findUnique({
        where: { id: messageId },
      });
    } catch {
      // If we cannot load the message metadata, fall back to plain FAILED update below.
    }

    if (existing) {
      const rawMetadata = (existing.metadata as any) ?? {};
      const retryCounter = Number(rawMetadata.retryCounter ?? 0) || 0;
      const transient = isTransientError(error);

      if (transient && retryCounter < MAX_RETRIES) {
        const nextRetry = retryCounter + 1;
        const updatedMetadata = {
          ...rawMetadata,
          retryCounter: nextRetry,
          lastError: error,
        };

        await prisma.agentMessage.update({
          where: { id: messageId },
          data: {
            status: MessageStatus.PENDING,
            error: sanitizeForPostgres(error) as any,
            metadata: updatedMetadata as any,
          },
        });

        try {
          await appendSessionLog(
            existing.sessionId,
            "error",
            `[AHP] Message ${messageId} transient LLM error, retry ${nextRetry}/${MAX_RETRIES}: ${error}`
          );
        } catch {
          // Logging failures must not break retry behaviour.
        }

        return;
      }

      // Max retries exhausted or non-transient error: mark as FAILED and preserve retryCounter.
      const finalMetadata =
        typeof rawMetadata === "object" && rawMetadata !== null
          ? { ...rawMetadata, retryCounter }
          : { retryCounter };

      await prisma.agentMessage.update({
        where: { id: messageId },
        data: {
          status: MessageStatus.FAILED,
          error: sanitizeForPostgres(error) as any,
          metadata: finalMetadata as any,
        },
      });
      return;
    }

    // Fallback: if we couldn't load the message for some reason, keep legacy behaviour.
    await prisma.agentMessage.update({
      where: { id: messageId },
      data: {
        status: MessageStatus.FAILED,
        error: sanitizeForPostgres(error) as any,
      },
    });
  }

  async getConversation(correlationId: string): Promise<AgentMessage[]> {
    return await prisma.agentMessage.findMany({
      where: { correlationId },
      orderBy: { createdAt: "asc" },
    });
  }

  private generateCorrelationId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

export const messageBus = new MessageBus();

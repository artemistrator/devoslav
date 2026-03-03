import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { messageBus } from "@/lib/execution/message-bus";
import { AgentRole, MessageType, MessageStatus } from "@prisma/client";
import { createAgentForRole, groupBy, findNextTask, isSessionComplete } from "@/lib/execution/agent-factory";
import { getProjectDir } from "@/lib/project-workspace";
import { initProjectWorkspace } from "@/lib/project/init-workspace";
import { ExecutionSessionManager } from "@/lib/execution/session-manager";
import { runReflexologistForSession } from "@/lib/agents/reflexologist";
import { appendSessionLog } from "@/lib/execution/file-logger";
import { ensureContainer, destroyContainer } from "@/lib/execution/container-manager";
import { makeExecutionPayloadLogSafe } from "@/lib/execution/log-sanitizer";

const MAX_ITERATIONS = 500;
const ITERATION_PAUSE_MS = 1000;
const STUCK_THRESHOLD_MS = 60 * 60 * 1000; // 1 hour without log updates = stuck
const STUCK_CHECK_INTERVAL = 10; // Check every N iterations to limit DB load

async function persistExecutionLog(
  sessionId: string,
  level: "info" | "error" | "success" | "warn",
  message: string
) {
  try {
    const type =
      level === "error" ? "error" :
      level === "success" ? "success" :
      level === "warn" ? "info" : // persist as info, metadata keeps level for display
      "info";

    await prisma.executionLog.create({
      data: {
        sessionId,
        type,
        message,
        metadata: {
          eventType: "system",
          data: makeExecutionPayloadLogSafe({ level }),
        },
      },
    });
    appendSessionLog(sessionId, level, message);
  } catch (error) {
    console.error("[AHP Dispatcher] Failed to persist executionLog:", error);
  }
}

async function runAHPDispatcher(sessionId: string) {
  try {
    const session = await prisma.executionSession.findUnique({
      where: { id: sessionId },
      include: { project: true },
    });

    if (!session) {
      console.error(`[AHP Dispatcher] Session ${sessionId} not found`);
      return;
    }

    let projectType: string | null = null;
    if (session.planId) {
      const plan = await prisma.plan.findUnique({
        where: { id: session.planId },
        select: { projectType: true },
      });
      projectType = plan?.projectType ?? null;
    }

    console.log(`[AHP Dispatcher ${sessionId}] Starting dispatcher...`);
    await persistExecutionLog(sessionId, "info", "[AHP] Dispatcher started");

    const sessionMetadata = (session.metadata as Record<string, any>) || {};
    const autoApprove = sessionMetadata.autoApprove || false;

    const projectDir = getProjectDir(session.projectId);
    await initProjectWorkspace(session.projectId);
    console.log(`[AHP Dispatcher] Initialized workspace: ${projectDir}`);
    appendSessionLog(sessionId, "info", `[AHP] Initialized workspace: ${projectDir}`);

    try {
      await ensureContainer(sessionId, projectDir, session.projectId, projectType);
      const containerName = `ai-orch-session-${sessionId.replace(/[^a-zA-Z0-9_.-]/g, "-")}`;
      await persistExecutionLog(
        sessionId,
        "info",
        `[AHP] Ensured container ${containerName} for workspace: ${projectDir}`
      );
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      console.error(`[AHP Dispatcher] Failed to ensure container for session ${sessionId}:`, error);
      await persistExecutionLog(
        sessionId,
        "error",
        `[AHP] Failed to ensure container for session: ${errMsg}`
      );
    }

    if (session.planId) {
      const todoCount = await prisma.task.count({
        where: { planId: session.planId, status: "TODO" },
      });
      if (todoCount === 0) {
        const updated = await prisma.task.updateMany({
          where: { planId: session.planId, status: "IN_PROGRESS" },
          data: { status: "TODO" },
        });
        if (updated.count > 0) {
          console.log(`[AHP Dispatcher] Reset ${updated.count} IN_PROGRESS tasks to TODO (stuck from previous run)`);
          await persistExecutionLog(sessionId, "info",
            `[AHP] Reset ${updated.count} IN_PROGRESS tasks to TODO (stuck from previous run)`);
        }
      }
    }

    const config = {
      onLog: async (level: "info" | "error" | "success" | "warn", message: string) => {
        console.log(`[AHP Dispatcher/${sessionId}] [${level.toUpperCase()}] ${message}`);
        await persistExecutionLog(sessionId, level, message);
      },
      mode: "cloud" as const,
    };

    let iteration = 0;

    while (iteration < MAX_ITERATIONS) {
      iteration++;

      if (iteration % STUCK_CHECK_INTERVAL === 0) {
        const lastLog = await prisma.executionLog.findFirst({
          where: { sessionId },
          orderBy: { createdAt: "desc" },
          select: { createdAt: true },
        });
        const lastActivityAt = lastLog?.createdAt ?? session.createdAt;
        const idleMs = Date.now() - lastActivityAt.getTime();
        if (idleMs > STUCK_THRESHOLD_MS) {
          console.log(
            `[AHP Dispatcher] Session ${sessionId} stuck (no log updates for ${Math.round(idleMs / 60000)} min). Stopping.`
          );
          await persistExecutionLog(
            sessionId,
            "info",
            "Session stuck (no progress for > 1 hour). Stopping."
          );
          await finalizeSession(
            sessionId,
            session.projectId,
            session.planId,
            "Stuck: no log updates for > 1 hour"
          );
          return;
        }
      }

      console.log(`[AHP Dispatcher] Iteration ${iteration}/${MAX_ITERATIONS}`);
      await persistExecutionLog(
        sessionId,
        "info",
        `[AHP] Iteration ${iteration}/${MAX_ITERATIONS}`
      );

      const isComplete = await isSessionComplete(sessionId);
      if (isComplete) {
        console.log(`[AHP Dispatcher] Session ${sessionId} is complete`);
        break;
      }

      const pendingMessages = await prisma.agentMessage.findMany({
        where: {
          sessionId,
          status: { in: [MessageStatus.PENDING, MessageStatus.PROCESSING] },
        },
        orderBy: { createdAt: "asc" },
      });

      console.log(`[AHP Dispatcher] Found ${pendingMessages.length} pending/processing messages`);
      appendSessionLog(sessionId, "info", `[AHP] Found ${pendingMessages.length} pending/processing messages`);

      if (pendingMessages.length === 0) {
        console.log(`[AHP Dispatcher] No pending messages, finding next task...`);
        appendSessionLog(sessionId, "info", "[AHP] No pending messages, finding next task...");

        const nextItem = await findNextTask(session.projectId, session.planId);

        if (!nextItem) {
          // Нет сообщений и нет доступных задач/тикетов. Проверяем состояние плана:
          if (session.planId) {
            const planTasks = await prisma.task.findMany({
              where: { planId: session.planId },
              select: { id: true, status: true },
            });

            const allDone = planTasks.length > 0 && planTasks.every((t) => t.status === "DONE");
            if (!allDone) {
              const nonDone = planTasks.filter((t) => t.status !== "DONE");
              const byStatus = nonDone.reduce<Record<string, string[]>>((acc, t) => {
                (acc[t.status] = acc[t.status] || []).push(t.id);
                return acc;
              }, {});

              const summary = Object.entries(byStatus)
                .map(([status, ids]) => `${status}: [${ids.join(", ")}]`)
                .join("; ");

              const msg =
                `[AHP] Dispatcher blocked: no runnable tasks, ` +
                `but some plan tasks are not DONE. Details: ${summary}`;
              console.log(`[AHP Dispatcher] ${msg}`);
              await persistExecutionLog(sessionId, "info", msg);
            } else {
              console.log(`[AHP Dispatcher] No more tasks/tickets to process (all plan tasks DONE)`);
              appendSessionLog(sessionId, "info", "[AHP] No more tasks/tickets to process (all plan tasks DONE)");
            }
          } else {
            console.log(`[AHP Dispatcher] No more tasks/tickets to process (no planId)`);
            appendSessionLog(sessionId, "info", "[AHP] No more tasks/tickets to process (no planId)");
          }
          break;
        }

        console.log(`[AHP Dispatcher] Found next ${nextItem.type}: ${nextItem.data.id}`);
        appendSessionLog(sessionId, "info", `[AHP] Found next ${nextItem.type}: ${nextItem.data.id}`);

        if (nextItem.type === "ticket") {
          const ticket = nextItem.data;
          await prisma.ticket.update({
            where: { id: ticket.id },
            data: { status: "IN_PROGRESS" },
          });

          await messageBus.postMessage({
            sessionId,
            sourceAgent: AgentRole.TEAMLEAD,
            targetAgent: ticket.relatedTaskId
              ? (await prisma.task.findUnique({
                  where: { id: ticket.relatedTaskId },
                  select: { executorAgent: true },
                }))?.executorAgent || AgentRole.TASK_EXECUTOR
              : AgentRole.TASK_EXECUTOR,
            eventType: MessageType.TICKET_REQUEST,
            payload: { ticketId: ticket.id, relatedTaskId: ticket.relatedTaskId },
          });

          console.log(`[AHP Dispatcher] Created TICKET_REQUEST for ticket ${ticket.id}`);
          appendSessionLog(sessionId, "info", `[AHP] Created TICKET_REQUEST for ticket ${ticket.id}`);
        } else {
          const task = nextItem.data;
          await prisma.task.update({
            where: { id: task.id },
            data: { status: "IN_PROGRESS" },
          });
          await messageBus.postMessage({
            sessionId,
            sourceAgent: AgentRole.TEAMLEAD,
            targetAgent: task.executorAgent || AgentRole.TASK_EXECUTOR,
            eventType: MessageType.TASK_REQUEST,
            payload: { taskId: task.id },
          });

          console.log(`[AHP Dispatcher] Created TASK_REQUEST for task ${task.id}`);
          appendSessionLog(sessionId, "info", `[AHP] Created TASK_REQUEST for task ${task.id}`);
        }

        continue;
      }

      const pendingOnlyMessages = pendingMessages.filter(
        (m) => m.status === MessageStatus.PENDING
      );

      if (pendingOnlyMessages.length === 0) {
        console.log(`[AHP Dispatcher] Only processing messages, waiting...`);
        appendSessionLog(sessionId, "info", "[AHP] Only processing messages, waiting...");
        await new Promise((resolve) => setTimeout(resolve, ITERATION_PAUSE_MS));
        continue;
      }

      const messagesByAgent = groupBy(pendingOnlyMessages, "targetAgent");

      console.log(
        `[AHP Dispatcher] Active agents: ${Object.keys(messagesByAgent).join(", ")}`
      );
      appendSessionLog(sessionId, "info", `[AHP] Active agents: ${Object.keys(messagesByAgent).join(", ")}`);

      const agentPromises = Object.entries(messagesByAgent).map(
        async ([agentRole, messages]) => {
          try {
            console.log(`[AHP Dispatcher] Creating agent for role: ${agentRole}`);
            appendSessionLog(sessionId, "info", `[AHP] Creating agent for role: ${agentRole}`);

            const agent = createAgentForRole(
              agentRole as AgentRole,
              sessionId,
              session.projectId,
              autoApprove,
              config.onLog,
              config.mode
            );

            console.log(`[AHP Dispatcher] Running ${agentRole} agent...`);
            appendSessionLog(sessionId, "info", `[AHP] Running ${agentRole} agent...`);
            await agent.run();

            console.log(`[AHP Dispatcher] ${agentRole} agent completed`);
            appendSessionLog(sessionId, "info", `[AHP] ${agentRole} agent completed`);
          } catch (error) {
            const errorMsg = error instanceof Error ? error.message : String(error);
            console.error(`[AHP Dispatcher] Agent ${agentRole} failed:`, error);
            appendSessionLog(sessionId, "error", `[AHP] Agent ${agentRole} failed: ${errorMsg}`);
            // Agent-level crash protection: only reset messages stuck in PROCESSING.
            // FAILED messages are handled via per-message retry logic in the message bus.
            try {
              const updated = await prisma.agentMessage.updateMany({
                where: {
                  sessionId,
                  targetAgent: agentRole as AgentRole,
                  status: MessageStatus.PROCESSING,
                },
                data: { status: MessageStatus.PENDING },
              });
              if (updated.count > 0) {
                appendSessionLog(
                  sessionId,
                  "info",
                  `[AHP] Reset ${updated.count} PROCESSING message(s) to PENDING after agent crash (FAILED messages keep their final status)`
                );
              }
            } catch (resetErr) {
              console.error(`[AHP Dispatcher] Failed to reset PROCESSING messages to PENDING:`, resetErr);
            }
          }
        }
      );

      await Promise.allSettled(agentPromises);

      await new Promise((resolve) => setTimeout(resolve, ITERATION_PAUSE_MS));
    }

    const stopReason = iteration >= MAX_ITERATIONS
      ? "MAX_ITERATIONS reached"
      : "No more tasks to process";

    console.log(`[AHP Dispatcher ${sessionId}] Loop finished. Reason: ${stopReason}`);
    await persistExecutionLog(sessionId, "info",
      `[AHP] Session stopping: ${stopReason}`);

    await finalizeSession(sessionId, session.projectId, session.planId, stopReason);
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    console.error(`[AHP Dispatcher] Error running session ${sessionId}:`, error);
    appendSessionLog(sessionId, "error", `[AHP] Dispatcher error: ${errMsg}`);
    try {
      const session = await prisma.executionSession.findUnique({
        where: { id: sessionId },
        select: { projectId: true, planId: true },
      });
      if (session) {
        await finalizeSession(
          sessionId,
          session.projectId,
          session.planId,
          `Dispatcher error: ${errMsg}`
        );
      } else {
        await destroyContainer(sessionId);
      }
    } catch (finalizeErr) {
      console.error(`[AHP Dispatcher] Finalize on error failed for ${sessionId}:`, finalizeErr);
      await destroyContainer(sessionId).catch(() => {});
    }
  }
}

async function finalizeSession(
  sessionId: string,
  projectId: string,
  planId: string | null | undefined,
  reason?: string
) {
  appendSessionLog(sessionId, "info", `[AHP] Finalizing session. Reason: ${reason || "All tasks completed"}`);
  console.log(`[AHP Dispatcher] Finalizing session ${sessionId}... Reason: ${reason || "All tasks completed"}`);

  const sessionManager = ExecutionSessionManager.getInstance();

  try {
    await destroyContainer(sessionId);
    appendSessionLog(sessionId, "info", "[AHP] Session container destroyed");
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    appendSessionLog(sessionId, "error", `[AHP] Failed to destroy session container: ${errMsg}`);
  }

  if (planId) {
    const planTasks = await prisma.task.findMany({
      where: { planId },
      select: { id: true, status: true },
    });

    const allDone = planTasks.length > 0 && planTasks.every((t) => t.status === "DONE");
    const hasWaitingApproval = planTasks.some(
      (t) => t.status === "WAITING_APPROVAL"
    );
    const hasNonDone = planTasks.some((t) => t.status !== "DONE");

    if (allDone) {
      await prisma.project.update({
        where: { id: projectId },
        data: { status: "COMPLETED" },
      });

      console.log(`[AHP Dispatcher] Project ${projectId} marked as COMPLETED`);

      await prisma.executionLog.create({
        data: {
          sessionId,
          type: "success",
          message:
            "✅ Session completed. All tasks DONE via Agent Hive Protocol.",
          metadata: { eventType: "session_completed" },
        },
      });

      await sessionManager.stopSession(sessionId, "All tasks DONE");
    } else if (hasNonDone) {
      const nonDoneByStatus = planTasks
        .filter((t) => t.status !== "DONE")
        .reduce<Record<string, string[]>>((acc, t) => {
          (acc[t.status] = acc[t.status] || []).push(t.id);
          return acc;
        }, {});

      const summary = Object.entries(nonDoneByStatus)
        .map(([status, ids]) => `${status}: [${ids.join(", ")}]`)
        .join("; ");

      await prisma.executionLog.create({
        data: {
          sessionId,
          type: "info",
          message:
            "[AHP] Session paused: some tasks are not DONE. Waiting for human intervention or approvals. " +
            `Details: ${summary}`,
          metadata: {
            eventType: "session_paused_blocked",
            data: makeExecutionPayloadLogSafe({
              sessionId,
              planId,
              summary,
            }),
          },
        },
      });

      if (hasWaitingApproval) {
        console.log(
          `[AHP Dispatcher] Some tasks remain in WAITING_APPROVAL after automatic QA retries. Human review required.`
        );
      }

      await sessionManager.pauseSession(sessionId);
    } else {
      // Нет задач в плане — просто останавливаем сессию.
      await sessionManager.stopSession(sessionId, reason || "No tasks in plan");
    }
  } else {
    // Сессия без плана: используем исходный stopSession, но более честный reason.
    await sessionManager.stopSession(sessionId, reason || "No planId, dispatcher finished");
  }

  await prisma.executionLog.create({
    data: {
      sessionId,
      type: "info",
      message: `[AHP] Session stopped. Reason: ${reason || "All tasks completed"}`,
      metadata: {
        eventType: "session_stopped",
        data: makeExecutionPayloadLogSafe({ sessionId, reason }),
      },
    },
  });

  try {
    await runReflexologistForSession({
      projectId,
      sessionId,
      planId,
      mode: "final",
      maxInsights: 3,
    });
  } catch (error) {
    console.error(
      `[AHP Dispatcher] Reflexologist failed for session ${sessionId}:`,
      error
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { sessionId } = body;

    if (!sessionId) {
      return NextResponse.json({ error: "sessionId is required" }, { status: 400 });
    }

    console.log("\n\n🚀 [AHP Dispatcher] Starting Agent Hive Protocol...\n\n");

    runAHPDispatcher(sessionId);

    return NextResponse.json({
      success: true,
      message: "AHP Dispatcher started",
      sessionId,
    });
  } catch (error) {
    console.error("[AHP Dispatcher] Error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to start AHP" },
      { status: 500 }
    );
  }
}

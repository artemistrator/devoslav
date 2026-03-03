import { AgentRole, MessageType, MessageStatus } from "@prisma/client";
import { BaseAgent, AgentConfig } from "@/lib/agents/base-agent";
import { TaskExecutorAgent } from "@/lib/agents/task-executor-agent";
import { CSSAgent } from "@/lib/agents/css-agent";
import { QAAgent } from "@/lib/agents/qa-agent";
import { TeamLeadAgent } from "@/lib/agents/team-lead-agent";

export function createAgentForRole(
  agentRole: AgentRole,
  sessionId: string,
  projectId: string,
  autoApprove: boolean,
  onLog: (level: "info" | "error" | "success" | "warn", message: string) => void,
  mode?: "local" | "cloud"
): BaseAgent {
  const baseConfig: AgentConfig = {
    sessionId,
    projectId,
    agentRole,
    autoApprove,
    onLog,
    mode: mode ?? "local",
  };

  switch (agentRole) {
    case AgentRole.TASK_EXECUTOR:
      return new TaskExecutorAgent(baseConfig);
    case AgentRole.QA:
      return new QAAgent(baseConfig);
    case AgentRole.CSS:
      return new CSSAgent(baseConfig);
    case AgentRole.TEAMLEAD:
      return new TeamLeadAgent(baseConfig);
    case AgentRole.BACKEND:
    case AgentRole.DEVOPS:
    case AgentRole.CURSOR:
      return new TaskExecutorAgent(baseConfig);
    default:
      throw new Error(`Unknown agent role: ${agentRole}`);
  }
}

export function groupBy<T>(array: T[], key: keyof T): Record<string, T[]> {
  return array.reduce((result, item) => {
    const groupKey = String(item[key]);
    (result[groupKey] = result[groupKey] || []).push(item);
    return result;
  }, {} as Record<string, T[]>);
}

interface NextItem {
  type: "task" | "ticket";
  data: any;
}

export async function findNextTask(
  projectId: string,
  planId: string | null | undefined
): Promise<NextItem | null> {
  // Тикеты подхватываем даже без planId (сессия может быть запущена только для тикетов).
  const openTicket = await prisma.ticket.findFirst({
    where: {
      projectId,
      status: "OPEN",
      relatedTaskId: { not: null },
    },
    orderBy: { createdAt: "asc" },
  });

  if (openTicket) {
    return { type: "ticket", data: openTicket };
  }

  if (!planId) return null;

  // Выбираем только такие TODO-задачи, у которых либо вообще нет зависимостей,
  // либо ВСЕ зависимости находятся в статусе DONE.
  const task = await prisma.task.findFirst({
    where: {
      planId,
      status: "TODO",
      AND: [
        {
          OR: [
            // Независимая задача
            { dependencies: { none: {} } },
            // Все зависимости завершены
            { dependencies: { every: { dependsOn: { status: "DONE" } } } },
          ],
        },
      ],
    },
    orderBy: { createdAt: "asc" },
  });

  if (task) {
    return { type: "task", data: task };
  }

  // Если TODO-задач без незавершённых зависимостей не нашли, но при этом в плане
  // ещё остаются незавершённые задачи, логируем это как «заблокированное» состояние.
  const remaining = await prisma.task.count({
    where: {
      planId,
      status: { in: ["TODO", "IN_PROGRESS", "REVIEW", "WAITING_APPROVAL", "REJECTED"] },
    },
  });

  if (remaining > 0) {
    console.log(
      `[AgentFactory] No runnable TODO tasks for plan ${planId}: all remaining tasks are blocked by dependencies or non-DONE statuses (count=${remaining}).`
    );
  }

  return null;
}

export async function isSessionComplete(sessionId: string): Promise<boolean> {
  const session = await prisma.executionSession.findUnique({
    where: { id: sessionId },
    include: {
      project: {
        include: {
          plans: true,
        },
      },
    },
  });

  if (!session) return true;

  const pendingMessages = await prisma.agentMessage.count({
    where: {
      sessionId,
      status: { in: ["PENDING", "PROCESSING"] },
    },
  });

  if (pendingMessages > 0) return false;

  if (session.planId) {
    const openTickets = await prisma.ticket.count({
      where: { projectId: session.projectId, status: "OPEN" },
    });
    const planTasks = await prisma.task.findMany({
      where: { planId: session.planId },
      select: { status: true },
    });

    if (planTasks.length === 0) {
      // Нет задач в плане — считаем завершённым, если нет открытых тикетов.
      return openTickets === 0;
    }

    const allDone = planTasks.every((t) => t.status === "DONE");

    return allDone && openTickets === 0;
  }

  // Без плана считаем сессию незавершённой, если есть OPEN-тикеты с привязкой к задаче.
  const openTicketsWithTask = await prisma.ticket.count({
    where: {
      projectId: session.projectId,
      status: "OPEN",
      relatedTaskId: { not: null },
    },
  });

  return openTicketsWithTask === 0;
}

import { prisma } from "@/lib/prisma";
import { messageBus } from "@/lib/execution/message-bus";

import { NextResponse } from "next/server";

import { AgentRole, MessageStatus, MessageType } from "@prisma/client";
import { prisma } from "@/lib/prisma";
import { replanTasks } from "@/lib/agents/architect";

export async function POST(
  request: Request,
  { params }: { params: Promise<{ taskId: string }> }
) {
  try {
    const { taskId } = await params;
    const body = await request.json().catch(() => ({}));
    const { approved } = body;

    if (typeof approved !== "boolean") {
      return NextResponse.json(
        { error: "approved field is required (boolean)" },
        { status: 400 }
      );
    }

    const task = await prisma.task.findUnique({
      where: { id: taskId },
      include: {
        plan: {
          include: {
            project: { select: { id: true } },
          },
        },
      },
    });

    if (!task) {
      return NextResponse.json({ error: "Task not found" }, { status: 404 });
    }

    if (task.status !== "WAITING_APPROVAL") {
      return NextResponse.json(
        { error: "Task is not in WAITING_APPROVAL status" },
        { status: 400 }
      );
    }

    let newStatus = "REVIEW";
    let commentContent = "";

    if (approved) {
      newStatus = "DONE";
      commentContent = "Task approved and moved to DONE status.";
    } else {
      newStatus = "REVIEW";
      commentContent = "Task rejected. Further changes are required.";
    }

    const updated = await prisma.task.update({
      where: { id: taskId },
      data: { status: newStatus as any },
    });

    await prisma.comment.create({
      data: {
        taskId,
        content: commentContent,
        authorRole: "TEAMLEAD",
      },
    });

    if (approved && task.plan?.project?.id) {
      try {
        const projectId = task.plan.project.id;
        const replanResult = await replanTasks(projectId, taskId);
        if (process.env.NODE_ENV !== "production" && replanResult.needsReplan) {
          console.log("[Dynamic Replanning]", replanResult);
        }

        // Try to resume a paused AHP execution session and notify TeamLead via QA_RESPONSE.
        const session = await prisma.executionSession.findFirst({
          where: {
            projectId,
            status: "PAUSED",
            engine: "ahp",
          },
          orderBy: { createdAt: "desc" },
        });

        if (session) {
          await prisma.agentMessage.create({
            data: {
              sessionId: session.id,
              sourceAgent: AgentRole.QA,
              targetAgent: AgentRole.TEAMLEAD,
              eventType: MessageType.QA_RESPONSE,
              status: MessageStatus.PENDING,
              payload: {
                taskId,
                finalStatus: "DONE",
                reasoning: "Human approved the task. Resuming execution.",
              },
            },
          });

          await prisma.executionSession.update({
            where: { id: session.id },
            data: { status: "RUNNING" },
          });

          const baseUrl =
            process.env.INTERNAL_APP_URL ||
            `http://127.0.0.1:${process.env.PORT || 3000}`;

          fetch(`${baseUrl}/api/execution-sessions/run-ahp`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sessionId: session.id }),
          }).catch((err) => {
            if (process.env.NODE_ENV !== "production") {
              console.error(
                "[tasks:approve] Failed to resume AHP dispatcher:",
                err
              );
            }
          });
        }
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.error("[Dynamic Replanning / AHP Resume] Error:", error);
        }
      }
    }

    return NextResponse.json({
      success: true,
      task: updated,
      approved,
    });
  } catch (error) {
    if (process.env.NODE_ENV !== "production") {
      console.error("[tasks:approve]", error);
    }
    return NextResponse.json(
      { error: "Failed to process approval" },
      { status: 500 }
    );
  }
}

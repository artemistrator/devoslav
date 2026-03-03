import { NextResponse } from "next/server";

import { prisma } from "@/lib/prisma";
import { runTaskGenerationJob } from "@/lib/task-generation/job-runner";

export async function POST(request: Request) {
  try {
    console.log("[generate-tasks] Starting async job request");
    const body = await request.json().catch(() => ({}));
    const planId = typeof body?.planId === "string" ? body.planId : "";

    if (!planId) {
      console.log("[generate-tasks] Missing planId");
      return NextResponse.json({ error: "planId is required" }, { status: 400 });
    }

    console.log("[generate-tasks] Fetching plan");
    const plan = await prisma.plan.findUnique({
      where: { id: planId },
      include: { project: true },
    });

    if (!plan) {
      console.log("[generate-tasks] Plan not found:", planId);
      return NextResponse.json({ error: "Plan not found" }, { status: 404 });
    }

    const taskGenerationJobClient = (prisma as any).taskGenerationJob;

    // Reuse existing pending/running job for this plan if it exists
    const existingJob = await taskGenerationJobClient.findFirst({
      where: {
        planId,
        status: { in: ["PENDING", "RUNNING"] },
      },
      orderBy: { createdAt: "desc" },
    });

    if (existingJob) {
      console.log(
        "[generate-tasks] Reusing existing job for plan:",
        planId,
        "jobId:",
        existingJob.id,
      );
      return NextResponse.json({
        jobId: existingJob.id,
        status: existingJob.status,
        progress: existingJob.progress ?? 0,
      });
    }

    console.log("[generate-tasks] Creating new TaskGenerationJob for plan:", planId);

    const job = await taskGenerationJobClient.create({
      data: {
        projectId: plan.projectId,
        planId: plan.id,
        executionSessionId: null,
        status: "PENDING",
        progress: 0,
        ideaText: plan.project.ideaText,
        input: body,
      },
    });

    // Fire-and-forget background processing, similar to run-ahp dispatcher
    runTaskGenerationJob(job.id).catch((error) => {
      console.error(
        "[generate-tasks] Background job failed for jobId",
        job.id,
        ":",
        error,
      );
    });

    console.log("[generate-tasks] Job created and dispatched:", job.id);

    return NextResponse.json({
      jobId: job.id,
      status: job.status,
      progress: job.progress ?? 0,
    });
  } catch (error) {
    console.error("[generate-tasks] Unhandled error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "Failed to start task generation job",
      },
      { status: 500 },
    );
  }
}


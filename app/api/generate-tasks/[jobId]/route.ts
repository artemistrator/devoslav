import { NextRequest, NextResponse } from "next/server";

import { prisma } from "@/lib/prisma";

export async function GET(
  _request: NextRequest,
  { params }: { params: { jobId: string } },
) {
  try {
    const { jobId } = params;

    if (!jobId) {
      return NextResponse.json({ error: "jobId is required" }, { status: 400 });
    }

    const taskGenerationJobClient = (prisma as any).taskGenerationJob;

    const job = await taskGenerationJobClient.findUnique({
      where: { id: jobId },
    });

    if (!job) {
      return NextResponse.json({ error: "Job not found" }, { status: 404 });
    }

    return NextResponse.json({
      jobId: job.id,
      status: job.status,
      progress: job.progress ?? 0,
      errorMessage: job.errorMessage ?? null,
      resultSummary: job.resultSummary ?? null,
    });
  } catch (error) {
    console.error("[generate-tasks-status] Unhandled error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Failed to fetch task generation job status",
      },
      { status: 500 },
    );
  }
}


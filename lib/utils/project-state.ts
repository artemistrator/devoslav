import { promises as fs } from "fs";
import { join } from "path";
import { prisma } from "@/lib/prisma";
import { getProjectDir } from "@/lib/project-workspace";

function formatIsoDate(date: Date): string {
  return date.toISOString();
}

function formatTaskChecklistStatus(status: string): "[x]" | "[~]" | "[ ]" {
  if (status === "DONE" || status === "WAITING_APPROVAL") {
    return "[x]";
  }
  if (status === "IN_PROGRESS" || status === "REVIEW") {
    return "[~]";
  }
  return "[ ]";
}

export async function updateProjectStateFile(
  projectId: string,
  completedTaskId?: string
): Promise<void> {
  const project = await prisma.project.findUnique({
    where: { id: projectId },
    include: {
      plans: {
        where: { selected: true },
        include: {
          tasks: true,
        },
      },
    },
  });

  if (!project) {
    throw new Error(`Project not found for id=${projectId}`);
  }

  const activePlan = project.plans.find((p) => p.selected) ?? project.plans[0] ?? null;

  const lastCompletedTask =
    completedTaskId && activePlan
      ? activePlan.tasks.find((t) => t.id === completedTaskId) ??
        (await prisma.task.findUnique({ where: { id: completedTaskId } }))
      : null;

  const now = new Date();
  const lines: string[] = [];

  lines.push("# 🧠 PROJECT STATE");
  lines.push(`*Last updated: ${formatIsoDate(now)}*`);
  if (lastCompletedTask) {
    lines.push(`*Last completed task: ${lastCompletedTask.title} (id: ${lastCompletedTask.id})*`);
  } else {
    lines.push(`*Last completed task: n/a*`);
  }
  lines.push("");

  lines.push(
    `## 🎯 Active Plan: ${activePlan ? activePlan.title : "(no active plan selected)"}`
  );
  if (activePlan?.description) {
    lines.push(activePlan.description);
  }
  lines.push("");

  lines.push("## 📋 Task Checklist");
  if (activePlan && activePlan.tasks.length > 0) {
    const sortedTasks = [...activePlan.tasks].sort((a, b) =>
      a.createdAt < b.createdAt ? -1 : a.createdAt > b.createdAt ? 1 : 0
    );
    for (const task of sortedTasks) {
      const box = formatTaskChecklistStatus(task.status);
      lines.push(`- ${box} ${task.title} (id: ${task.id})`);
    }
  } else {
    lines.push("- No tasks in active plan");
  }
  lines.push("");

  lines.push("## 🏗️ Technical Stack & Details");
  if (activePlan) {
    lines.push(`- Tech Stack: ${activePlan.techStack}`);
    if (activePlan.projectType) {
      lines.push(`- Project Type: ${activePlan.projectType}`);
    }
  }
  if (project.context) {
    lines.push("");
    lines.push(project.context);
  }
  lines.push("");

  const content = lines.join("\n");

  const projectDir = getProjectDir(projectId);
  const targetPath = join(projectDir, "PROJECT_STATE.md");

  await fs.mkdir(projectDir, { recursive: true });
  await fs.writeFile(targetPath, content, "utf-8");

  const existing = await prisma.projectFile.findFirst({
    where: { projectId, name: "PROJECT_STATE.md" },
  });

  if (existing) {
    await prisma.projectFile.update({
      where: { id: existing.id },
      data: {
        content,
      },
    });
  } else {
    await prisma.projectFile.create({
      data: {
        projectId,
        name: "PROJECT_STATE.md",
        url: `projects/${projectId}/PROJECT_STATE.md`,
        mimeType: "text/markdown",
        content,
      },
    });
  }
}


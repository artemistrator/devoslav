import { generateText } from "ai";
import { z } from "zod";
import { AgentRole, TaskStatus } from "@prisma/client";

import { prisma } from "@/lib/prisma";
import {
  getModel,
  getProviderApiKey,
  resolveProvider,
} from "@/lib/ai/providers";
import { trackAIUsage } from "@/lib/ai/call";
import { generateADR, saveDocToClient } from "@/lib/agents/doc-writer";
import { getCompactProjectContext } from "@/lib/agents/project-context";
import { generateTextZai } from "@/lib/ai/zai";
import { getLLMSettings } from "@/lib/settings";
import { searchGlobalInsights } from "@/lib/rag/search";
import { logInfo } from "@/lib/execution/file-logger";
import {
  createLLMAbortSignal,
  HEAVY_LLM_TIMEOUT_MS,
} from "@/lib/ai/timeout";

const taskSchema = z.object({
  estimatedComplexity: z.enum(["S", "M", "L", "XL"]).default("M"),
  projectType: z
    .enum([
      "static", // plain HTML/CSS/JS, no build system
      "frontend", // React, Vue, Vite, Next.js etc
      "backend", // Node.js, Python, Go, Rust API etc
      "fullstack", // frontend + backend together
      "script", // single script, CLI tool, automation
    ])
    .default("frontend"),
  reasoning: z.string().default(""),
  tasks: z
    .array(
      z.object({
        title: z.string(),
        description: z.string(),
        executorAgent: z.enum(["TASK_EXECUTOR", "BACKEND", "DEVOPS"]),
        branchName: z.string().optional(),
        difficulty: z.string().optional(),
        dependencyIndices: z.array(z.number()).default([]),
        verificationCriteria: z.object({
          artifacts: z.array(z.string()).default([]),
          manualCheck: z.string().default(""),
          automatedCheck: z.string().optional(),
        }),
      })
    )
    .default([]),
});

export type TaskGenerationJobStatus =
  | "PENDING"
  | "RUNNING"
  | "DONE"
  | "ERROR"
  | "CANCELLED";

export async function runTaskGenerationJob(jobId: string): Promise<void> {
  console.log("[task-generation] Starting job", jobId);

  const taskGenerationJobClient = (prisma as any).taskGenerationJob;

  const job = await taskGenerationJobClient.findUnique({
    where: { id: jobId },
    include: {
      plan: {
        include: { project: true },
      },
    },
  });

  if (!job) {
    console.error("[task-generation] Job not found:", jobId);
    return;
  }

  if (["DONE", "ERROR", "CANCELLED"].includes(job.status)) {
    console.log(
      "[task-generation] Job already finished, skipping:",
      jobId,
      "status:",
      job.status,
    );
    return;
  }

  await taskGenerationJobClient.update({
    where: { id: jobId },
    data: {
      status: "RUNNING",
      startedAt: new Date(),
      progress: 5,
    },
  });

  try {
    const { plan } = job;

    if (!plan) {
      console.error(
        "[task-generation] Plan not found for job:",
        jobId,
        "planId:",
        job.planId,
      );
      await taskGenerationJobClient.update({
        where: { id: jobId },
        data: {
          status: "ERROR",
          errorMessage: `Plan not found for id ${job.planId}`,
          finishedAt: new Date(),
          progress: 100,
        },
      });
      return;
    }

    const input = (job.input ?? {}) as any;
    const providerInput =
      typeof input?.provider === "string" ? input.provider : undefined;
    const modelInput =
      typeof input?.model === "string" ? input.model : undefined;

    const settings = await getLLMSettings();

    const resolvedProvider = resolveProvider(
      providerInput ?? plan.project?.aiProvider ?? settings.defaultProvider,
    );
    const resolvedModel =
      modelInput ??
      plan.project?.aiModel ??
      settings.defaultModel ??
      (resolvedProvider === "anthropic"
        ? "claude-3-5-sonnet-latest"
        : resolvedProvider === "zai"
          ? "glm-4.7"
          : "gpt-4o-mini");

    const apiKey = getProviderApiKey(resolvedProvider);
    if (!apiKey) {
      const message = `Missing API key for provider: ${resolvedProvider}`;
      console.warn("[task-generation] " + message);
      await taskGenerationJobClient.update({
        where: { id: jobId },
        data: {
          status: "ERROR",
          errorMessage: message,
          finishedAt: new Date(),
          progress: 100,
        },
      });
      return;
    }

    await taskGenerationJobClient.update({
      where: { id: jobId },
      data: { progress: 10 },
    });

    const existingTasks = await prisma.task.findMany({
      where: { planId: plan.id },
      orderBy: { createdAt: "asc" },
    });

    if (existingTasks.length > 0) {
      console.log(
        "[task-generation] Found existing tasks for plan:",
        plan.id,
        "count:",
        existingTasks.length,
      );

      await taskGenerationJobClient.update({
        where: { id: jobId },
        data: {
          status: "DONE",
          progress: 100,
          finishedAt: new Date(),
          resultSummary: {
            estimatedComplexity: plan.estimatedComplexity ?? null,
            reasoning: plan.reasoning ?? null,
            tasks: existingTasks,
          },
        },
      });

      return;
    }

    console.log(
      "[task-generation] Getting project context for plan:",
      plan.id,
    );
    const projectStateContext = await getCompactProjectContext(plan.projectId);

    const insightsQuery = `${plan.title} ${plan.techStack ?? ""}`.trim();
    const globalInsights = await searchGlobalInsights(insightsQuery, 5);

    let systemPrompt = `CRITICAL RULE FOR PROJECT INITIALIZATION: When the plan requires initializing a new Next.js or Vite frontend project, you MUST create a dedicated initialization task whose description instructs the TASK_EXECUTOR to use executeCommand to run the appropriate init command in the CURRENT directory (e.g. npx create-next-app@latest . --yes, or npm create vite@latest . -- --template react-ts). NEVER use a project name in the command (no my-app, no kanban-pwa). If the scaffolding tool creates a named subfolder, the description MUST instruct the executor to immediately move its contents to the project root (e.g. mv <subdir>/* <subdir>/.[!.]* . 2>/dev/null; rm -rf <subdir>) and then run npm install from project root. All files must end up in workspace root; do not assume cd into project-name.

Example of a good initialization task:
- title: "Initialize Next.js app skeleton"
- description: "Use executeCommand to run npx create-next-app@latest . --yes (or npm create vite@latest . -- --template react-ts for Vite) in the current directory. If a subfolder was created, move its contents to project root (mv <subdir>/* <subdir>/.[!.]* . 2>/dev/null; rm -rf <subdir>), then run npm install. Ensure package.json is in project root and the app builds."
- executorAgent: "TASK_EXECUTOR"
- verificationCriteria: { "artifacts": ["package.json"], "manualCheck": "Open localhost:3000 and see the default Next.js welcome page", "automatedCheck": "npm run build" }

Ты строгий системный архитектор. Сначала оцени сложность проекта (S, M, L, XL). Исходя из этого, определи оптимальное количество задач. НЕ создавай лишних задач для простых проектов. Лучше одна большая задача 'Реализовать Core Logic', чем 5 мелких 'Создать файл', 'Написать функцию'. Составь задачи техническим языком. ВАЖНО: Используй методологию Goal-Backward Verification. Перед созданием каждой задачи думай: как QA-агент докажет выполнение без глаз? Для каждой задачи укажи verificationCriteria: artifacts (пути к файлам), manualCheck (как проверить руками), automatedCheck (команда теста). ВАЖНО: Ты создаёшь Граф Зависимостей. Для каждой задачи укажи dependencyIndices — это индексы задач (начиная с 0), которые БЛОКИРУЮТ текущую задачу. Это массив целых чисел. Например: [0, 2] означает, что текущую задачу нельзя начать, пока не завершены задачи с индексами 0 и 2. Логика должна быть строгой: нельзя делать Фронтенд, пока не готов Бэкенд (если они связаны). Пример: Если задача 0 — 'Настройка БД', а задача 1 — 'API Аутентификации', то у задачи 1 в dependencyIndices должно быть [0]. Задачи без зависимостей имеют пустой массив dependencyIndices: []. Циклические зависимости запрещены. Зависимости должны быть логичны: API зависит от БД, Фронтенд зависит от API и т.д.

ARTIFACTS RULE:
Distinguish between two types of files:

TYPE 1 — FIXED-PATH REQUIRED FILES: config files, assets, icons, public files, env templates, manifests — anything that MUST exist at a specific path to make the feature work (e.g. public/icons/icon-192.png, vite.config.ts, .env.example, manifest.json). ALWAYS include these in artifacts.

TYPE 2 — IMPLEMENTATION FILES: components, modules, classes, handlers — files where TASK_EXECUTOR may choose different naming (e.g. LoginForm.tsx vs login-form.tsx). DO NOT include these in artifacts. Rely on automatedCheck to validate behavior instead.

Keep artifacts list focused: include ALL type 1 files mentioned or implied by the task, and NO type 2 files.

CRITICAL UAT RULE:
The VERY LAST task in your generated plan MUST ALWAYS be titled exactly "End-to-End Project Verification".
This task MUST depend on ALL other tasks in the plan.

The verificationCriteria.automatedCheck for this task MUST be chosen based on projectType:

IF projectType is "static" or "script":
  automatedCheck must verify files exist and contain expected content using shell commands only. Example:
  "test -f index.html && grep -q 'expected text' index.html"
  Do NOT use npm run build, do NOT start a server, do NOT use wget. There is no build system.

IF projectType is "frontend" (Vite):
  "npm run build && (npm run preview & sleep 5) && wget --spider -q http://localhost:4173 || exit 1"

IF projectType is "frontend" (Next.js):
  "npm run build && (npm run start & sleep 5) && wget --spider -q http://localhost:3000 || exit 1"

IF projectType is "backend":
  "npm run build && (npm run start & sleep 5) && wget --spider -q http://localhost:3000 || exit 1"
  (adjust port and start command to match the stack)

IF projectType is "fullstack":
  Use the frontend verification command.

ALWAYS use wget (not curl) for HTTP checks.
For projectType "static" or "script", the description of this final task MUST be exactly:
"Verify that all required files exist and contain the expected content."
For all other projectType values, the description of this final task MUST be exactly:
"Run the production build and verify the application works."
manualCheck MUST be exactly:
"Open the application in the browser and verify all features work together without console errors."

ВАЖНО: Отвечай ТОЛЬКО валидным JSON в следующем формате (без markdown кодов):
{
  "estimatedComplexity": "S|M|L|XL",
  "projectType": "static|frontend|backend|fullstack|script",
  "reasoning": "почему выбрана именно эта сложность и тип проекта",
  "tasks": [
    {
      "title": "заголовок задачи",
      "description": "описание задачи",
      "executorAgent": "TASK_EXECUTOR|BACKEND|DEVOPS",
      "branchName": "feat/описание",
      "verificationCriteria": {
        "artifacts": ["пути/к/файлам"],
        "manualCheck": "как проверить",
        "automatedCheck": "команда"
      },
      "dependencyIndices": [0, 1]
    }
  ]
}`;

    if (globalInsights.length > 0) {
      const lessonsBlock = globalInsights
        .map(
          (i) =>
            `- [${i.category ?? "N/A"}] ${i.title ?? "N/A"}: ${i.recommendation ?? ""}`,
        )
        .join("\n");
      systemPrompt += `\n\n### CRITICAL LESSONS LEARNED FROM PAST PROJECTS:\n${lessonsBlock}\nAlways apply these recommendations to avoid repeating past mistakes.`;
      logInfo(
        "[RAG] Injected " +
          globalInsights.length +
          " insights into Architect prompt",
      );
    }

    const prompt = `Ты — опытный Technical Team Lead. Твоя задача — декомпозировать план разработки проекта на конкретные технические задачи для MVP.

${projectStateContext}

План: ${plan.title}
Стек: ${plan.techStack}
Описание: ${plan.description ?? ""}

ШАГ 1: ОЦЕНКА СЛОЖНОСТИ
Сначала оцени сложность проекта по шкале:
- S (Small): Простой скрипт, виджет, лендинг. → 2-4 задачи
- M (Medium): MVP сервиса, бот с базой данных. → 5-8 задач
- L (Large): Сложная система, микросервисы. → 8-15 задач
- XL (Extra Large): Очень сложная система. → 15+ задач

В поле reasoning объясни, почему ты выбрал именно этот уровень сложности.

ШАГ 2: ОПРЕДЕЛЕНИЕ ТИПА ПРОЕКТА
Определи тип проекта на основе стека и описания:
- static: чистый HTML/CSS/JS без сборщика, лендинги, простые страницы
- frontend: React, Vue, Angular, Vite, Next.js, SvelteKit
- backend: Node.js API, Python Flask/FastAPI/Django, Go HTTP, Rust Actix, любой серверный фреймворк
- fullstack: проект содержит и фронтенд и бэкенд
- script: скрипт, CLI, автоматизация, cron-задача, без HTTP-сервера

Укажи выбранный тип в поле projectType.

ШАГ 3: ДЕКОМПОЗИЦИЯ С GOAL-BACKWARD VERIFICATION
Исходя из оценки сложности, создай соответствующее количество задач. НЕ создавай лишних задач для простых проектов. Лучше одна большая задача "Реализовать Core Logic", чем 5 мелких "Создать файл", "Написать функцию".

ВАЖНО: Используй методологию Goal-Backward Verification. Перед тем как создать задачу, подумай: как QA-агент сможет доказать её выполнение, не имея глаз?

Для каждой задачи:
1. Укажи исполнителя (TASK_EXECUTOR, BACKEND или DEVOPS).
2. Сгенерируй короткое имя ветки branchName в формате feat/короткое-описание или fix/короткое-описание (например, feat/setup-nextjs, fix/auth-redirect). Имена веток должны быть в lowercase, с дефисами вместо пробелов, без спецсимволов.
3. ДОБАВЬ verificationCriteria с тремя полями:
   - artifacts: МАССИВ путей к файлам, которые должны быть созданы или изменены (например, ["src/app/api/auth/route.ts", "src/components/Login.tsx"]).
   - manualCheck: Описание ручной проверки, которую может выполнить человек (например, "Open localhost:3000, click Login button, see redirect to dashboard").
   - automatedCheck: Команда для автоматизированной проверки, если применимо (например, "npm run test:api", "npm run build"). Обязательна, если есть тесты.

ШАГ 4: ГРАФ ЗАВИСИМОСТЕЙ (ОБЯЗАТЕЛЬНО!)
Ты создаёшь Граф Зависимостей. Для каждой задачи укажи dependencyIndices — это индексы задач (начиная с 0), которые БЛОКИРУЮТ текущую задачу. Это массив целых чисел. Например: [0, 2] означает, что текущую задачу нельзя начать, пока не завершены задачи с индексами 0 и 2. Логика должна быть строгой: нельзя делать Фронтенд, пока не готов Бэкенд (если они связаны). Пример: Если задача 0 — 'Настройка БД', а задача 1 — 'API Аутентификации', то у задачи 1 в dependencyIndices должно быть [0]. Задачи без зависимостей имеют пустой массив dependencyIndices: []. Циклические зависимости запрещены. Зависимости должны быть логичны: API зависит от БД, Фронтенд зависит от API и т.д.`;

    const finalPrompt = `${prompt}\n\nОтвечай ТОЛЬКО валидным JSON (без markdown кодов).`;

    await taskGenerationJobClient.update({
      where: { id: jobId },
      data: { progress: 30 },
    });

    let aiResult: any = null;
    let resultText: string;

    if (resolvedProvider === "zai") {
      resultText = await generateTextZai({
        systemPrompt,
        userMessage: finalPrompt,
        model: resolvedModel,
        temperature: settings.temperature,
        maxTokens: settings.maxTokens,
        signal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
      });
    } else {
      aiResult = await generateText({
        model: getModel(resolvedProvider, resolvedModel),
        system: systemPrompt,
        prompt: finalPrompt,
        temperature: settings.temperature,
        maxTokens: settings.maxTokens,
        abortSignal: createLLMAbortSignal(HEAVY_LLM_TIMEOUT_MS),
      });
      resultText = aiResult.text ?? "";
    }

    console.log(
      "[task-generation] AI response received, length:",
      resultText.length,
    );

    if (aiResult) {
      await trackAIUsage(aiResult, {
        projectId: plan.projectId,
        actionType: "generate_tasks",
        model: resolvedModel,
      });
    }

    await taskGenerationJobClient.update({
      where: { id: jobId },
      data: { progress: 50 },
    });

    let parsed;
    try {
      const cleanJson = resultText
        .replace(/```json/g, "")
        .replace(/```/g, "")
        .trim();
      parsed = JSON.parse(cleanJson);
    } catch (error) {
      console.error("[task-generation] Failed to parse JSON:", error);
      console.error(
        "[task-generation] Raw response:",
        resultText.slice(0, 500),
      );

      await taskGenerationJobClient.update({
        where: { id: jobId },
        data: {
          status: "ERROR",
          errorMessage:
            "Не удалось разобрать ответ модели: модель не вернула валидный JSON",
          finishedAt: new Date(),
          progress: 100,
        },
      });

      return;
    }

    const validated = taskSchema.parse(parsed);
    const { estimatedComplexity, projectType, reasoning, tasks } = validated;

    console.log(
      "[task-generation] Generated",
      tasks.length,
      "tasks, complexity:",
      estimatedComplexity,
    );

    await taskGenerationJobClient.update({
      where: { id: jobId },
      data: { progress: 60 },
    });

    await prisma.plan.update({
      where: { id: plan.id },
      data: {
        estimatedComplexity,
        projectType,
        reasoning,
      },
    });

    let taskRecords;
    try {
      taskRecords = await prisma.$transaction(async (tx) => {
        const records = await Promise.all(
          tasks.map((task: any) =>
            tx.task.create({
              data: {
                planId: plan.id,
                title: task.title,
                description: task.description,
                executorAgent: task.executorAgent as AgentRole,
                status: TaskStatus.TODO,
                observerAgent: AgentRole.TEAMLEAD,
                branchName:
                  typeof task.branchName === "string" &&
                  task.branchName.trim()
                    ? task.branchName.trim()
                    : null,
              },
            }) as any,
          ),
        );

        const taskIds = records.map((t) => t.id);

        const updatePromises: Promise<any>[] = [];
        const dependencyPromises: Promise<any>[] = [];

        tasks.forEach((task: any, index: number) => {
          const dependencyIndices = task.dependencyIndices ?? [];
          const data: any = {};

          if (task.verificationCriteria) {
            data.verificationCriteria = task.verificationCriteria;
          }

          if (Object.keys(data).length > 0) {
            updatePromises.push(
              tx.task.update({
                where: { id: taskIds[index] },
                data,
              }),
            );
          }

          if (dependencyIndices.length > 0) {
            const validDependencyIndices = dependencyIndices.filter(
              (depIndex: number) =>
                depIndex >= 0 && depIndex < taskIds.length,
            );

            validDependencyIndices.forEach((depIndex: number) => {
              dependencyPromises.push(
                tx.taskDependency.create({
                  data: {
                    taskId: taskIds[index],
                    dependsOnId: taskIds[depIndex],
                  },
                }),
              );
            });
          }
        });

        await Promise.all(updatePromises);
        await Promise.all(dependencyPromises);

        return records;
      });
    } catch (transactionError) {
      console.error(
        "[task-generation] Transaction failed while creating tasks:",
        transactionError,
      );
      throw transactionError;
    }

    await taskGenerationJobClient.update({
      where: { id: jobId },
      data: { progress: 80 },
    });

    const tasksWithDependencies = await Promise.all(
      taskRecords.map(async (task: any) => {
        const taskWithDeps = await prisma.task.findUnique({
          where: { id: task.id },
          include: {
            dependencies: {
              include: {
                dependsOn: {
                  select: { id: true, title: true, status: true },
                },
              },
            },
          },
        });
        return taskWithDeps;
      }),
    );

    try {
      const adrContent = await generateADR(plan.projectId, plan.id);
      await saveDocToClient(
        plan.projectId,
        "001-initial-architecture.md",
        adrContent,
      );
    } catch (error) {
      console.error("[task-generation] Failed to generate ADR:", error);
    }

    await taskGenerationJobClient.update({
      where: { id: jobId },
      data: {
        status: "DONE",
        progress: 100,
        finishedAt: new Date(),
        resultSummary: {
          estimatedComplexity,
          reasoning,
          tasks: tasksWithDependencies,
        },
      },
    });

    console.log("[task-generation] Job completed successfully:", jobId);
  } catch (error) {
    console.error("[task-generation] Unhandled error for job", jobId, ":", error);
    const message =
      error instanceof Error ? error.message : "Failed to generate tasks";

    const taskGenerationJobClient = (prisma as any).taskGenerationJob;
    await taskGenerationJobClient.update({
      where: { id: jobId },
      data: {
        status: "ERROR",
        errorMessage: message,
        finishedAt: new Date(),
        progress: 100,
      },
    });
  }
}


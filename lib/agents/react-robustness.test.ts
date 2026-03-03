import { describe, it, expect, vi, beforeEach } from "vitest";

import { generateText } from "ai";
import { AgentRole } from "@prisma/client";

import { TaskExecutorAgent } from "./task-executor-agent";
import { createSearchCodebaseTool, createWriteFileTool } from "./tools";
import { prisma } from "@/lib/prisma";
import * as embeddings from "@/lib/ai/embeddings";

vi.mock("ai", () => ({
  generateText: vi.fn(),
  // Simple stub so tool-based helpers can be imported without error.
  tool: vi.fn((config) => config),
}));

vi.mock("@/lib/ai/zai", () => ({
  generateTextZai: vi.fn().mockResolvedValue(""),
}));

vi.mock("@/lib/ai/providers", () => ({
  getModel: vi.fn(() => ({})),
  resolveProvider: vi.fn((p: string | undefined) => p || "openai"),
  getProviderApiKey: vi.fn(() => "fake-openai-key"),
}));

vi.mock("@/lib/ai/call", () => ({
  trackAIUsage: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./project-context", () => ({
  getCompactProjectContext: vi.fn().mockResolvedValue("PROJECT_CTX"),
}));

vi.mock("@/lib/settings", () => ({
  getLLMSettings: vi.fn().mockResolvedValue({
    temperature: 0,
    maxTokens: 1024,
  }),
}));

vi.mock("@/lib/ai/timeout", () => ({
  createLLMAbortSignal: vi.fn(() => undefined),
  HEAVY_LLM_TIMEOUT_MS: 5 * 60 * 1000,
}));

vi.mock("@/lib/prisma", () => ({
  prisma: {
    $queryRawUnsafe: vi.fn(),
    task: {
      findUnique: vi.fn(),
      update: vi.fn().mockResolvedValue(undefined),
    },
    ticket: {
      findUnique: vi.fn(),
      update: vi.fn().mockResolvedValue(undefined),
    },
    comment: {
      create: vi.fn().mockResolvedValue(undefined),
    },
    setting: {
      findMany: vi.fn().mockResolvedValue([]),
    },
    agentMessage: {
      create: vi.fn().mockResolvedValue({ id: "msg-out-1" }),
    },
    projectFile: {
      findMany: vi.fn(),
    },
    syncCommand: {
      findUnique: vi.fn(),
    },
  },
}));

vi.mock("@/lib/ai/embeddings", async () => {
  const actual =
    await vi.importActual<typeof import("@/lib/ai/embeddings")>(
      "@/lib/ai/embeddings"
    );
  return {
    ...actual,
    generateEmbedding: vi.fn(),
  };
});

describe("ReAct robustness against malformed outputs", () => {
  const projectId = "proj-1";
  const planId = "plan-1";
  const sessionId = "sess-1";

  const baseTask = {
    id: "task-1",
    title: "Robustness task",
    description: "Do something robustly",
    status: "PENDING",
    planId,
    plan: {
      id: planId,
      project: {
        id: projectId,
        aiProvider: "openai",
        aiModel: "gpt-4o-mini",
      },
    },
    verificationCriteria: null,
    attachments: [] as any[],
    dependencies: [] as any[],
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("parses nested JSON from dirty LLM output", async () => {
    const dirty =
      'Certainly! Here is your plan: { "thought": "Refactoring...", "action": { "toolName": "writeFile", "params": { "filePath": "src/app.ts", "content": "const x = 1;" } } }';

    vi.mocked(generateText).mockResolvedValue({
      text: dirty,
    } as any);

    const agent = new TaskExecutorAgent({
      projectId,
      planId,
      sessionId,
      mode: "local",
      onLog: vi.fn(),
      agentRole: AgentRole.TASK_EXECUTOR,
    } as any);

    // @ts-expect-error accessing private method for robustness testing
    const step = await (agent as any).getNextReActStep(
      "HISTORY",
      projectId,
      "openai",
      "gpt-4o-mini"
    );

    expect(step.thought).toBe("Refactoring...");
    expect(step.action).toEqual({
      toolName: "writeFile",
      params: { filePath: "src/app.ts", content: "const x = 1;" },
    });
  });

  it("searchCodebase returns [] for null/undefined/empty queries", async () => {
    vi.mocked(embeddings.generateEmbedding).mockResolvedValue(
      [0.1, 0.2, 0.3] as any
    );

    const tool = createSearchCodebaseTool(projectId);

    const values: any[] = [null, undefined, ""];

    for (const value of values) {
      const result = await (tool as any).execute({
        query: value,
        limit: 5,
      });
      expect(result).toEqual([]);
    }

    expect(embeddings.generateEmbedding).not.toHaveBeenCalled();
    expect(prisma.$queryRawUnsafe).not.toHaveBeenCalled();
  });

  it("writeFile with runExecuteCommand uses heredoc and returns success", async () => {
    const runExecuteCommand = vi.fn().mockResolvedValue({ success: true, exitCode: 0 });
    const tool = createWriteFileTool(projectId, "exec-1", runExecuteCommand);

    const result = await (tool as any).execute({
      filePath: "file.txt",
      content: "hello world",
    });

    expect(result.success).toBe(true);
    expect(result.filePath).toBe("file.txt");
    expect(runExecuteCommand).toHaveBeenCalledTimes(1);
    const heredocCmd = runExecuteCommand.mock.calls[0][0];
    expect(heredocCmd).toContain("cat <<");
    expect(heredocCmd).toContain("file.txt");
    expect(heredocCmd).toContain("hello world");
  });

  it("broken JSON triggers REJECTED state with TASK_EXECUTOR CRASH comment", async () => {
    vi.mocked(prisma.task.findUnique).mockImplementation(
      async (args: any) => {
        if (args?.where?.id === baseTask.id) {
          return baseTask as any;
        }
        return null;
      }
    );

    vi.mocked(generateText).mockResolvedValue({
      text: "Sorry, I cannot comply",
    } as any);

    const agent = new TaskExecutorAgent({
      projectId,
      planId,
      sessionId,
      mode: "local",
      onLog: vi.fn(),
      agentRole: AgentRole.TASK_EXECUTOR,
    } as any);

    const response = await agent.processMessage({
      id: "msg-1",
      eventType: "TASK_REQUEST",
      agentRole: AgentRole.TASK_EXECUTOR,
      payload: { taskId: baseTask.id },
    } as any);

    expect(prisma.task.update).toHaveBeenCalled();
    const updateArg = vi.mocked(prisma.task.update).mock.calls[0][0];
    expect(updateArg.where).toEqual({ id: baseTask.id });

    expect(prisma.comment.create).toHaveBeenCalled();
    const commentArg = vi.mocked(prisma.comment.create).mock.calls[0][0];
    expect(commentArg.data.content).toContain("IMPLEMENTATION REPORT");
  });

  it("handles JSON with trailing writeCodeBlock markdown by extracting code correctly", async () => {
    const code = "export const x = 1;";

    vi.mocked(generateText).mockResolvedValue({
      text:
        '{ "thought": "Write file via markdown", "action": { "toolName": "writeCodeBlock", "params": { "filePath": "src/app.tsx", "content": "<CODE_BELOW>" } } }\n' +
        "```tsx\n" +
        code +
        "\n```",
    } as any);

    const agent = new TaskExecutorAgent({
      projectId,
      planId,
      sessionId,
      mode: "local",
      onLog: vi.fn(),
      agentRole: AgentRole.TASK_EXECUTOR,
    } as any);

    // @ts-expect-error accessing private method for robustness testing
    const step = await (agent as any).getNextReActStep(
      "HISTORY",
      projectId,
      "openai",
      "gpt-4o-mini"
    );

    expect(step.action).toEqual({
      toolName: "writeFile",
      params: {
        filePath: "src/app.tsx",
        content: code,
      },
    });
  });

  it("sends a single CSS STYLE_REQUEST based on the last successful CSS write", async () => {
    const agent = new TaskExecutorAgent({
      projectId,
      planId,
      sessionId,
      mode: "local",
      onLog: vi.fn(),
      agentRole: AgentRole.TASK_EXECUTOR,
    } as any);

    const writeFileExecute = vi.fn().mockResolvedValue({
      success: true,
    });

    const tools: any = {
      writeFile: { execute: writeFileExecute },
    };

    const sendMessageMock = vi.fn().mockResolvedValue(undefined);
    // @ts-expect-error overriding protected method for test
    (agent as any).sendMessage = sendMessageMock;

    // Native FC path: mock generateText to return writeFile steps then FINISH
    vi.mocked(generateText)
      .mockResolvedValueOnce({
        steps: [
          {
            toolCalls: [
              {
                toolName: "writeFile",
                args: {
                  filePath: "src/styles.css",
                  content: "body { color: red; }",
                },
              },
            ],
            toolResults: [{ success: true }],
            text: "",
          },
        ],
        response: { messages: [] },
        text: "",
      } as any)
      .mockResolvedValueOnce({
        steps: [
          {
            toolCalls: [
              {
                toolName: "writeFile",
                args: {
                  filePath: "src/styles.css",
                  content: "body { color: blue; }",
                },
              },
            ],
            toolResults: [{ success: true }],
            text: "",
          },
        ],
        response: { messages: [] },
        text: "",
      } as any)
      .mockResolvedValueOnce({
        steps: [{ toolCalls: [], toolResults: [], text: "Done" }],
        response: { messages: [] },
        text: "Done",
      } as any);

    // @ts-expect-error accessing private method for robustness testing
    const result = await (agent as any).runReActLoop({
      projectId,
      projectProvider: "openai",
      projectModel: "gpt-4o-mini",
      initialTaskContext: "TASK_CTX",
      tools,
      primaryTaskId: baseTask.id,
      parentMessageId: "msg-parent-2",
    });

    expect(result).toBeTruthy();
    expect(result.lastSuccessfulCssWrite).toEqual({
      filePath: "src/styles.css",
      content: "body { color: blue; }",
    });
    expect(result.shouldEscalate).toBe(false);
    expect(typeof result.lastCompletedIteration).toBe("number");
    expect(sendMessageMock).toHaveBeenCalledTimes(0);
  });

  it("supports logical escalateToTeamLead tool in Native FC and sets shouldEscalate", async () => {
    const agent = new TaskExecutorAgent({
      projectId,
      planId,
      sessionId,
      mode: "local",
      onLog: vi.fn(),
      agentRole: AgentRole.TASK_EXECUTOR,
    } as any);

    const tools: any = {};

    vi.mocked(generateText)
      // First call: logical escalation tool
      .mockResolvedValueOnce({
        steps: [
          {
            toolCalls: [
              {
                toolName: "escalateToTeamLead",
                args: {},
              },
            ],
            toolResults: [],
            text: "",
          },
        ],
        response: { messages: [] },
        text: "",
      } as any)
      // Second call should not be reached, but provide a safe fallback
      .mockResolvedValueOnce({
        steps: [{ toolCalls: [], toolResults: [], text: "Done" }],
        response: { messages: [] },
        text: "Done",
      } as any);

    // @ts-expect-error accessing private method for robustness testing
    const result = await (agent as any).runReActLoop({
      projectId,
      projectProvider: "openai",
      projectModel: "gpt-4o-mini",
      initialTaskContext: "TASK_CTX",
      tools,
      primaryTaskId: baseTask.id,
      parentMessageId: "msg-parent-escalate",
    });

    expect(result).toBeTruthy();
    expect(result.shouldEscalate).toBe(true);
    expect(result.lastCompletedIteration).toBeGreaterThanOrEqual(1);
    expect(result.lastCompletedIteration).toBeLessThanOrEqual(30);
  });

  it("detects and blocks repeated identical actions in the ReAct loop", async () => {
    const agent = new TaskExecutorAgent({
      projectId,
      planId,
      sessionId,
      mode: "local",
      onLog: vi.fn(),
      agentRole: AgentRole.TASK_EXECUTOR,
    } as any);

    const readFileExecute = vi.fn().mockResolvedValue({
      filePath: "src/app.tsx",
      content: "// app",
      mimeType: "text/typescript",
      success: true,
    });

    const tools: any = {
      readFile: { execute: readFileExecute },
    };

    const readFileStep = {
      toolCalls: [
        { toolName: "readFile", args: { filePath: "src/app.tsx" } },
      ],
      toolResults: [
        {
          filePath: "src/app.tsx",
          content: "// app",
          mimeType: "text/typescript",
          success: true,
        },
      ],
      text: "",
    };

    // Native FC path: 3x readFile (3rd triggers repeat monitor), then FINISH
    vi.mocked(generateText)
      .mockResolvedValueOnce({
        steps: [readFileStep],
        response: { messages: [] },
        text: "",
      } as any)
      .mockResolvedValueOnce({
        steps: [readFileStep],
        response: { messages: [] },
        text: "",
      } as any)
      .mockResolvedValueOnce({
        steps: [readFileStep],
        response: { messages: [] },
        text: "",
      } as any)
      .mockResolvedValueOnce({
        steps: [{ toolCalls: [], toolResults: [], text: "Done" }],
        response: { messages: [] },
        text: "Done",
      } as any);

    // @ts-expect-error accessing private method for robustness testing
    const result = await (agent as any).runReActLoop({
      projectId,
      projectProvider: "openai",
      projectModel: "gpt-4o-mini",
      initialTaskContext: "TASK_CTX",
      tools,
      primaryTaskId: baseTask.id,
      parentMessageId: "msg-parent-1",
    });

    expect(result).toBeTruthy();
    // Repeat Action Monitor: after 3 identical readFile steps we inject warning and continue; 4th call returns FINISH
    expect(generateText).toHaveBeenCalledTimes(4);
  });
});


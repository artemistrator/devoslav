import { describe, it, expect, vi, beforeEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

vi.mock("ai", () => ({
  generateText: vi.fn(),
}));

vi.mock("@/lib/ai/zai", () => ({
  generateTextZai: vi.fn().mockResolvedValue(""),
}));

vi.mock("@/lib/ai/providers", () => ({
  getModel: vi.fn(() => ({})),
  resolveProvider: vi.fn(() => "openai"),
  getProviderApiKey: vi.fn(() => "fake-openai-key"),
}));

vi.mock("@/lib/ai/call", () => ({
  trackAIUsage: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./project-context", () => ({
  getCompactProjectContext: vi.fn().mockResolvedValue("PROJECT_CTX"),
}));

vi.mock("./tools", () => ({
  createAgentTools: vi.fn(() => ({})),
}));

vi.mock("@/lib/project-workspace", () => ({
  getProjectDir: vi.fn(),
}));

vi.mock("@/lib/prisma", () => ({
  prisma: {
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
  },
}));

import { generateText } from "ai";
import { prisma } from "@/lib/prisma";
import { getProjectDir } from "@/lib/project-workspace";
import { TaskExecutorAgent } from "./task-executor-agent";
import { AgentRole, MessageType } from "@prisma/client";

describe("TaskExecutorAgent", () => {
  const projectId = "proj-1";
  const planId = "plan-1";
  const sessionId = "sess-1";

  const baseTask = {
    id: "task-1",
    title: "Original task",
    description: "Do something",
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

  it("includes RCA ticket retry block in system prompt for ticket requests", async () => {
    vi.mocked(prisma.ticket.findUnique).mockResolvedValue({
      id: "ticket-1",
      title: "QA rejection",
      description: "Missing evidence in report",
      status: "OPEN",
    } as any);

    vi.mocked(prisma.task.findUnique).mockImplementation(async (args: any) => {
      if (args?.where?.id === "task-1") {
        return baseTask as any;
      }
      return null;
    });

    vi.mocked(generateText).mockImplementation(async (opts: any) => {
      // First call is from generateTaskPrompt (Tech Lead prompt); second is from handleTicketRequest (ticket retry).
      // Only assert on the ticket retry call (system contains CRITICAL: TICKET RETRY PROTOCOL).
      if (typeof opts?.system === "string" && opts.system.includes("CRITICAL: TICKET RETRY PROTOCOL")) {
        expect(opts.system).toContain("Your VERY FIRST element in the \"steps\" array MUST be a thought-only step");
        expect(opts.system).toContain("Do NOT use any tools until you have outputted this RCA thought");
        expect(opts.system).toContain("IMPORTANT: Keep your execution plans concise.");
      }
      // generateTaskPrompt expects generated instructions text; handleTicketRequest expects JSON steps.
      const isTicketRetry = typeof opts?.system === "string" && opts.system.includes("CRITICAL: TICKET RETRY PROTOCOL");
      return {
        text: isTicketRetry
          ? JSON.stringify({ steps: [{ thought: "RCA" }] })
          : "Step 1. Do X. Step 2. Do Y.",
      } as any;
    });

    const onLog = vi.fn();
    const agent = new TaskExecutorAgent({
      projectId,
      planId,
      sessionId,
      mode: "local",
      onLog,
      agentRole: AgentRole.TASK_EXECUTOR,
    } as any);

    // @ts-expect-error accessing private method in test
    await agent.processMessage({
      id: "msg-1",
      eventType: "TICKET_REQUEST",
      agentRole: AgentRole.TASK_EXECUTOR,
      payload: { ticketId: "ticket-1", relatedTaskId: "task-1" },
    });
  });

  it("Hard Gate appends 2>&1 and preserves the tail of compile output", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "hard-gate-test-"));
    try {
      writeFileSync(
        join(tmpDir, "package.json"),
        JSON.stringify({ scripts: { build: "npm run build-app" } })
      );
      vi.mocked(getProjectDir).mockReturnValue(tmpDir);

      const agent = new TaskExecutorAgent({
      projectId,
      planId,
      sessionId,
      mode: "local",
      onLog: vi.fn(),
      agentRole: AgentRole.TASK_EXECUTOR,
    } as any);

    const longStdout =
      "START\n" + "x".repeat(5000) + "\nEND_OF_BUILD_OUTPUT_MARKER";

    const executeCommandExecute = vi.fn().mockResolvedValue({
      success: false,
      exitCode: 1,
      stdout: longStdout,
      stderr: "",
    });

    const tools: any = {
      executeCommand: { execute: executeCommandExecute },
    };

    // @ts-expect-error accessing private method in test
    const result = await (agent as any).runVerificationAndHardGate({
      mode: "task",
      projectId,
      taskId: baseTask.id,
      verificationCriteria: null,
      tools,
      parentMessageId: "msg-parent-hard-gate",
    });

    expect(executeCommandExecute).toHaveBeenCalledTimes(1);
    const firstCallArgs = executeCommandExecute.mock.calls[0][0];
    // Hard Gate uses detectStack; nodejs with scripts.build => npm build command (normalized)
    expect(firstCallArgs.command).toBe(
      '$(which npm 2>/dev/null || echo npm) run build'
    );

    expect(result.hardGateFailed).toBe(true);
    expect(result.hardGateCompileOutput.startsWith("STOP! Your code failed the Hard Gate")).toBe(
      true
    );
    // Truncation is first 2000 + [MIDDLE TRUNCATED] + last 2000, plus STOP header
    expect(result.hardGateCompileOutput).toContain("[MIDDLE TRUNCATED]");
    expect(result.hardGateCompileOutput).toContain(
      "END_OF_BUILD_OUTPUT_MARKER"
    );
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("Hard Gate skips REJECTED on infrastructure error", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "hard-gate-test-"));
    try {
      writeFileSync(
        join(tmpDir, "package.json"),
        JSON.stringify({ scripts: { build: "npm run build-app" } })
      );
      vi.mocked(getProjectDir).mockReturnValue(tmpDir);

      const agent = new TaskExecutorAgent({
        projectId,
        planId,
        sessionId,
        mode: "local",
        onLog: vi.fn(),
        agentRole: AgentRole.TASK_EXECUTOR,
      } as any);

      const executeCommandExecute = vi.fn().mockRejectedValue(
        new Error("Command failed: docker exec container_name sh -c ...")
      );
      const tools: any = {
        executeCommand: { execute: executeCommandExecute },
      };
      const sendMessageMock = vi.fn().mockResolvedValue(undefined);
      (agent as any).sendMessage = sendMessageMock;

      const result = await (agent as any).runVerificationAndHardGate({
        mode: "task",
        projectId,
        taskId: baseTask.id,
        verificationCriteria: null,
        tools,
        parentMessageId: "msg-parent-hard-gate",
      });

      expect(result.hardGateFailed).toBe(false);
      const sendCalls = sendMessageMock.mock.calls;
      expect(
        sendCalls.some(
          (c: any[]) =>
            c[1] === MessageType.QA_RESPONSE && c[2]?.finalStatus === "REJECTED"
        )
      ).toBe(false);
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("Hard Gate sets REJECTED with BUILD EXECUTOR ERROR prefix on non-infrastructure error", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "hard-gate-test-"));
    try {
      writeFileSync(
        join(tmpDir, "package.json"),
        JSON.stringify({ scripts: { build: "npm run build-app" } })
      );
      vi.mocked(getProjectDir).mockReturnValue(tmpDir);

      const agent = new TaskExecutorAgent({
        projectId,
        planId,
        sessionId,
        mode: "local",
        onLog: vi.fn(),
        agentRole: AgentRole.TASK_EXECUTOR,
      } as any);

      const executeCommandExecute = vi.fn().mockRejectedValue(
        new Error("SyntaxError: Unexpected token in /app/src/index.ts")
      );
      const tools: any = {
        executeCommand: { execute: executeCommandExecute },
      };
      const result = await (agent as any).runVerificationAndHardGate({
        mode: "task",
        projectId,
        taskId: baseTask.id,
        verificationCriteria: {
          automatedCheck: '$(which npm 2>/dev/null || echo npm) run build',
        },
        tools,
        parentMessageId: "msg-parent-hard-gate",
      });

      expect(result.hardGateFailed).toBe(true);
      expect(
        result.hardGateCompileOutput.startsWith(
          "STOP! Your code failed the Hard Gate"
        )
      ).toBe(true);
      expect(result.hardGateCompileOutput).toContain("BUILD EXECUTOR ERROR: ");
      expect(result.hardGateCompileOutput).toContain("SyntaxError: Unexpected token");
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("Hard Gate marks failure and returns compile output when it fails", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "hard-gate-test-"));
    const callOrder: string[] = [];
    try {
      writeFileSync(
        join(tmpDir, "package.json"),
        JSON.stringify({ scripts: { build: "npm run build-app" } })
      );
      vi.mocked(getProjectDir).mockReturnValue(tmpDir);

      vi.mocked(prisma.comment.create).mockImplementation(async (arg: any) => {
        callOrder.push("comment.create");
        return {} as any;
      });

      const agent = new TaskExecutorAgent({
        projectId,
        planId,
        sessionId,
        mode: "local",
        onLog: vi.fn(),
        agentRole: AgentRole.TASK_EXECUTOR,
      } as any);

      const executeCommandExecute = vi.fn().mockResolvedValue({
        success: false,
        exitCode: 1,
        stdout: "ERROR: compilation failed",
        stderr: "",
      });
      const tools: any = {
        executeCommand: { execute: executeCommandExecute },
      };
      const result = await (agent as any).runVerificationAndHardGate({
        mode: "task",
        projectId,
        taskId: baseTask.id,
        verificationCriteria: null,
        tools,
        parentMessageId: "msg-parent-hard-gate",
      });

      expect(result.hardGateFailed).toBe(true);
      expect(
        result.hardGateCompileOutput.startsWith(
          "STOP! Your code failed the Hard Gate"
        )
      ).toBe(true);
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});


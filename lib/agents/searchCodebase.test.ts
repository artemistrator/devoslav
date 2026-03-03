import { describe, it, expect, vi, beforeEach } from "vitest";
import { prisma } from "@/lib/prisma";
import { createSearchCodebaseTool } from "./tools";
import * as embeddings from "@/lib/ai/embeddings";

vi.mock("@/lib/prisma", () => ({
  prisma: {
    $queryRawUnsafe: vi.fn(),
  },
}));

vi.mock("@/lib/ai/embeddings", async () => {
  const actual = await vi.importActual<typeof import("@/lib/ai/embeddings")>(
    "@/lib/ai/embeddings"
  );
  return {
    ...actual,
    generateEmbedding: vi.fn(),
  };
});

describe("createSearchCodebaseTool", () => {
  const projectId = "project-1";

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns mapped results from pgvector query", async () => {
    vi.mocked(embeddings.generateEmbedding).mockResolvedValue([0.1, 0.2, 0.3] as any);

    vi.mocked(prisma.$queryRawUnsafe).mockResolvedValue([
      {
        filePath: "src/app/page.tsx",
        content: "const foo = 1;",
        similarity: 0.9,
      },
      {
        filePath: "src/lib/utils.ts",
        content: "export function bar() {}",
        similarity: 0.8,
      },
    ] as any);

    const tool = createSearchCodebaseTool(projectId);
    const result = await (tool as any).execute({
      query: "utility function bar",
      limit: 5,
    });

    expect(embeddings.generateEmbedding).toHaveBeenCalledWith("utility function bar");
    expect(prisma.$queryRawUnsafe).toHaveBeenCalled();
    expect(result).toEqual([
      {
        filePath: "src/app/page.tsx",
        content: "const foo = 1;",
        similarity: 0.9,
      },
      {
        filePath: "src/lib/utils.ts",
        content: "export function bar() {}",
        similarity: 0.8,
      },
    ]);
  });

  it("fails soft when embeddings are unavailable", async () => {
    vi.mocked(embeddings.generateEmbedding).mockResolvedValue(null as any);

    const tool = createSearchCodebaseTool(projectId);
    const result = await (tool as any).execute({
      query: "something",
      limit: 5,
    });

    expect(result).toEqual([]);
    expect(prisma.$queryRawUnsafe).not.toHaveBeenCalled();
  });
});


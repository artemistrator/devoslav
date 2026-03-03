import { describe, it, expect } from "vitest";
import { enforcePromptBudget, MAX_PROMPT_CHARS } from "./prompt-generator";

describe("prompt-generator enforcePromptBudget", () => {
  const system = "SYSTEM\n";

  it("keeps system and instructions intact and trims project state when over budget", () => {
    const baseProjectState = "HEADER\n" + "RAG Insights...\n".repeat(1000);
    const projectStateBlock = baseProjectState;
    const taskBlock = "TASK\n";
    const instructionsBlock = "INSTRUCTIONS\n";

    const result = enforcePromptBudget(
      system,
      projectStateBlock,
      taskBlock,
      instructionsBlock
    );

    const totalLength =
      result.system.length +
      result.projectStateBlock.length +
      result.taskBlock.length +
      result.instructionsBlock.length;

    expect(totalLength).toBeLessThanOrEqual(MAX_PROMPT_CHARS);
    expect(result.system).toBe(system);
    expect(result.instructionsBlock).toContain("INSTRUCTIONS");
  });

  it("drops project state completely and annotates instructions when no budget remains", () => {
    // Simulate extreme overflow by crafting very long system + instructions relative to limit.
    const longSystem = "S".repeat(MAX_PROMPT_CHARS * 2);
    const projectStateBlock = "PROJECT\n".repeat(50);
    const taskBlock = "TASK\n".repeat(50);
    const instructionsBlock = "INSTRUCTIONS\n";

    const result = enforcePromptBudget(
      longSystem,
      projectStateBlock,
      taskBlock,
      instructionsBlock
    );

    expect(result.projectStateBlock.length).toBe(0);
    expect(result.instructionsBlock).toContain(
      "[Note: Project state was truncated due to context limits.]"
    );
  });
});


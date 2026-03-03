import fs from "fs/promises";
import path from "path";
import { parse } from "yaml";
import type { VibeConfig } from "./types";
import { getProjectDir } from "@/lib/project-workspace";

export async function loadProjectVibe(projectId: string): Promise<VibeConfig | null> {
  try {
    const projectDir = getProjectDir(projectId);
    const yamlPath = path.join(projectDir, "vibe.yaml");
    const jsonPath = path.join(projectDir, "vibe.json");

    try {
      await fs.access(yamlPath);
      const fileContent = await fs.readFile(yamlPath, "utf8");
      const parsed = parse(fileContent) as unknown;
      if (!parsed || typeof parsed !== "object") {
        return null;
      }
      return parsed as VibeConfig;
    } catch {
      await fs.access(jsonPath);
      const fileContent = await fs.readFile(jsonPath, "utf8");
      return JSON.parse(fileContent) as VibeConfig;
    }
  } catch {
    // If neither exists or parsing fails, return null safely. DO NOT THROW.
    return null;
  }
}

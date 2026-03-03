import fs from "fs/promises";
import path from "path";
import {
  ensureContainer,
  executeInContainer,
  destroyContainer,
} from "../lib/execution/container-manager";

async function main() {
  const sessionId = "test-session-" + Date.now();
  const hostProjectDir = "/tmp/ai-orch-test-dir-" + Date.now();

  console.log("[Test] Using sessionId=", sessionId);
  console.log("[Test] Host project dir=", hostProjectDir);

  await fs.mkdir(hostProjectDir, { recursive: true });

  const dummyPackageJson = {
    name: "ai-orch-dummy-project",
    version: "1.0.0",
    description: "Dummy project for ContainerManager E2E test",
  };

  await fs.writeFile(
    path.join(hostProjectDir, "package.json"),
    JSON.stringify(dummyPackageJson, null, 2),
    "utf8"
  );

  try {
    console.log("[Test] Ensuring container...");
    await ensureContainer(sessionId, hostProjectDir, "test-container-manager");

    console.log("[Test] Listing /app/project inside container...");
    const lsResult = await executeInContainer(sessionId, "ls -la /app/project");
    console.log("[Test] ls exitCode=", lsResult.exitCode);
    console.log("[Test] ls stdout=\n" + lsResult.stdout);
    console.log("[Test] ls stderr=\n" + lsResult.stderr);

    console.log("[Test] Checking node -v inside container...");
    const nodeResult = await executeInContainer(sessionId, "node -v");
    console.log("[Test] node -v exitCode=", nodeResult.exitCode);
    console.log("[Test] node -v stdout=\n" + nodeResult.stdout);
    console.log("[Test] node -v stderr=\n" + nodeResult.stderr);

    console.log("[Test] Destroying container...");
    await destroyContainer(sessionId);
  } finally {
    console.log("[Test] Cleaning up host temp dir...");
    await fs.rm(hostProjectDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error("[Test] Unhandled error in test-container-manager:", err);
  process.exit(1);
});


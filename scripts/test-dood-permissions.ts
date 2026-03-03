import {
  ensureContainer,
  executeInContainer,
  destroyContainer,
} from "../lib/execution/container-manager";

async function main() {
  const sessionId = `dood-test-${Date.now()}`;
  // This simulates the internal path the Orchestrator would pass in DooD mode
  const orchestratorProjectPath = "/app/projects/test-dood-project";

  console.log("[DooD Test] sessionId =", sessionId);
  console.log(
    "[DooD Test] Orchestrator project path (passed to ensureContainer) =",
    orchestratorProjectPath
  );

  try {
    console.log("[DooD Test] Ensuring container...");
    await ensureContainer(sessionId, orchestratorProjectPath, "test-dood-project");

    console.log(
      '[DooD Test] Writing test file inside container: echo "hello DooD" > /app/project/dood-test.txt'
    );
    const writeResult = await executeInContainer(
      sessionId,
      'echo "hello DooD" > /app/project/dood-test.txt'
    );
    console.log("[DooD Test] write exitCode =", writeResult.exitCode);
    if (writeResult.stdout) {
      console.log("[DooD Test] write stdout:\n" + writeResult.stdout);
    }
    if (writeResult.stderr) {
      console.log("[DooD Test] write stderr:\n" + writeResult.stderr);
    }

    console.log(
      "[DooD Test] Checking permissions of /app/project/dood-test.txt inside container..."
    );
    const lsResult = await executeInContainer(
      sessionId,
      "ls -l /app/project/dood-test.txt"
    );

    console.log("[DooD Test] ls exitCode =", lsResult.exitCode);
    console.log("[DooD Test] ls stdout:\n" + lsResult.stdout);
    console.log("[DooD Test] ls stderr:\n" + lsResult.stderr);

    console.log(
      "[DooD Test] NOTE: Visually confirm that the owner/group of dood-test.txt"
    );
    console.log(
      "[DooD Test]       matches the HOST_UID/HOST_GID you passed to Docker,"
    );
    console.log(
      "[DooD Test]       and is not root (unless your host user is root)."
    );
  } finally {
    console.log("[DooD Test] Destroying container...");
    await destroyContainer(sessionId);
    console.log("[DooD Test] Done.");
  }
}

main().catch((err) => {
  console.error("[DooD Test] Unhandled error in test-dood-permissions:", err);
  process.exit(1);
});


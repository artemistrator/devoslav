/**
 * Next.js instrumentation hook. Runs once when the Node process starts.
 * Used to clean up orphaned session containers from previous runs.
 */
export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    try {
      const { cleanupOrphanedContainers } = await import(
        "@/lib/execution/container-manager"
      );
      await cleanupOrphanedContainers();
    } catch (error) {
      console.error(
        "[instrumentation] Failed to run cleanupOrphanedContainers (non-fatal):",
        error
      );
    }
  }
}

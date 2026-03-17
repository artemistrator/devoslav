![Devoslav dashboard](https://github.com/artemistrator/devoslav/blob/main/docs/logo-w.png) 

# Devoslav

Devoslav is an experimental AI-assisted software orchestration workbench built with Next.js, Prisma, and a multi-agent execution loop. It takes a product idea, generates a small set of implementation plans, decomposes a selected plan into tasks, and runs those tasks against a real project workspace with logging, QA, retries, and export flows. The orchestration core is already substantial; broad open-ended app generation is still being stabilized.

## What Is Already Implemented

- Project creation from a free-form idea, including "new project" and "evolve an existing project" flows.
- LLM-based plan generation with three candidate plans and stored plan metadata.
- Async task-generation jobs that normalize planner output and create executable task graphs.
- A default Agent Hive Protocol (AHP) dispatcher plus a legacy execution path.
- Agent roles for task execution, QA, team lead coordination, CSS specialization, and architectural review hooks.
- Task validation, deterministic verification criteria, QA rejection handling, and ticket-based retries.
- On-disk project workspaces under `projects/<projectId>`, including `.orchestrator` helper files.
- File sync into the database, basic code parsing, embeddings/chunking, and searchable project/global insight storage.
- Session logs, cost tracking, pause/resume/stop controls, exports, and a web UI for projects, plans, tasks, and execution state.
- A Vitest regression suite around planning, prompt generation, QA, sync, execution routing, and provider connectivity.

## Architecture / Pipeline

1. A user submits an idea in the web UI.
2. `app/api/decompose-idea/route.ts` calls an LLM to generate three plan options and stores them in PostgreSQL.
3. A selected plan is expanded by `app/api/generate-tasks/route.ts` and `lib/task-generation/job-runner.ts` into normalized tasks with dependencies and verification contracts.
4. `app/api/execution-sessions/start/route.ts` validates the plan and starts an execution session.
5. The default worker is `app/api/execution-sessions/run-ahp/route.ts`, which acts as a dispatcher: it finds the next runnable task or retry ticket, posts agent messages, and coordinates agent runs.
6. `lib/agents/task-executor-agent.ts` performs the task in the project workspace using read/write/command/search tools and a generated task brief from `lib/agents/prompt-generator.ts`.
7. `lib/agents/qa.ts` reviews the executor report using deterministic gates plus LLM judgment, updates task state, and can trigger retries through tickets.
8. `lib/agents/team-lead-agent.ts` reacts to QA results, creates retry tickets, updates `PROJECT_STATE.md`, and keeps the loop moving.
9. Workspace files can be synced into the database and indexed by the lightweight RAG layer in `lib/rag/*`.
10. Completed or partial outputs can be exported as source, build artifacts, or full project ZIP archives.

## Main Modules

- `app/`: Next.js App Router pages and API routes.
- `components/`: UI for projects, plans, execution sessions, graphs, tickets, billing, and logs.
- `lib/agents/`: executor, QA, team lead, prompt generation, plan validation, architect review, reflexologist, and tooling.
- `lib/task-generation/`: async task-generation job runner and normalization/error handling.
- `lib/execution/`: dispatcher support, session manager, message bus, container manager, logging, and run guards.
- `lib/rag/`: chunking, embeddings, code parsing, and similarity search support.
- `lib/sync/` and `app/api/sync/*`: file sync and command approval/execution bridge.
- `lib/project/` and `lib/project-workspace.ts`: workspace initialization and project directory handling.
- `prisma/`: database schema and migrations.
- `public/sync-client.js` and `public/sync-init.js`: local helper scripts copied into project workspaces.

## What It Can Do Today

- Generate architecture/stack options for a new idea.
- Clone an existing project into a new "evolution" project and plan further work against it.
- Turn a selected plan into structured tasks with dependencies, task types, and verification criteria.
- Run execution sessions that process tasks and retry tickets through an agent loop.
- Track logs, costs, approvals, and session state in the UI and database.
- Keep project files on disk and sync them back into searchable database records.
- Export project outputs as `full`, `source`, or `build` ZIP bundles.
- Expose a basic MCP-style task endpoint and an IDE/sync bridge for external tooling.

## Current Version Limitations

- This repository is still experimental. The code shows active stabilization work around task decomposition, prompt generation, QA gating, and retry behavior.
- The orchestration core is stronger than the breadth of supported outputs. Narrow, well-understood flows are more reliable than arbitrary "build anything" requests.
- Universal application generation is not yet consistently stable across all stacks and project shapes. The test suite contains many targeted regressions and guardrails, which is useful, but it also reflects ongoing hardening work.
- Two execution paths coexist: the default AHP dispatcher and a legacy runner. That improves backward compatibility, but it also increases operational complexity.
- Full execution depends on external model providers and, for containerized/cloud-style task runs, Docker access from the app process.
- Some prompts, comments, and UI/log strings are still mixed-language and developer-oriented.
- The MCP layer is narrow today; it is closer to a task/status bridge than a full general-purpose MCP server.
- The RAG/indexing layer is lightweight. It stores files, chunks, embeddings, and parsed entities, but it is not a full code intelligence platform.

##Screenshots
Main page
![Devoslav dashboard](https://github.com/artemistrator/devoslav/blob/main/docs/Group%201%20(2).png) 
Plan page
![Devoslav dashboard](https://github.com/artemistrator/devoslav/blob/main/docs/Group%202%20(3).png) 
Tasks and execute page
![Devoslav dashboard](https://github.com/artemistrator/devoslav/blob/main/docs/Group%203.png) 

## How To Run Locally

### Option A: Docker Compose

Recommended if you want the app, PostgreSQL/pgvector, and the current container-oriented execution environment together.

1. Create a `.env` file with at least one provider key such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, or `ZAI_API_KEY`.
2. Start the stack:

```bash
docker-compose up --build
```

3. Open:

- App: `http://localhost:13002`
- Adminer: `http://localhost:4080`
- Beszel: `http://localhost:13080`

Notes:

- The compose stack provisions PostgreSQL with pgvector and runs `prisma db push` automatically.
- The app container mounts the Docker socket because the execution layer can create per-session containers.

### Option B: Local Node.js + database

1. Install dependencies:

```bash
npm install
```

2. Provide a PostgreSQL database in `DATABASE_URL`.
3. Add at least one LLM provider key to `.env`.
4. Sync Prisma schema and generate the client:

```bash
npx prisma db push
npx prisma generate
```

5. Start the app:

```bash
npm run dev
```

Important environment expectations:

- `DATABASE_URL` is required.
- At least one provider key is required for LLM features.
- Optional defaults can be controlled through settings or environment values such as `AI_PROVIDER` and `AI_MODEL`.
- If you want container-backed execution, the process must be able to talk to Docker.

## How To Run Tests

Run the full test suite:

```bash
npm test
```

Useful targeted commands:

```bash
npx vitest run lib/task-generation/job-runner.test.ts
npx vitest run lib/agents/task-executor-agent.test.ts
npx vitest run lib/agents/prompt-generator.test.ts
npx vitest run lib/agents/qa.test.ts
npx tsc --noEmit
```

The repository currently contains around 30 Vitest files covering the planner, executor loop, QA, sync routes, provider connectivity, workspace initialization, and related regressions.

## Project Status / Roadmap

Current status:

- Active experimental orchestration platform.
- Core flows for planning, task normalization, execution sessions, QA, retries, and workspace/export management are present.
- The codebase is being hardened through regression tests and deterministic gates rather than broad feature expansion.

Likely next steps, based on the current architecture:

- Continue stabilizing task decomposition and executor briefing.
- Reduce overlap between the legacy runner and the AHP dispatcher.
- Tighten deterministic verification and retry semantics across more project types.
- Harden workspace sync, shadow workspace handling, and export paths.
- Improve operator-facing UX and reduce mixed-language/debug-heavy surfaces.

## Possible Product Forks

The current codebase is broad enough that it could later split into narrower tools if that becomes desirable. The most natural candidates, based on existing modules, are:

- a planning/decomposition tool,
- a bug-fix and retry orchestration loop,
- a QA/verification assistant,
- a test-generation workflow around planner-authored test files.

That said, this repository today is best understood as an orchestration core and experimentation platform, not a polished single-purpose product.

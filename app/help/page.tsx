import Link from "next/link";
import {
  Lightbulb,
  Layout,
  ListTodo,
  Play,
  Shield,
  MessageSquare,
  FileText,
  GitBranch,
  Zap,
  Eye,
  FolderOpen,
  CheckSquare,
  Brain,
  Terminal,
  Wallet,
  HelpCircle,
  ChevronRight,
} from "lucide-react";

export const metadata = {
  title: "Help — devoslav",
  description: "Guide for using devoslav",
};

export default function HelpPage() {
  return (
    <div className="mx-auto max-w-4xl px-6 py-12">
      <div className="mb-12">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-50">
          devoslav help
        </h1>
        <p className="mt-2 text-slate-600 dark:text-slate-400">
          Everything you need to know to work with the service.
        </p>
      </div>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <Lightbulb className="h-5 w-5 text-amber-500" />
          What the service does
        </h2>
        <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
          devoslav is a project management system powered by AI agents. You describe your project idea
          in natural language, and the service turns it into structured plans and tasks. AI agents can
          automatically execute tasks, while the QA agent verifies the results. The service learns from
          past projects and takes global context into account when working.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <Layout className="h-5 w-5 text-blue-500" />
          Main workflow
        </h2>
        <ol className="space-y-4">
          <li className="flex gap-3 rounded-lg border border-slate-200 bg-slate-50/50 p-4 dark:border-slate-800 dark:bg-slate-900/30">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 text-sm font-bold text-blue-700 dark:bg-blue-900/50 dark:text-blue-300">
              1
            </span>
            <div>
              <strong>Home page</strong> — enter your project idea (what you want to build).
              Click “Generate plans” to get 3 architectural options with different technologies.
            </div>
          </li>
          <li className="flex gap-3 rounded-lg border border-slate-200 bg-slate-50/50 p-4 dark:border-slate-800 dark:bg-slate-900/30">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 text-sm font-bold text-blue-700 dark:bg-blue-900/50 dark:text-blue-300">
              2
            </span>
            <div>
              <strong>Project page</strong> — select one of the plans and click “Open tasks”
              or “Generate tasks” if there are none yet. Tasks will appear on a kanban board with dependencies.
            </div>
          </li>
          <li className="flex gap-3 rounded-lg border border-slate-200 bg-slate-50/50 p-4 dark:border-slate-800 dark:bg-slate-900/30">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 text-sm font-bold text-blue-700 dark:bg-blue-900/50 dark:text-blue-300">
              3
            </span>
            <div>
              <strong>Plan page</strong> — work with tasks: click them to see details,
              switch between Plan and Execute modes, and start auto-execution.
            </div>
          </li>
        </ol>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <ListTodo className="h-5 w-5 text-emerald-500" />
          Kanban and task statuses
        </h2>
        <p className="mb-4 text-slate-600 dark:text-slate-300">
          Tasks move through the kanban columns:
        </p>
        <div className="grid gap-2 sm:grid-cols-2">
            {[
            { status: "To Do", desc: "The task is waiting to be started." },
            { status: "In Progress", desc: "The task is currently being worked on." },
            { status: "In Review", desc: "The result is under review." },
            { status: "Waiting Approval", desc: "Awaiting your approval." },
            { status: "Done", desc: "The task is completed." },
          ].map((item, i) => (
            <div
              key={i}
              className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900"
            >
              <span className="font-medium text-slate-900 dark:text-slate-100">{item.status}</span>
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <Play className="h-5 w-5 text-violet-500" />
          Plan and Execute modes
        </h2>
        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            <div className="flex items-center gap-2 font-medium text-slate-900 dark:text-slate-100">
              <Eye className="h-4 w-4" />
              Plan
            </div>
            <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
              Planning and review mode. On the left — file navigation (Workspace), in the center —
              the task list (Kanban) or dependency graph (Graph). Clicking a task opens its card
              in a side panel where you can study the plan and edit tasks.
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            <div className="flex items-center gap-2 font-medium text-slate-900 dark:text-slate-100">
              <Zap className="h-4 w-4" />
              Execute
            </div>
            <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
              Execution mode. On the left — Workspace (project files) and Tasks (by status).
              In the center — the execution console with logs and AI chat. On the right — the selected task panel
              with Details, Agent and Output tabs. This is where auto‑execution and communication with the agent happen.
            </p>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <FolderOpen className="h-4 w-4 text-amber-500" />
          Navigation buttons (left sidebar)
        </h2>
        <ul className="space-y-2">
          <li className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2 dark:border-slate-800 dark:bg-slate-900">
            <FolderOpen className="h-4 w-4 text-slate-500" />
            <div>
              <strong>Workspace</strong> — project file tree. Browse contents and navigate the structure.
            </div>
          </li>
          <li className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2 dark:border-slate-800 dark:bg-slate-900">
            <CheckSquare className="h-4 w-4 text-slate-500" />
            <div>
              <strong>Tasks</strong> — list of tasks grouped by status. Available only in Execute mode.
            </div>
          </li>
        </ul>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <FileText className="h-4 w-4 text-cyan-500" />
          Task panel tabs (right)
        </h2>
        <ul className="space-y-2">
          <li className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2 dark:border-slate-800 dark:bg-slate-900">
            <FileText className="h-4 w-4 text-slate-500" />
            <div>
              <strong>Details</strong> — description, status, assignee, branch, prompt, dependencies.
            </div>
          </li>
          <li className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2 dark:border-slate-800 dark:bg-slate-900">
            <Brain className="h-4 w-4 text-slate-500" />
            <div>
              <strong>Agent</strong> — conversation with the AI agent about the task.
            </div>
          </li>
          <li className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2 dark:border-slate-800 dark:bg-slate-900">
            <Terminal className="h-4 w-4 text-slate-500" />
            <div>
              <strong>Output</strong> — logs from QA verification of the task.
            </div>
          </li>
        </ul>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <GitBranch className="h-4 w-4 text-green-500" />
          Task executors (agents)
        </h2>
        <p className="mb-4 text-slate-600 dark:text-slate-300">
          Each task can have an executor — the agent type that will work on it:
        </p>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {[
            { role: "TASK_EXECUTOR", name: "Frontend", desc: "Frontend, UI, styling." },
            { role: "BACKEND", name: "Backend", desc: "APIs and server-side logic." },
            { role: "DEVOPS", name: "DevOps", desc: "Infrastructure and deployment." },
            { role: "TEAMLEAD", name: "Teamlead", desc: "Architecture and coordination." },
            { role: "CURSOR", name: "Cursor", desc: "Integration with Cursor IDE." },
            { role: "QA", name: "QA", desc: "Quality assurance and verification." },
          ].map((a) => (
            <div
              key={a.role}
              className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900"
            >
              <span className="font-mono text-xs font-medium text-slate-700 dark:text-slate-300">
                {a.name}
              </span>
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{a.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <Zap className="h-4 w-4 text-amber-500" />
          Auto execution
        </h2>
        <p className="mb-4 text-slate-600 dark:text-slate-300">
          In Execute mode, click the start button (green button with a Play icon) next to the Plan/Execute toggle.
          A modal window will open where you can:
        </p>
        <ul className="list-inside list-disc space-y-2 text-slate-600 dark:text-slate-400">
          <li>Enable <strong>Auto-approve</strong> — commands execute without manual confirmation.</li>
          <li>Set a <strong>Cost limit</strong> — the maximum spend per session in dollars.</li>
          <li>Select <strong>Local</strong> or <strong>Cloud</strong> execution mode.</li>
        </ul>
        <p className="mt-4 text-slate-600 dark:text-slate-300">
          After starting, the Execution Agent automatically picks up tasks, generates prompts, runs commands,
          and sends results for QA review. The console shows logs and the AI chat.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <MessageSquare className="h-5 w-5 text-indigo-500" />
          Project Context
        </h2>
        <p className="mb-4 text-slate-600 dark:text-slate-300">
          On the project page, the <strong>Project Context</strong> button opens a panel where you can:
        </p>
        <ul className="list-inside list-disc space-y-2 text-slate-600 dark:text-slate-400">
          <li>Write text context — rules, requirements, and project specifics. The AI uses this when generating prompts and answers.</li>
          <li>Specify a GitHub repository — a link to the project repo for analysis.</li>
          <li>Upload files — additional documents for context.</li>
          <li>Toggle <strong>Require Approval</strong> — require manual approval before a task moves to DONE.</li>
        </ul>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <Shield className="h-5 w-5 text-emerald-500" />
          QA verification
        </h2>
        <p className="text-slate-600 dark:text-slate-300">
          The QA agent reviews task execution reports. It requires concrete evidence:
          test logs, build output, and paths to created files. Tasks without sufficient evidence are rejected
          with explanations. The agent uses web search and project file reading to perform checks.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <Wallet className="h-5 w-5 text-slate-500" />
          Billing and usage
        </h2>
        <p className="text-slate-600 dark:text-slate-300">
          The header and home page show a spending indicator. On the project page, under
          the “Project idea” heading, there is a card with total cost and token usage.
          You can expand it to see breakdowns by model, actions, and usage history.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-slate-900 dark:text-slate-100">
          <HelpCircle className="h-5 w-5 text-slate-500" />
          Additional features
        </h2>
        <ul className="space-y-2">
          <li><strong>Graph</strong> — the List/Graph toggle shows tasks as a dependency graph.</li>
          <li><strong>Task card</strong> — clicking a task on the kanban or graph opens a full card with editing, agent dialog, and prompt generation.</li>
          <li><strong>Sync Status</strong> — indicator of sync-client connection for file sync and command execution.</li>
          <li><strong>Copy ID</strong> — buttons to copy project and task IDs to the clipboard.</li>
        </ul>
      </section>

        <div className="mt-12 rounded-lg border border-slate-200 bg-slate-50 p-6 dark:border-slate-800 dark:bg-slate-900/50">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-slate-700 hover:text-slate-900 dark:text-slate-300 dark:hover:text-slate-100"
        >
          <ChevronRight className="h-4 w-4 rotate-180" />
          Back to home
        </Link>
      </div>
    </div>
  );
}

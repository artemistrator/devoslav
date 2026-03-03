"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { shortErrorDescription } from "@/lib/error-message";

export function GenerateTasksButton({
  planId,
  projectId,
  label = "Generate Tasks & Start",
  onGenerated,
  onGenerateStart,
  disabled,
  isGenerating: controlledGenerating,
}: {
  planId: string;
  projectId: string;
  label?: string;
  /** If set, do not navigate; call this after success so parent can e.g. slide to tasks view */
  onGenerated?: () => void;
  /** Called when generate is starting (so parent can show loading during delay + slide) */
  onGenerateStart?: () => void;
  disabled?: boolean;
  /** When using onGenerated, parent can control loading so it can show loading during slide delay */
  isGenerating?: boolean;
}) {
  const [internalGenerating, setInternalGenerating] = useState(false);
  const { toast } = useToast();
  const router = useRouter();

  const isGenerating = controlledGenerating ?? internalGenerating;

  const POLL_INTERVAL_MS = 2000;
  const MAX_POLLS = 180; // ~6 минут ожидания

  async function handleClick() {
    if (isGenerating || disabled) {
      return;
    }

    onGenerateStart?.();
    if (controlledGenerating === undefined) {
      setInternalGenerating(true);
    }

    try {
      const response = await fetch("/api/generate-tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ planId }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const desc = shortErrorDescription(null, {
          status: response.status,
          serverMessage: payload?.error,
        });
        throw new Error(desc);
      }

      const payload = await response.json().catch(() => ({}));
      const jobId: string | undefined =
        typeof payload?.jobId === "string" ? payload.jobId : undefined;

      if (!jobId) {
        throw new Error("Сервер вернул некорректный ответ (нет jobId).");
      }

      // Polling статуса генерации задач
      let attempt = 0;
      let lastStatus: string | undefined;
      // eslint-disable-next-line no-constant-condition
      while (true) {
        if (attempt > 0) {
          await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
        }

        attempt++;
        if (attempt > MAX_POLLS) {
          throw new Error(
            "Генерация задач занимает слишком много времени. Попробуйте позже или проверьте логи сервера.",
          );
        }

        const statusResponse = await fetch(`/api/generate-tasks/${jobId}`);
        if (!statusResponse.ok) {
          const statusPayload = await statusResponse.json().catch(() => ({}));
          const desc = shortErrorDescription(null, {
            status: statusResponse.status,
            serverMessage: statusPayload?.error,
          });
          throw new Error(desc);
        }

        const statusPayload = await statusResponse.json().catch(() => ({}));
        const status: string | undefined = statusPayload?.status;
        lastStatus = status;

        if (status === "DONE") {
          break;
        }

        if (status === "ERROR" || status === "CANCELLED") {
          const errMsg: string =
            typeof statusPayload?.errorMessage === "string" &&
            statusPayload.errorMessage.trim()
              ? statusPayload.errorMessage
              : "Генерация задач завершилась с ошибкой.";
          throw new Error(errMsg);
        }
      }

      // К этому моменту статус job = DONE, задачи уже в БД
      if (onGenerated) {
        onGenerated();
        // Parent controls loading; do not set internal false
      } else {
        router.push(`/project/${projectId}/plan/${planId}`);
      }
    } catch (error) {
      const desc = shortErrorDescription(error);
      toast({
        variant: "destructive",
        title: "Не удалось сгенерировать задачи",
        description: desc,
      });
      setInternalGenerating(false);
    } finally {
      if (!onGenerated) {
        setInternalGenerating(false);
      }
    }
  }

  return (
    <Button
      size="sm"
      onClick={handleClick}
      disabled={disabled ?? isGenerating}
      className="h-9 bg-[var(--pp-accent)] text-[var(--pp-bg)] hover:opacity-90"
    >
      {isGenerating ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Генерация...
        </>
      ) : (
        label
      )}
    </Button>
  );
}

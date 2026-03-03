/** Короткая расшифровка ошибки для уведомления (не длинная). */
export function shortErrorDescription(
  error: unknown,
  options?: { status?: number; serverMessage?: string }
): string {
  const msg = error instanceof Error ? error.message : String(error ?? "Ошибка");
  const status = options?.status;
  const server = options?.serverMessage;

  if (status !== undefined) {
    if (status === 404) return "Не найдено (404).";
    if (status === 422) return server && server.length < 120 ? server : "Сервер отклонил запрос (422).";
    if (status === 500) {
      const hint = server?.trim();
      if (hint && hint.length <= 90) return `Ошибка сервера (500): ${hint}`;
      if (hint) return `Ошибка сервера (500): ${hint.slice(0, 87)}…`;
      return "Ошибка сервера (500).";
    }
    if (status === 502 || status === 503) return "Сервер временно недоступен.";
    if (status >= 400) return server && server.length < 80 ? server : `Ошибка запроса (${status}).`;
  }

  if (msg === "Failed to fetch" || msg.includes("NetworkError") || msg.includes("Load failed")) {
    return "Сеть недоступна или сервер не отвечает.";
  }
  if (msg.includes("abort") || msg.includes("timeout") || msg.includes("Timeout")) {
    return "Превышено время ожидания.";
  }

  return msg.length > 100 ? msg.slice(0, 97) + "…" : msg;
}

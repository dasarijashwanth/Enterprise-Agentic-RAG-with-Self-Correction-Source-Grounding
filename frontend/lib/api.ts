/** API client for the VeriRAG FastAPI backend (Day 6).
 *
 * Matches the backend contract:
 *  - POST /api/v1/ask/stream -> text/event-stream with events:
 *      event: trace   data: {"node": "<name>"}
 *      event: answer   data: {AskResponse JSON}
 *      event: done     data: {}
 *  - POST /api/v1/feedback  {query, answer, rating: "up"|"down", comment?}
 *  - POST /api/v1/ingest    multipart form, .pdf file
 */

export interface Citation {
  chunk_id: string;
  page: number;
  bbox: number[];
  text: string;
}

export interface AskResponse {
  question: string;
  answer: string;
  citations: Citation[];
  grounded: boolean;
  rewrites: number;
  trace: string[];
}

export interface IngestResponse {
  filename: string;
  pages: number;
  words: number;
  chunks: number;
  empty_pages: number[];
  index_path: string;
  backend: string;
  embedding_model: string;
}

export function apiBase(): string {
  return (
    process.env.NEXT_PUBLIC_API_URL ||
    (typeof window !== "undefined" ? "http://localhost:8000" : "http://localhost:8000")
  );
}

function apiKeyHeader(): Record<string, string> {
  const key = process.env.NEXT_PUBLIC_VERIRAG_API_KEY;
  return key ? { "X-API-Key": key } : {};
}

export async function streamAsk(
  question: string,
  topK: number,
  onTrace: (node: string) => void,
  signal: AbortSignal
): Promise<AskResponse> {
  const res = await fetch(`${apiBase()}/api/v1/ask/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...apiKeyHeader(),
    },
    body: JSON.stringify({ question, top_k: topK }),
    signal,
  });

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(`Ask failed (${res.status}): ${text || res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer: AskResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      let event = "";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (event === "trace") {
        try {
          onTrace((JSON.parse(data) as { node: string }).node);
        } catch {
          /* ignore malformed trace frame */
        }
      } else if (event === "answer") {
        answer = JSON.parse(data) as AskResponse;
      }
    }
  }

  if (!answer) throw new Error("Stream ended without an answer event");
  return answer;
}

export async function sendFeedback(
  query: string,
  answerText: string,
  rating: "up" | "down",
  comment?: string
): Promise<void> {
  const res = await fetch(`${apiBase()}/api/v1/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...apiKeyHeader() },
    body: JSON.stringify({ query, answer: answerText, rating, comment: comment ?? null }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Feedback failed (${res.status}): ${text || res.statusText}`);
  }
}

export async function uploadPdf(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await fetch(`${apiBase()}/api/v1/ingest`, {
    method: "POST",
    headers: { ...apiKeyHeader() },
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Ingest failed (${res.status}): ${text || res.statusText}`);
  }
  return (await res.json()) as IngestResponse;
}

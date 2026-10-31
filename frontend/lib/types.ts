import type { Citation } from "../lib/api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  question: string;
  answer?: string;
  citations?: Citation[];
  grounded?: boolean;
  rewrites?: number;
  trace?: string[];
  liveNodes?: string[];
  streaming?: boolean;
  error?: string;
  feedback?: "up" | "down" | null;
  feedbackSent?: boolean;
}
export function uid(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

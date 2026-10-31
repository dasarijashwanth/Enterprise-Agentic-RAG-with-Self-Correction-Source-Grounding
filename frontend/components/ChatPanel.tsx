"use client";

import { useState } from "react";
import type { ChatMessage } from "../lib/types";
import { streamAsk } from "../lib/api";
import AnswerCard from "./AnswerCard";

let nextId = 1;
function uid(): string {
  return `msg-${Date.now()}-${nextId++}`;
}

export default function ChatPanel({
  messages,
  setMessages,
  onNewCitations,
}: {
  messages: ChatMessage[];
  setMessages: (updater: (prev: ChatMessage[]) => ChatMessage[]) => void;
  onNewCitations: (question: string) => void;
}) {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = async () => {
    const question = input.trim();
    if (!question || busy) return;
    setBusy(true);
    setError(null);
    setInput("");

    const userMsg: ChatMessage = { id: uid(), role: "user", question };
    const assistantId = uid();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      question,
      streaming: true,
      liveNodes: [],
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    const controller = new AbortController();
    try {
      const answer = await streamAsk(
        question,
        5,
        (node) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, liveNodes: [...(m.liveNodes ?? []), node] }
                : m
            )
          );
        },
        controller.signal
      );
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                streaming: false,
                liveNodes: [],
                answer: answer.answer,
                citations: answer.citations,
                grounded: answer.grounded,
                rewrites: answer.rewrites,
                trace: answer.trace,
                feedback: null,
                feedbackSent: false,
              }
            : m
        )
      );
      onNewCitations(question);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Ask failed";
      setError(msg);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, streaming: false, liveNodes: [], error: msg } : m
        )
      );
    } finally {
      setBusy(false);
    }
  };

  const handleFeedback = (id: string, rating: "up" | "down") => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, feedback: rating, feedbackSent: true } : m))
    );
  };

  return (
    <div className="chat-panel">
      <div className="chat-history">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Ask VeriRAG</h2>
            <p>
              Upload a PDF on the right, then ask a question. Answers come with
              page-level citations and a visible agent trace.
            </p>
          </div>
        )}
        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="msg user">
              <div className="msg-bubble">{m.question}</div>
            </div>
          ) : (
            <div key={m.id} className="msg assistant">
              {m.error ? (
                <div className="error-box">{m.error}</div>
              ) : m.streaming ? (
                <div className="streaming-box">
                  <div className="spinner" />
                  <span>Agent running{m.liveNodes && m.liveNodes.length > 0 ? `: ${m.liveNodes[m.liveNodes.length - 1]}` : "..."}</span>
                </div>
              ) : (
                <AnswerCard message={m} onFeedback={handleFeedback} />
              )}
            </div>
          )
        )}
      </div>
      {error && <div className="error-bar">{error}</div>}
      <div className="chat-input-row">
        <input
          className="chat-input"
          type="text"
          placeholder="Ask a question about your documents..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void send();
          }}
          disabled={busy}
        />
        <button className="send-btn" onClick={() => void send()} disabled={busy || !input.trim()}>
          {busy ? "Running..." : "Ask"}
        </button>
      </div>
    </div>
  );
}

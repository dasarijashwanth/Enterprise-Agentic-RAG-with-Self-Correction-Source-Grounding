"use client";

import { useState } from "react";
import type { ChatMessage } from "../lib/types";
import { sendFeedback } from "../lib/api";

function formatBbox(bbox: number[]): string {
  if (!bbox || bbox.length < 4) return "";
  const [x1, y1, x2, y2] = bbox.map((v) => Math.round(v));
  return `[Page bbox X1=${x1} Y1=${y1} X2=${x2} Y2=${y2}]`;
}

export default function AnswerCard({
  message,
  onFeedback,
}: {
  message: ChatMessage;
  onFeedback: (id: string, rating: "up" | "down") => void;
}) {
  const [showTrace, setShowTrace] = useState(false);
  const [sending, setSending] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);

  const handleFeedback = async (rating: "up" | "down") => {
    if (message.feedbackSent || sending) return;
    setSending(true);
    setFeedbackError(null);
    try {
      await sendFeedback(message.question, message.answer ?? "", rating);
      onFeedback(message.id, rating);
    } catch (err) {
      setFeedbackError(err instanceof Error ? err.message : "Feedback failed");
    } finally {
      setSending(false);
    }
  };

  const groundedLabel =
    message.grounded === undefined
      ? null
      : message.grounded
        ? "Grounded"
        : "Not grounded";

  return (
    <div className="answer-card">
      <div className="answer-meta">
        {groundedLabel && (
          <span className={message.grounded ? "badge grounded" : "badge ungrounded"}>
            {groundedLabel}
          </span>
        )}
        {message.rewrites !== undefined && message.rewrites > 0 && (
          <span className="badge rewrites">{message.rewrites} rewrite{message.rewrites === 1 ? "" : "s"}</span>
        )}
        {message.trace && message.trace.length > 0 && (
          <button className="link-btn" onClick={() => setShowTrace((v) => !v)}>
            {showTrace ? "Hide trace" : "Show trace"}
          </button>
        )}
      </div>

      {showTrace && message.trace && (
        <div className="trace-box">
          <div className="trace-title">Agent node trace</div>
          <ol>
            {message.trace.map((node, i) => (
              <li key={i}>{node}</li>
            ))}
          </ol>
        </div>
      )}

      <div className="answer-text">{message.answer}</div>

      {message.citations && message.citations.length > 0 && (
        <div className="citations">
          <div className="citations-title">Sources ({message.citations.length})</div>
          <ul>
            {message.citations.map((c) => (
              <li key={c.chunk_id}>
                <span className="cite-page">Page {c.page}</span>{" "}
                <span className="cite-bbox">{formatBbox(c.bbox)}</span>
                {c.text && <div className="cite-text">{c.text.slice(0, 220)}{c.text.length > 220 ? "..." : ""}</div>}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="feedback-row">
        <span className="feedback-label">Was this answer useful?</span>
        <button
          className={`thumb-btn ${message.feedback === "up" ? "active" : ""}`}
          disabled={message.feedbackSent || sending}
          onClick={() => handleFeedback("up")}
          aria-label="Thumbs up"
        >
          👍
        </button>
        <button
          className={`thumb-btn ${message.feedback === "down" ? "active" : ""}`}
          disabled={message.feedbackSent || sending}
          onClick={() => handleFeedback("down")}
          aria-label="Thumbs down"
        >
          👎
        </button>
        {message.feedbackSent && <span className="feedback-ok">Recorded, thanks.</span>}
        {feedbackError && <span className="feedback-err">{feedbackError}</span>}
      </div>
    </div>
  );
}

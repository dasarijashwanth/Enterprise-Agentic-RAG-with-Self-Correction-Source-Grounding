"use client";

import { useState } from "react";
import ChatPanel from "../components/ChatPanel";
import ContextPanel from "../components/ContextPanel";
import type { ChatMessage } from "../lib/types";
import type { IngestResponse } from "../lib/api";

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [docs, setDocs] = useState<IngestResponse[]>([]);

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>VeriRAG</h1>
        <span className="subtitle">Agentic RAG with self-correction and source grounding</span>
      </header>
      <div className="split">
        <ChatPanel
          messages={messages}
          setMessages={setMessages}
          onNewCitations={() => undefined}
        />
        <ContextPanel docs={docs} setDocs={setDocs} messages={messages} />
      </div>
    </main>
  );
}

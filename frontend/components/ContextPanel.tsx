"use client";

import { useRef, useState } from "react";
import type { IngestResponse } from "../lib/api";
import { uploadPdf } from "../lib/api";
import type { ChatMessage } from "../lib/types";

export default function ContextPanel({
  docs,
  setDocs,
  messages,
}: {
  docs: IngestResponse[];
  setDocs: (docs: IngestResponse[]) => void;
  messages: ChatMessage[];
}) {
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadOk, setUploadOk] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File | undefined) => {
    if (!file || uploading) return;
    setUploading(true);
    setUploadError(null);
    setUploadOk(null);
    try {
      const result = await uploadPdf(file);
      setDocs([...docs, result]);
      setUploadOk(`${result.filename}: ${result.pages} pages, ${result.chunks} chunks indexed.`);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const latestCited = [...messages]
    .reverse()
    .find((m) => m.role === "assistant" && m.citations && m.citations.length > 0);

  return (
    <div className="context-panel">
      <div className="panel-section">
        <h3>Documents</h3>
        <div className="upload-box">
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf,.pdf"
            disabled={uploading}
            onChange={(e) => void handleFile(e.target.files?.[0])}
          />
          {uploading && <div className="upload-status">Uploading and indexing...</div>}
          {uploadOk && <div className="upload-ok">{uploadOk}</div>}
          {uploadError && <div className="upload-err">{uploadError}</div>}
        </div>
        {docs.length === 0 ? (
          <p className="muted">No documents indexed yet. Upload a PDF to get started.</p>
        ) : (
          <ul className="doc-list">
            {docs.map((d, i) => (
              <li key={`${d.filename}-${i}`}>
                <div className="doc-name">{d.filename}</div>
                <div className="doc-meta">
                  {d.pages} pages, {d.chunks} chunks, {d.backend} index
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="panel-section">
        <h3>Sources for latest answer</h3>
        {!latestCited || !latestCited.citations ? (
          <p className="muted">Citations with page numbers appear here after an answer.</p>
        ) : (
          <ul className="cite-list">
            {latestCited.citations.map((c) => (
              <li key={c.chunk_id}>
                <span className="cite-page">Page {c.page}</span>
                {c.bbox && c.bbox.length >= 4 && (
                  <span className="cite-bbox">
                    [{c.bbox.map((v) => Math.round(v)).join(", ")}]
                  </span>
                )}
                {c.text && (
                  <div className="cite-text">
                    {c.text.slice(0, 160)}
                    {c.text.length > 160 ? "..." : ""}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

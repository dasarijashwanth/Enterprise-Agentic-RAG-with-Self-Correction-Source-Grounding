import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VeriRAG — Grounded Document Chat",
  description: "Agentic RAG with self-correction and source grounding",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

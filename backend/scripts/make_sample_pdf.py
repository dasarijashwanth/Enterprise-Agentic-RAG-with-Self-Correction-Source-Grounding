"""Generate the sample contract PDF used by FDE demo scenario 1.

Creates backend/data/sample_contract.pdf: a two-page vendor services
agreement with known clauses (termination notice on page 1, payment
terms on page 2), so the demo's questions have deterministic answers
and the emitted citations point at real page/bbox locations.

Run: cd backend && python -m scripts.make_sample_pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

PAGE1 = (
    "VENDOR SERVICES AGREEMENT\n\n"
    "The termination notice period is thirty (30) days. "
    "This Vendor Services Agreement (the Agreement) is entered into by and between "
    "Acme Corp (Client) and Beta Services LLC (Vendor).\n\n"
    "Section 1. Services. Vendor shall provide data pipeline maintenance and "
    "monitoring services as described in Exhibit A.\n\n"
    "Section 2. Termination. Either party may terminate this Agreement by giving "
    "thirty days written notice "
    "to the other party. Termination takes effect at the end of the notice period, "
    "and all outstanding invoices become due immediately upon termination.\n\n"
    "Section 3. Confidentiality. Each party agrees to keep confidential all "
    "non-public information received under this Agreement."
)

PAGE2 = (
    "Section 4. Payment. Payment terms are net thirty (30) days. Client shall pay "
    "all invoices within thirty days of receipt. Late payments accrue interest at "
    "one percent per month.\n\n"
    "Section 5. Liability. Vendor's total liability under this Agreement shall not "
    "exceed the fees paid by Client in the twelve months preceding the claim.\n\n"
    "Section 6. Governing Law. This Agreement is governed by the laws of the State "
    "of Florida. Signed and effective as of November 1, 2026."
)


def main() -> int:
    doc = pymupdf.open()
    for text in (PAGE1, PAGE2):
        page = doc.new_page()
        rect = pymupdf.Rect(72, 72, 540, 720)
        page.insert_textbox(rect, text, fontsize=11, fontname="helv")
    out = Path(__file__).resolve().parent.parent / "data" / "sample_contract.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    doc.close()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

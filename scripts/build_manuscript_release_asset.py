"""Prepend the VERA-IDS repository-edition cover to a rendered thesis PDF."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


REPOSITORY_URL = "https://github.com/AFR0011/VERA-IDS"
CORRECTION_URL = f"{REPOSITORY_URL}/blob/v2026.08/PROTOCOL_A_CORRECTION.md"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def _wrapped_lines(text: str, font: str, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and stringWidth(candidate, font, size) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _paragraph(
    pdf: canvas.Canvas,
    text: str,
    *,
    x: float,
    y: float,
    width: float,
    font: str = "Helvetica",
    size: float = 10.5,
    leading: float = 15,
    color: HexColor = HexColor("#243247"),
) -> float:
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in _wrapped_lines(text, font, size, width):
        pdf.drawString(x, y, line)
        y -= leading
    return y


def build_cover(path: Path) -> None:
    width, height = A4
    pdf = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    pdf.setTitle("VERA-IDS Thesis — Repository Edition")
    pdf.setAuthor("Ali Farrokhnejad")
    pdf.setSubject("Repository-edition cover for the VERA-IDS thesis release")

    navy = HexColor("#14253D")
    teal = HexColor("#0B8A8F")
    pale = HexColor("#EAF5F5")
    gray = HexColor("#5B6675")

    pdf.setFillColor(navy)
    pdf.rect(0, height - 200, width, 200, stroke=0, fill=1)
    pdf.setFillColor(teal)
    pdf.rect(0, height - 207, width, 7, stroke=0, fill=1)

    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(52, height - 48, "VERA-IDS  ·  RELEASE v2026.08")
    pdf.setFont("Helvetica-Bold", 25)
    title_y = height - 83
    for line in _wrapped_lines(
        "Beyond Closed-Set Accuracy: A Validity-Aware Evaluation Framework for Machine Learning-Based Intrusion Detection",
        "Helvetica-Bold",
        25,
        width - 104,
    ):
        pdf.drawString(52, title_y, line)
        title_y -= 30

    y = height - 250
    pdf.setFillColor(navy)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(52, y, "Repository Edition")
    y -= 28
    pdf.setFillColor(gray)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(52, y, "Ali Farrokhnejad  ·  2026")
    y -= 46

    pdf.setFillColor(pale)
    pdf.roundRect(45, y - 134, width - 90, 145, 9, stroke=0, fill=1)
    pdf.setFillColor(teal)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(62, y - 16, "STATUS OF THIS EDITION")
    y = _paragraph(
        pdf,
        "This PDF is a public repository edition of the thesis. The GitHub repository and release are not the institutional record; consult the host institution for the officially deposited version and administrative status.",
        x=62,
        y=y - 40,
        width=width - 124,
        size=10.5,
        leading=15,
    )
    y = _paragraph(
        pdf,
        "The thesis body is preserved as rendered from the cleaned authoritative source. This cover is the only page added for the repository release.",
        x=62,
        y=y - 8,
        width=width - 124,
        size=10.5,
        leading=15,
    )

    y -= 38
    pdf.setFillColor(navy)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(52, y, "PROTOCOL A CORRECTION")
    y = _paragraph(
        pdf,
        "Protocol A macro-F1 is corrected in the repository supplement: supported labels are the primary averaging set, while the original declared-output-label value is retained as historical evidence.",
        x=52,
        y=y - 24,
        width=width - 104,
    )
    pdf.setFillColor(teal)
    pdf.setFont("Helvetica-Bold", 9.5)
    pdf.drawString(52, y - 4, CORRECTION_URL)
    pdf.linkURL(CORRECTION_URL, (52, y - 7, width - 52, y + 7), relative=0)

    y -= 52
    pdf.setFillColor(navy)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(52, y, "LICENSE AND ACCESS")
    y = _paragraph(
        pdf,
        "© 2026 Ali Farrokhnejad. This manuscript release asset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Software in the repository is licensed separately under MIT.",
        x=52,
        y=y - 24,
        width=width - 104,
    )
    pdf.setFillColor(teal)
    pdf.setFont("Helvetica-Bold", 9.5)
    pdf.drawString(52, y - 4, REPOSITORY_URL)
    pdf.linkURL(REPOSITORY_URL, (52, y - 7, width - 52, y + 7), relative=0)
    pdf.drawString(52, y - 22, LICENSE_URL)
    pdf.linkURL(LICENSE_URL, (52, y - 25, width - 52, y - 11), relative=0)

    pdf.setFillColor(HexColor("#D5DBE3"))
    pdf.line(52, 62, width - 52, 62)
    pdf.setFillColor(gray)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(52, 46, "Public repository edition · release tag v2026.08 · 3 August 2026")
    pdf.save()


def build_release_asset(body_pdf: Path, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vera_ids_cover_") as temp_dir:
        cover_path = Path(temp_dir) / "cover.pdf"
        build_cover(cover_path)
        writer = PdfWriter()
        for page in PdfReader(cover_path).pages:
            writer.add_page(page)
        for page in PdfReader(body_pdf).pages:
            writer.add_page(page)
        writer.add_metadata(
            {
                "/Title": "Beyond Closed-Set Accuracy — VERA-IDS Repository Edition",
                "/Author": "Ali Farrokhnejad",
                "/Subject": "Machine learning-based intrusion detection; repository edition v2026.08",
                "/Keywords": "VERA-IDS, intrusion detection, machine learning, validity-aware evaluation",
                "/Creator": "VERA-IDS release pipeline",
                "/Producer": "pypdf",
            }
        )
        with output_pdf.open("wb") as stream:
            writer.write(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("body_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()
    build_release_asset(args.body_pdf.resolve(), args.output_pdf.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

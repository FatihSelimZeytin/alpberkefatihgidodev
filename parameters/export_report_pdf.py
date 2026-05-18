from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_MD = ROOT / "REPORT.md"
OUTPUT_PDF = ROOT / "REPORT.pdf"


def escape(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def clean_inline_markdown(text):
    text = escape(text)
    text = text.replace("`", "")
    text = text.replace("**", "")
    return text


def parse_table(lines):
    rows = []
    for line in lines:
        parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(part) <= {"-", ":"} for part in parts):
            continue
        rows.append(parts)
    return rows


def add_table(story, rows, styles):
    if not rows:
        return

    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    data = [
        [Paragraph(clean_inline_markdown(cell), styles["TableCell"]) for cell in row]
        for row in normalized
    ]

    available_width = A4[0] - 4 * cm
    col_widths = [available_width / max_cols] * max_cols
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F2EE")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F3329")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BFCBC5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.25 * cm))


def add_paragraph_block(story, paragraph_lines, style):
    if not paragraph_lines:
        return
    text = " ".join(line.strip() for line in paragraph_lines).strip()
    if text:
        story.append(Paragraph(clean_inline_markdown(text), style))
        story.append(Spacer(1, 0.18 * cm))


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=25,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F3329"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading1Custom",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#1F3329"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading2Custom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#284537"),
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            leftIndent=12,
            firstLineIndent=-8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Meta",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#56645E"),
        )
    )
    return styles


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#66736D"))
    canvas.drawCentredString(A4[0] / 2, 1.05 * cm, f"Page {doc.page}")
    canvas.restoreState()


def markdown_to_story(markdown_text, styles):
    story = []
    lines = markdown_text.splitlines()
    paragraph_lines = []
    table_lines = []
    code_lines = []
    in_code = False

    def flush_paragraph():
        nonlocal paragraph_lines
        add_paragraph_block(story, paragraph_lines, styles["BodyCustom"])
        paragraph_lines = []

    def flush_table():
        nonlocal table_lines
        if table_lines:
            add_table(story, parse_table(table_lines), styles)
            table_lines = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            story.append(
                Preformatted(
                    "\n".join(code_lines),
                    ParagraphStyle(
                        "Code",
                        fontName="Courier",
                        fontSize=7.5,
                        leading=10,
                        backColor=colors.HexColor("#F4F6F5"),
                        borderColor=colors.HexColor("#D5DDD9"),
                        borderWidth=0.4,
                        borderPadding=5,
                    ),
                )
            )
            story.append(Spacer(1, 0.2 * cm))
            code_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                flush_table()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if "|" in stripped and stripped.startswith("|"):
            flush_paragraph()
            table_lines.append(line)
            continue
        else:
            flush_table()

        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(clean_inline_markdown(stripped[2:]), styles["ReportTitle"]))
            story.append(Paragraph("CME434 Geographic Information Systems Final Project", styles["Meta"]))
            story.append(Paragraph("Konya and Surroundings, Turkiye", styles["Meta"]))
            story.append(Spacer(1, 0.35 * cm))
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(clean_inline_markdown(stripped[3:]), styles["Heading1Custom"]))
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(clean_inline_markdown(stripped[4:]), styles["Heading2Custom"]))
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            story.append(Paragraph("- " + clean_inline_markdown(stripped[2:]), styles["BulletCustom"]))
            continue

        if len(stripped) > 3 and stripped[0].isdigit() and ". " in stripped[:5]:
            flush_paragraph()
            story.append(Paragraph(clean_inline_markdown(stripped), styles["BulletCustom"]))
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    flush_table()
    flush_code()
    return story


def main():
    styles = build_styles()
    markdown_text = INPUT_MD.read_text(encoding="utf-8-sig")
    story = markdown_to_story(markdown_text, styles)

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Konya Agricultural Suitability Mapping Report",
        author="CME434 Final Project",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"PDF report saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()

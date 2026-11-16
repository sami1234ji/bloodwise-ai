"""Build the downloadable PDF analysis report with ReportLab.

The report carries the results, the interval used for each one and where that
interval came from, the explanation, the limitations and the sources. It opens
and closes with the statement that this is not a diagnosis.

No patient name is written into the report, because the application never
extracts one.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether)

from .ai_explainer import DISCLAIMER

NAVY = colors.HexColor("#123a5f")
BLUE = colors.HexColor("#2a7ab0")
LIGHT = colors.HexColor("#eaf2f8")
GREY = colors.HexColor("#5b6b77")
RED = colors.HexColor("#b3261e")
AMBER = colors.HexColor("#8a5a00")
GREEN = colors.HexColor("#1e6b45")

STATUS_COLOR = {"low": AMBER, "high": RED, "normal": GREEN, "unknown": GREY}


def _styles():
    s = getSampleStyleSheet()
    base = dict(fontName="Helvetica", leading=13.5, spaceAfter=6)
    return {
        "title": ParagraphStyle("t", parent=s["Title"], fontName="Helvetica-Bold",
                                fontSize=20, textColor=NAVY, spaceAfter=2),
        "sub": ParagraphStyle("s", fontSize=9.5, textColor=GREY, spaceAfter=10, **{}),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5,
                             textColor=NAVY, spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.5,
                             textColor=BLUE, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("b", fontSize=9.5, alignment=TA_JUSTIFY, **base),
        "small": ParagraphStyle("sm", fontSize=8, textColor=GREY, leading=10.5,
                                spaceAfter=4, fontName="Helvetica"),
        "cell": ParagraphStyle("c", fontSize=8.5, leading=10.5, fontName="Helvetica"),
        "cellb": ParagraphStyle("cb", fontSize=8.5, leading=10.5,
                                fontName="Helvetica-Bold"),
        "banner": ParagraphStyle("bn", fontSize=8.8, leading=11.5, textColor=NAVY,
                                 fontName="Helvetica-Oblique"),
    }


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 14 * mm, A4[0], 14 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(18 * mm, A4[1] - 9.5 * mm, "BloodWise AI - Blood Report Analysis")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 9.5 * mm,
                           "Educational screening tool, not a diagnosis")
    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, 10 * mm,
                      "Not a medical diagnostic system. Discuss results with a healthcare professional.")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _status_cell(status, st):
    return Paragraph(f'<font color="{STATUS_COLOR[status].hexval()}"><b>{status.upper()}</b></font>',
                     st["cell"])


def build_report(findings, explanation, anomaly=None, meta=None,
                 path: str | Path | None = None) -> bytes:
    """Render the PDF. Returns the bytes, and writes them to `path` if given."""
    st = _styles()
    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=20 * mm, bottomMargin=16 * mm,
                          title="BloodWise AI analysis report", author="BloodWise AI")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=_header_footer)])

    meta = meta or {}
    flow = []
    flow.append(Paragraph("Blood Report Analysis", st["title"]))
    flow.append(Paragraph(
        f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')}"
        + (f" &nbsp;|&nbsp; Sex used for intervals: {meta['sex']}" if meta.get("sex") else "")
        + (f" &nbsp;|&nbsp; Age: {meta['age']}" if meta.get("age") else ""),
        st["sub"]))

    flow.append(Table([[Paragraph(DISCLAIMER, st["banner"])]],
                      colWidths=[doc.width],
                      style=TableStyle([
                          ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                          ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
                          ("LEFTPADDING", (0, 0), (-1, -1), 8),
                          ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                          ("TOPPADDING", (0, 0), (-1, -1), 7),
                          ("BOTTOMPADDING", (0, 0), (-1, -1), 7)])))
    flow.append(Spacer(1, 10))

    # ---- summary ----
    flow.append(Paragraph("Summary", st["h2"]))
    flow.append(Paragraph(explanation.summary, st["body"]))

    # ---- results table, grouped by category ----
    flow.append(Paragraph("Results", st["h2"]))
    by_cat = {}
    for f in findings:
        by_cat.setdefault(f.category, []).append(f)

    head = [Paragraph(h, st["cellb"]) for h in
            ("Test", "Result", "Interval used", "Status", "Interval from")]
    widths = [doc.width * w for w in (0.30, 0.17, 0.23, 0.12, 0.18)]
    origin_label = {"report": "your report", "reference_table": "reference table",
                    "none": "none available"}

    for category, items in by_cat.items():
        rows = [head]
        for f in items:
            rows.append([
                Paragraph(f.display_name, st["cell"]),
                Paragraph(f"{f.value:g} {f.unit}", st["cell"]),
                Paragraph(f.range_text(), st["cell"]),
                _status_cell(f.status, st),
                Paragraph(origin_label.get(f.ref_origin, f.ref_origin), st["cell"]),
            ])
        t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, BLUE),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8df")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ]))
        flow.append(KeepTogether([Paragraph(category, st["h3"]), t, Spacer(1, 6)]))

    # ---- explanation ----
    if explanation.sections:
        flow.append(Paragraph("What these results mean", st["h2"]))
        for s in explanation.sections:
            flow.append(KeepTogether([
                Paragraph(s["heading"], st["h3"]),
                Paragraph(s["body"].replace("&", "&amp;").replace("<", "&lt;"), st["body"]),
            ]))

    if explanation.associations:
        flow.append(Paragraph("Possible associations to discuss with a clinician", st["h2"]))
        for a in explanation.associations:
            flow.append(Paragraph(f"<b>{a.name}.</b> {a.text}", st["body"]))

    # ---- anomaly ----
    if anomaly is not None and getattr(anomaly, "available", False):
        flow.append(Paragraph("Unusual-combination check", st["h2"]))
        flow.append(Paragraph(anomaly.message, st["body"]))
        if anomaly.mahalanobis is not None:
            flow.append(Paragraph(
                f"Mahalanobis distance {anomaly.mahalanobis:.2f} against a threshold of "
                f"{anomaly.mahalanobis_threshold:.2f}; isolation forest "
                f"{anomaly.isolation:.3f} against {anomaly.isolation_threshold:.3f}. "
                f"{anomaly.features_used} of {anomaly.features_total} parameters were "
                f"available.", st["small"]))
        if anomaly.triggered_by:
            flow.append(Paragraph("Raised by: " + ", ".join(anomaly.triggered_by) + ".",
                                  st["small"]))
        if anomaly.contributors:
            bits = ", ".join(f"{c['parameter']} ({c['direction']} the population average)"
                             for c in anomaly.contributors)
            flow.append(Paragraph(f"Contributing most: {bits}.", st["small"]))
        flow.append(Paragraph(
            "This is a statistical comparison with a survey population, not a clinical "
            "judgement and not disease prediction.", st["small"]))

    # ---- next steps ----
    if explanation.next_steps:
        flow.append(Paragraph("What may be worth doing next", st["h2"]))
        for s in explanation.next_steps:
            flow.append(Paragraph(f"&bull; {s}", st["body"]))

    # ---- limitations ----
    flow.append(Paragraph("Limitations", st["h2"]))
    for line in (
        "This tool does not diagnose, and does not recommend medicines or treatment.",
        "Values were read automatically from the uploaded file and may contain errors "
        "even after review.",
        "Reference intervals differ between laboratories, methods, ages and sexes. Where "
        "your report printed its own interval, that interval was used; otherwise a stored "
        "interval was used and is labelled as such.",
        "The unusual-combination check compares your values with a United States health "
        "survey population, not with a clinical population.",
        "A result inside an interval does not guarantee health, and a result outside one "
        "does not by itself mean disease.",
    ):
        flow.append(Paragraph(f"&bull; {line}", st["small"]))

    # ---- sources ----
    flow.append(Paragraph("Sources", st["h2"]))
    seen = set()
    for c in explanation.citations:
        if c not in seen:
            seen.add(c)
            flow.append(Paragraph(f"&bull; {c}", st["small"]))
    for f in findings:
        if f.ref_origin == "reference_table" and f.ref_source_name:
            key = f"{f.ref_source_name} ({f.ref_source_url})"
            if key not in seen:
                seen.add(key)
                flow.append(Paragraph(f"&bull; Reference intervals: {key}", st["small"]))
    flow.append(Paragraph(
        "&bull; Explanatory text adapted from MedlinePlus, National Library of Medicine.",
        st["small"]))

    flow.append(Spacer(1, 8))
    flow.append(Paragraph(DISCLAIMER, st["banner"]))

    doc.build(flow)
    data = buf.getvalue()
    if path:
        Path(path).write_bytes(data)
    return data

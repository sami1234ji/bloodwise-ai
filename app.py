"""BloodWise AI - AI-Powered Blood Report Analyzer.

    streamlit run app.py

An educational screening and information tool. It does not diagnose, and it
does not recommend medicines or treatment.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from dashboard import charts
from dashboard.results import (ORIGIN_LABEL, STATUS_STYLE, headline,
                               group_by_category, results_rows, review_warnings,
                               status_counts)
from modules.ai_explainer import DISCLAIMER, AIExplainer
from modules.data_normalizer import UnitConverter
from modules.image_ocr import TesseractMissing, ocr_image, tesseract_available
from modules.ml_analyzer import AnomalyDetector, values_from_findings
from modules.parameter_extractor import (ParameterExtractor, from_review_table,
                                         to_review_table)
from modules.pdf_extractor import extract_text, is_pdf
from modules.rag_engine import Retriever
from modules.reference_analyzer import ReferenceTable, analyse
from modules.report_generator import build_report

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_MB = 12
PAGES = ["Home", "Upload Report", "Review & Analysis", "Health Insights",
         "Visual Analytics", "Download Report", "About"]

st.set_page_config(page_title="BloodWise AI", page_icon="\U0001FA78",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
  .stApp { background: #f6f9fc; }
  section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e3eaf0; }
  h1, h2, h3 { color: #123a5f; }
  .block-container { padding-top: 2.2rem; max-width: 1200px; }
  .bw-top { background: #fff; border: 1px solid #e3eaf0; border-left: 5px solid #2a7ab0;
            border-radius: 12px; padding: 14px 18px; margin-bottom: 18px; }
  .bw-top-row { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .bw-brand { font-size: 1.25rem; font-weight: 700; color: #123a5f; }
  .bw-tag { font-size: .95rem; color: #33475b; }
  .bw-steps { margin-top: 8px; font-size: .82rem; color: #5b6b77; }
  .bw-steps b { color: #123a5f; }
  .bw-warn { margin-top: 6px; font-size: .78rem; color: #8a5a00; }
  .bw-note { background: #eaf2f8; border: 1px solid #bcd6ea; color: #123a5f;
             padding: 11px 15px; border-radius: 10px; font-size: .88rem; }
  .bw-card { background: #fff; border: 1px solid #e3eaf0; border-radius: 12px;
             padding: 14px 16px; height: 100%; }
  .bw-metric { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
  .bw-metric-label { font-size: .78rem; text-transform: uppercase;
                     letter-spacing: .06em; color: #5b6b77; }
  .bw-pill { display: inline-block; padding: 2px 10px; border-radius: 999px;
             font-size: .74rem; font-weight: 700; letter-spacing: .04em; }
  .bw-src { font-size: .78rem; color: #5b6b77; }
  .bw-finding { background:#fff; border:1px solid #e3eaf0; border-left-width:5px;
                border-radius:10px; padding:12px 15px; margin-bottom:10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------- loading ---
@st.cache_resource(show_spinner=False)
def load_engine():
    table = ReferenceTable()
    return {
        "table": table,
        "extractor": ParameterExtractor(table, UnitConverter()),
        "detector": AnomalyDetector(),
        "retriever": Retriever(),
        "explainer": AIExplainer(),
    }


E = load_engine()

DEFAULTS = {"page": "Home", "raw_text": None, "extraction": None, "review": None,
            "findings": None, "explanation": None, "anomaly": None,
            "source_name": None, "source_notes": [], "used_ocr": False,
            "sex": None, "age": None, "analysed_at": None}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def pill(status: str) -> str:
    s = STATUS_STYLE[status]
    return (f'<span class="bw-pill" style="background:{s["bg"]};color:{s["color"]}">'
            f'{s["icon"]} {s["label"]}</span>')


def disclaimer_banner():
    st.markdown(f'<div class="bw-note">{DISCLAIMER}</div>', unsafe_allow_html=True)


def top_bar():
    """One short strip on every page saying what the site does."""
    st.markdown(
        '<div class="bw-top"><div class="bw-top-row">'
        '<span class="bw-brand">\U0001FA78 BloodWise AI</span>'
        '<span class="bw-tag">Upload a blood report and get a plain-language '
        'explanation of every value.</span></div>'
        '<div class="bw-warn">Educational tool, not a diagnosis. Discuss results '
        'with a doctor.</div></div>', unsafe_allow_html=True)


def reset_analysis():
    for key in ("findings", "explanation", "anomaly", "analysed_at"):
        st.session_state[key] = None


# --------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.session_state.page = st.radio("Menu", PAGES,
                                     index=PAGES.index(st.session_state.page))
    st.divider()

    st.markdown("**Patient context**")
    st.caption("Some reference intervals differ by sex. The name is never read "
               "from the report.")
    sex_options = ["not set", "female", "male"]
    current = st.session_state.sex or "not set"
    chosen = st.selectbox("Sex", sex_options, index=sex_options.index(current)
                          if current in sex_options else 0)
    st.session_state.sex = None if chosen == "not set" else chosen
    st.session_state.age = st.number_input("Age", 0, 120,
                                           value=int(st.session_state.age or 30))
    st.divider()

    with st.expander("System status"):
        ocr_ok = tesseract_available()
        st.write(("✅" if ocr_ok else "⚠️") + " Tesseract OCR"
                 + ("" if ocr_ok else " not found"))
        st.write(("✅" if E["detector"].available else "⚠️")
                 + " Anomaly model"
                 + ("" if E["detector"].available else " not trained"))
        st.write("✅ Retrieval: " + E["retriever"].backend)
        llm = E["explainer"].llm_available
        st.write(("✅" if llm else "ℹ️") + " AI rewriting"
                 + ("" if llm else " off (no API key)"))
        st.toggle("Use AI rewriting", value=llm, disabled=not llm, key="use_llm",
                  help="With no API key the app uses its rule-based explanations, "
                       "which are complete on their own.")
    st.caption("Uploads are processed in memory and are not stored.")


# ------------------------------------------------------------------ home ---
def page_home():
    findings = st.session_state.findings
    if findings:
        c = status_counts(findings)
        cols = st.columns(4)
        for col, (label, value, color) in zip(cols, [
                ("Tests read", c["total"], "#123a5f"),
                ("Outside interval", c["abnormal"], "#b3261e"),
                ("Low", c["low"], "#8a5a00"),
                ("High", c["high"], "#b3261e")]):
            col.markdown(f'<div class="bw-card"><div class="bw-metric-label">{label}'
                         f'</div><div class="bw-metric" style="color:{color}">{value}'
                         f'</div></div>', unsafe_allow_html=True)
        st.write("")
        st.info(headline(findings))
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"**{charts.TITLES['status_donut']}**")
            st.plotly_chart(charts.status_donut(findings), use_container_width=True)
        with right:
            st.markdown(f"**{charts.TITLES['category_bars']}**")
            st.plotly_chart(charts.category_bars(findings), use_container_width=True)
        st.caption(f"Analysed {st.session_state.analysed_at}. "
                   f"Source file: {st.session_state.source_name}")
    else:
        st.subheader("How it works")
        st.write("Upload a PDF or a photo of your report. You will see every value "
                 "that was read from it, and you can fix any that are wrong. After "
                 "that, each result is marked low, normal or high and explained in "
                 "plain words.")

        table = E["table"]
        groups = {}
        for r in table.rows:
            groups.setdefault(r.category, [])
            if r.display_name not in groups[r.category]:
                groups[r.category].append(r.display_name)
        total = sum(len(v) for v in groups.values())
        st.subheader("What it covers")
        st.write(f"{total} common tests: blood count, vitamins, thyroid, sugar, "
                 "cholesterol, kidney, liver and iron.")
        with st.expander("See the full list"):
            for category, names in groups.items():
                st.markdown(f"**{category}:** {', '.join(names)}")
        st.write("")
        if st.button("Upload a report", type="primary"):
            st.session_state.page = "Upload Report"
            st.rerun()


# ---------------------------------------------------------------- upload ---
def read_upload(data: bytes, filename: str):
    notes, used_ocr = [], False
    if is_pdf(data):
        out = extract_text(data)
        text, used_ocr = out.text, out.used_ocr
        notes.extend(out.notes)
        notes.append(f"Read {out.pages} page(s) with {out.engine}.")
    else:
        res = ocr_image(data)
        text, used_ocr = res.text, True
        notes.extend(res.notes)
        notes.append(f"OCR confidence {res.mean_confidence:.0%}.")
    return text, used_ocr, notes


def page_upload():
    st.header("Upload a blood report")
    disclaimer_banner()
    st.write("")

    if not tesseract_available():
        st.warning("Tesseract OCR is not installed, so photographs and scanned PDFs "
                   "cannot be read. PDFs with a text layer still work. Installing "
                   "pytesseract is not enough; see the README.")

    uploaded = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"],
                                help=f"PDF, PNG or JPG, up to {MAX_UPLOAD_MB} MB.")
    st.caption("Or try one of the generated sample reports, which contain invented "
               "values and no real patient data.")

    sample_dir = ROOT / "sample_reports"
    samples = sorted(p.name for p in sample_dir.glob("*")
                     if p.suffix.lower() in (".pdf", ".png", ".jpg")) \
        if sample_dir.exists() else []
    picked = st.selectbox("Sample report", ["none"] + samples)

    data = name = None
    if uploaded is not None:
        data = uploaded.getvalue()
        name = uploaded.name
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"That file is larger than {MAX_UPLOAD_MB} MB.")
            return
    elif picked != "none":
        data = (sample_dir / picked).read_bytes()
        name = picked

    if data is None:
        return
    if not is_pdf(data) and data[:3] not in (b"\xff\xd8\xff",) and data[:8] != \
            b"\x89PNG\r\n\x1a\n":
        st.error("That file does not look like a PDF, PNG or JPG.")
        return

    if st.button("Read this report", type="primary"):
        with st.spinner("Reading the report..."):
            try:
                text, used_ocr, notes = read_upload(data, name)
            except TesseractMissing as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"The file could not be read: {exc}")
                return
            result = E["extractor"].extract(text, from_ocr=used_ocr)

        st.session_state.update(
            raw_text=text, extraction=result, source_name=name,
            source_notes=notes, used_ocr=used_ocr,
            review=to_review_table(result, E["table"]),
            sex=st.session_state.sex or result.sex,
            age=result.age or st.session_state.age)
        reset_analysis()

        if not result.readings:
            st.error("No test results were recognised in that file. If it is a photo, "
                     "try a sharper, straighter image with even lighting.")
            with st.expander("Text that was read from the file"):
                st.text(text[:4000] or "(nothing)")
            return

        st.session_state.page = "Review & Analysis"
        st.rerun()


# ---------------------------------------------------------------- review ---
def page_review():
    st.header("Review and analysis")
    result = st.session_state.extraction
    if result is None:
        st.info("Upload a report first.")
        return

    st.success(f"Read {len(result.readings)} test results from "
               f"{st.session_state.source_name}.")
    for note in st.session_state.source_notes:
        st.caption(note)

    st.markdown('<div class="bw-note">Check every value against your own report '
                'before continuing. Values read with low confidence are marked. '
                'Nothing is analysed until you confirm.</div>',
                unsafe_allow_html=True)
    st.write("")

    for note in review_warnings(result):
        st.warning(note)

    edited = st.data_editor(
        st.session_state.review,
        key="review_editor",
        use_container_width=True,
        hide_index=True,
        column_config={
            "parameter": None,
            "Test": st.column_config.TextColumn(disabled=True),
            "Value": st.column_config.NumberColumn(format="%.4g", required=True),
            "Unit": st.column_config.TextColumn(),
            "Report low": st.column_config.NumberColumn(format="%.4g"),
            "Report high": st.column_config.NumberColumn(format="%.4g"),
            "Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0,
                                                          format="%.2f"),
            "Needs review": st.column_config.CheckboxColumn(disabled=True),
            "Issues": st.column_config.TextColumn(disabled=True, width="medium"),
            "Read from": st.column_config.TextColumn(disabled=True, width="medium"),
        })
    st.session_state.review = edited

    if result.unmatched_lines:
        with st.expander(f"{len(result.unmatched_lines)} lines were not recognised"):
            st.text("\n".join(result.unmatched_lines[:60]))

    with st.expander("Text that was read from the file"):
        st.text((st.session_state.raw_text or "")[:6000])

    st.write("")
    if st.button("Confirm and analyse", type="primary"):
        readings = from_review_table(
            edited.to_dict("records") if hasattr(edited, "to_dict") else edited)
        if not readings:
            st.error("There are no values to analyse.")
            return
        with st.spinner("Analysing..."):
            findings = analyse(readings, E["table"], sex=st.session_state.sex)
            anomaly = E["detector"].score_panel(values_from_findings(findings))
            passages = E["retriever"].context_for(findings)
            explanation = E["explainer"].explain(
                findings, passages=passages, anomaly=anomaly,
                use_llm=bool(st.session_state.get("use_llm", False)))
        st.session_state.update(
            findings=findings, anomaly=anomaly, explanation=explanation,
            analysed_at=datetime.now().strftime("%d %b %Y, %H:%M"))
        st.success("Analysis complete.")

    findings = st.session_state.findings
    if not findings:
        return

    st.divider()
    st.subheader("Results")
    st.info(headline(findings))
    for category, items in group_by_category(findings).items():
        st.markdown(f"**{category}**")
        for f in items:
            st.markdown(
                f'<div class="bw-finding" style="border-left-color:'
                f'{STATUS_STYLE[f.status]["color"]}">'
                f'<b>{f.display_name}</b> &nbsp; {pill(f.status)}<br>'
                f'<span style="font-size:1.05rem"><b>{f.value:g} {f.unit}</b></span>'
                f'&nbsp;&middot;&nbsp; interval used: {f.range_text()} '
                f'<span class="bw-src">({ORIGIN_LABEL.get(f.ref_origin, "")})</span>'
                f'</div>', unsafe_allow_html=True)


# --------------------------------------------------------------- insights ---
def page_insights():
    st.header("Health insights")
    explanation = st.session_state.explanation
    findings = st.session_state.findings
    if not explanation:
        st.info("Run an analysis first.")
        return

    disclaimer_banner()
    st.write("")
    st.subheader("Summary")
    st.write(explanation.summary)
    if explanation.llm_note:
        st.caption(explanation.llm_note)

    anomaly = st.session_state.anomaly
    if anomaly is not None and anomaly.available and anomaly.mahalanobis is not None:
        st.subheader("Unusual-combination check")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            figure = charts.anomaly_gauge(anomaly)
            if figure:
                st.markdown(f"**{charts.TITLES['anomaly_gauge']}**")
                st.plotly_chart(figure, use_container_width=True)
        with c2:
            (st.warning if anomaly.is_unusual else st.success)(anomaly.message)
            st.caption(f"Mahalanobis distance {anomaly.mahalanobis:.2f} against a "
                       f"threshold of {anomaly.mahalanobis_threshold:.2f}; isolation "
                       f"forest {anomaly.isolation:.3f} against "
                       f"{anomaly.isolation_threshold:.3f}. "
                       f"{anomaly.features_used} of {anomaly.features_total} "
                       f"parameters available.")
            st.caption(anomaly.training_note)
    elif anomaly is not None and anomaly.message:
        st.subheader("Unusual-combination check")
        st.info(anomaly.message)

    if explanation.associations:
        st.subheader("Possible associations to discuss with a clinician")
        for a in explanation.associations:
            with st.container(border=True):
                st.markdown(f"**{a.name}**")
                st.write(a.text)
                st.caption("Sources: " + "  ".join(a.sources))

    if explanation.sections:
        st.subheader("Each result in turn")
        for s in explanation.sections:
            with st.expander(s["heading"], expanded=s["heading"] == "In plain language"):
                st.write(s["body"])
                if s.get("sources"):
                    st.caption("Sources: " + "  ".join(dict.fromkeys(s["sources"])))

    if explanation.next_steps:
        st.subheader("What may be worth doing next")
        for s in explanation.next_steps:
            st.markdown(f"- {s}")

    if explanation.citations:
        st.subheader("Sources")
        for c in dict.fromkeys(explanation.citations):
            st.markdown(f"- {c}")
    st.caption("Explanatory text adapted from MedlinePlus, National Library of "
               "Medicine. Reference intervals from the source named on each result.")


# ------------------------------------------------------------- analytics ---
def page_analytics():
    st.header("Visual analytics")
    findings = st.session_state.findings
    if not findings:
        st.info("Run an analysis first.")
        return

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(f"**{charts.TITLES['status_donut']}**")
        st.plotly_chart(charts.status_donut(findings), use_container_width=True)
    with c2:
        st.markdown(f"**{charts.TITLES['category_bars']}**")
        st.plotly_chart(charts.category_bars(findings), use_container_width=True)

    figure = charts.range_chart(findings)
    if figure:
        st.markdown(f"**{charts.TITLES['range_chart']}**")
        st.plotly_chart(figure, use_container_width=True)
        st.caption("Each interval is drawn as 0 to 1 so every test fits one scale. "
                   "A cross means the value is far outside its interval and has been "
                   "pulled in to keep the chart readable. Tests with no interval, or "
                   "with only one limit, are not shown here.")

    anomaly = st.session_state.anomaly
    if anomaly is not None and anomaly.contributors:
        figure = charts.contributor_bars(anomaly)
        if figure:
            st.markdown(f"**{charts.TITLES['contributor_bars']}**")
            st.plotly_chart(figure, use_container_width=True)
            st.caption("How much each parameter contributed to the "
                       "unusual-combination distance.")

    st.subheader("All results")
    st.dataframe(results_rows(findings), use_container_width=True, hide_index=True)


# -------------------------------------------------------------- download ---
def page_download():
    st.header("Download the analysis")
    findings = st.session_state.findings
    explanation = st.session_state.explanation
    if not findings or not explanation:
        st.info("Run an analysis first.")
        return

    st.write("The PDF holds every result, the interval used and where it came from, "
             "the explanation, the limitations and the sources. No name is included, "
             "because the application never reads one.")
    meta = {"sex": st.session_state.sex, "age": st.session_state.age}
    with st.spinner("Building the PDF..."):
        pdf = build_report(findings, explanation, anomaly=st.session_state.anomaly,
                           meta=meta)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button("Download PDF report", data=pdf,
                       file_name=f"bloodwise_analysis_{stamp}.pdf",
                       mime="application/pdf", type="primary")
    st.caption(f"{len(pdf) // 1024} KB")
    st.divider()
    st.subheader("Results as text")
    st.code(explanation.as_text(), language=None)


# ----------------------------------------------------------------- about ---
def page_about():
    st.header("About this project")
    disclaimer_banner()
    st.write("")

    st.subheader("What it is")
    st.write(
        "BloodWise AI reads a blood report, compares each value with a reference "
        "interval, points out patterns worth discussing with a doctor and produces a "
        "downloadable summary. It was built as a B.Tech one-credit project.")

    st.subheader("Where the medical information comes from")
    st.markdown(
        "- **Reference intervals**: the ABIM Laboratory Test Reference Ranges "
        "(January 2026). Every row in `data/reference_ranges.csv` records its source "
        "and the date it was read. A parameter with no verified interval is reported "
        "without a judgement rather than guessed at.\n"
        "- **Explanations**: the MedlinePlus Medical Tests pages from the US National "
        "Library of Medicine, which are in the public domain. Content MedlinePlus "
        "marks as copyrighted is not used.\n"
        "- **Model training data**: the NHANES 2017-2018 public laboratory files from "
        "the CDC.")
    st.caption("Source: MedlinePlus, National Library of Medicine.")

    st.subheader("How intervals are chosen")
    st.write(
        "Intervals differ between laboratories, methods, ages and sexes, so there is "
        "no single correct set. Where your report prints its own interval, that one is "
        "used and labelled as such. Otherwise the stored interval is used and labelled "
        "as that instead, so you always know which you are looking at.")

    st.subheader("What the machine learning does")
    detector = E["detector"]
    if detector.available:
        meta = detector.metadata
        st.write(
            f"Two detectors are trained on {meta['n_train']} adults from NHANES, using "
            f"{len(meta['features'])} parameters. They judge how unusual a *combination* "
            f"of values is, which is the one thing comparing each value with its own "
            f"interval cannot do. They do not predict disease.")
        st.caption(meta["training_note"])
        evaluation = ROOT / "reports" / "ml_evaluation.json"
        if evaluation.exists():
            with st.expander("Measured performance"):
                st.json(evaluation.read_text()[:4000])
    else:
        st.info("The model has not been trained in this copy. Run "
                "`python scripts/prepare_nhanes.py` then `python scripts/train_model.py`.")

    st.subheader("Privacy")
    st.markdown(
        "- The patient name is never extracted from the report.\n"
        "- Uploads are held in memory for the session and are not written to disk.\n"
        "- Nothing is sent to any external service unless AI rewriting is switched on, "
        "and then only the findings and the reference passages are sent.\n"
        "- The API key is read from an environment variable and is never stored in "
        "the code.")

    st.subheader("Limitations")
    st.markdown(
        "- It does not diagnose, and it does not recommend medicines or treatment.\n"
        "- Values are read automatically and can be misread, which is why the review "
        "step exists.\n"
        "- The training data is a United States survey population, so the "
        "unusual-combination check reflects that group.\n"
        "- No thyroid or vitamin B12 data was available in that survey cycle, so the "
        "combination check cannot take those parameters into account.\n"
        "- A result inside an interval does not guarantee health, and one outside it "
        "does not by itself mean disease.")

    st.subheader("Team")
    st.write("Abdul Sami Siddiqui (2427030061) and Aditya Singh (2427030054), "
             "supervised by Dr. Shashvat. Department of Computer Science and "
             "Engineering, Manipal University Jaipur.")


ROUTES = {"Home": page_home, "Upload Report": page_upload,
          "Review & Analysis": page_review, "Health Insights": page_insights,
          "Visual Analytics": page_analytics, "Download Report": page_download,
          "About": page_about}

top_bar()
ROUTES[st.session_state.page]()

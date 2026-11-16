"""Shaping findings for display.

The functions here are pure: they take findings and return plain data. Keeping
them free of Streamlit means they can be tested without a running interface,
which is where most display bugs would otherwise hide.
"""

from __future__ import annotations

STATUS_STYLE = {
    "low": {"label": "LOW", "icon": "▼", "color": "#8a5a00", "bg": "#fdf3e0"},
    "high": {"label": "HIGH", "icon": "▲", "color": "#b3261e", "bg": "#fdecea"},
    "normal": {"label": "NORMAL", "icon": "●", "color": "#1e6b45", "bg": "#e8f5ee"},
    "unknown": {"label": "NO REFERENCE", "icon": "—", "color": "#5b6b77",
                "bg": "#eef2f5"},
}

ORIGIN_LABEL = {
    "report": "interval printed on your report",
    "reference_table": "stored reference interval",
    "none": "no reference interval available",
}


def status_counts(findings) -> dict:
    counts = {"low": 0, "high": 0, "normal": 0, "unknown": 0}
    for f in findings:
        counts[f.status] = counts.get(f.status, 0) + 1
    counts["total"] = len(findings)
    counts["abnormal"] = counts["low"] + counts["high"]
    return counts


def group_by_category(findings) -> dict:
    grouped: dict[str, list] = {}
    for f in findings:
        grouped.setdefault(f.category, []).append(f)
    for items in grouped.values():
        order = {"high": 0, "low": 1, "unknown": 2, "normal": 3}
        items.sort(key=lambda f: (order.get(f.status, 9), f.display_name))
    return grouped


def results_rows(findings) -> list[dict]:
    """Flat rows for a results table."""
    rows = []
    for f in findings:
        rows.append({
            "Category": f.category,
            "Test": f.display_name,
            "Result": f"{f.value:g} {f.unit}".strip(),
            "Interval used": f.range_text(),
            "Status": STATUS_STYLE[f.status]["label"],
            "Interval from": ORIGIN_LABEL.get(f.ref_origin, f.ref_origin),
        })
    return rows


def position_in_range(finding):
    """Where a value sits in its interval, as a fraction from 0 to 1.

    Returns None when the interval is open ended or absent, because there is
    then no meaningful position to draw.
    """
    lo, hi = finding.ref_low, finding.ref_high
    if lo is None or hi is None or hi <= lo:
        return None
    return (finding.value - lo) / (hi - lo)


def range_bar_data(findings) -> list[dict]:
    """Data for the per-parameter interval chart.

    Values are scaled so every parameter can be drawn on one axis: the interval
    becomes 0 to 1 and the value is placed relative to it. Values outside the
    interval are clamped for drawing but the real number is kept for the label.
    """
    out = []
    for f in findings:
        pos = position_in_range(f)
        if pos is None:
            continue
        out.append({
            "test": f.display_name,
            "category": f.category,
            "status": f.status,
            "value": f.value,
            "unit": f.unit,
            "position": max(-0.35, min(1.35, pos)),
            "clipped": not (-0.35 <= pos <= 1.35),
            "low": f.ref_low,
            "high": f.ref_high,
            "color": STATUS_STYLE[f.status]["color"],
            "label": f"{f.value:g} {f.unit} (interval {f.ref_low:g}-{f.ref_high:g})",
        })
    out.sort(key=lambda d: (d["category"], d["test"]))
    return out


def category_summary(findings) -> list[dict]:
    rows = []
    for category, items in group_by_category(findings).items():
        counts = status_counts(items)
        rows.append({
            "category": category,
            "total": counts["total"],
            "abnormal": counts["abnormal"],
            "low": counts["low"],
            "high": counts["high"],
            "normal": counts["normal"],
        })
    rows.sort(key=lambda r: (-r["abnormal"], r["category"]))
    return rows


def review_warnings(result) -> list[str]:
    """Plain warnings to show above the review grid."""
    notes = []
    uncertain = [r for r in result.readings if r.needs_review]
    if uncertain:
        notes.append(f"{len(uncertain)} of {len(result.readings)} values need checking "
                     f"before analysis.")
    for r in result.readings:
        for issue in r.issues:
            notes.append(f"{r.raw_name or r.parameter}: {issue}")
    if result.unmatched_lines:
        notes.append(f"{len(result.unmatched_lines)} lines held numbers but were not "
                     f"recognised as tests. Add anything important by hand.")
    if not result.sex:
        notes.append("No sex was found on the report. Some intervals differ by sex, so "
                     "set it in the sidebar for a correct comparison.")
    return notes


def headline(findings) -> str:
    c = status_counts(findings)
    if c["total"] == 0:
        return "No results were read from this file."
    if c["abnormal"] == 0:
        return f"All {c['total']} results sit inside the interval used."
    return (f"{c['abnormal']} of {c['total']} results sit outside the interval used "
            f"({c['low']} low, {c['high']} high).")

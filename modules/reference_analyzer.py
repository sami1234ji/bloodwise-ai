"""Reference intervals and the low / normal / high decision.

Order of precedence, which the interface must show to the user:

1. The interval printed on the uploaded report, when one was extracted.
2. A verified row in data/reference_ranges.csv.

A parameter whose row is marked `unverified` is reported with its value and no
judgement. Reference intervals differ between laboratories, methods, ages and
sexes, so a table interval is never presented as universal.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

LOW, NORMAL, HIGH, UNKNOWN = "low", "normal", "high", "unknown"


def _num(text):
    text = (text or "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass
class ReferenceRow:
    parameter: str
    display_name: str
    category: str
    aliases: list
    unit: str
    low: float | None
    high: float | None
    sex: str
    source_name: str
    source_url: str
    date_accessed: str
    verification_status: str
    source_text: str

    @property
    def verified(self) -> bool:
        return self.verification_status == "verified"


@dataclass
class Finding:
    """One parameter after comparison with a reference interval."""
    parameter: str
    display_name: str
    category: str
    value: float
    unit: str
    status: str
    ref_low: float | None = None
    ref_high: float | None = None
    ref_origin: str = "none"          # "report" | "reference_table" | "none"
    ref_source_name: str = ""
    ref_source_url: str = ""
    note: str = ""
    issues: list = field(default_factory=list)

    @property
    def abnormal(self) -> bool:
        return self.status in (LOW, HIGH)

    def range_text(self) -> str:
        if self.ref_low is not None and self.ref_high is not None:
            return f"{_fmt(self.ref_low)} - {_fmt(self.ref_high)} {self.unit}"
        if self.ref_high is not None:
            return f"below {_fmt(self.ref_high)} {self.unit}"
        if self.ref_low is not None:
            return f"above {_fmt(self.ref_low)} {self.unit}"
        return "no reference available"

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(x: float) -> str:
    if x is None:
        return ""
    return f"{x:g}"


class ReferenceTable:
    """The reference intervals in data/reference_ranges.csv."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DATA_DIR / "reference_ranges.csv"
        self.rows: list[ReferenceRow] = []
        self._load()

    def _load(self) -> None:
        with self.path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append(ReferenceRow(
                    parameter=r["parameter"],
                    display_name=r["display_name"],
                    category=r["category"],
                    aliases=[a.strip().lower() for a in r["aliases"].split("|") if a.strip()],
                    unit=r["unit"],
                    low=_num(r["low"]),
                    high=_num(r["high"]),
                    sex=r["sex"] or "any",
                    source_name=r["source_name"],
                    source_url=r["source_url"],
                    date_accessed=r["date_accessed"],
                    verification_status=r["verification_status"],
                    source_text=r["source_text"],
                ))

    # -- lookups ----------------------------------------------------------
    @property
    def parameters(self) -> list[str]:
        seen, out = set(), []
        for r in self.rows:
            if r.parameter not in seen:
                seen.add(r.parameter)
                out.append(r.parameter)
        return out

    def rows_for(self, parameter: str) -> list[ReferenceRow]:
        return [r for r in self.rows if r.parameter == parameter]

    def row(self, parameter: str, sex: str | None = None) -> ReferenceRow | None:
        """The row for a parameter, preferring one that matches the given sex."""
        rows = self.rows_for(parameter)
        if not rows:
            return None
        sex = (sex or "any").lower()
        for r in rows:
            if r.sex == sex:
                return r
        for r in rows:
            if r.sex == "any":
                return r
        return rows[0]

    def meta(self, parameter: str) -> ReferenceRow | None:
        """Any row for the parameter, for its display name, unit and category."""
        rows = self.rows_for(parameter)
        return rows[0] if rows else None

    def alias_index(self) -> dict:
        """Lowercased alias -> parameter id, longest alias first when matching."""
        index: dict[str, str] = {}
        for r in self.rows:
            for a in r.aliases:
                index.setdefault(a, r.parameter)
            index.setdefault(r.display_name.lower(), r.parameter)
            index.setdefault(r.parameter.replace("_", " "), r.parameter)
        return index

    def categories(self) -> list[str]:
        out = []
        for r in self.rows:
            if r.category not in out:
                out.append(r.category)
        return out

    def unit_for(self, parameter: str) -> str:
        m = self.meta(parameter)
        return m.unit if m else ""


def classify(value: float, low: float | None, high: float | None) -> str:
    """Compare a value with an interval where either end may be absent."""
    if low is None and high is None:
        return UNKNOWN
    if low is not None and value < low:
        return LOW
    if high is not None and value > high:
        return HIGH
    return NORMAL


def analyse(extracted: Iterable, table: ReferenceTable, sex: str | None = None) -> list[Finding]:
    """Turn extracted measurements into findings.

    `extracted` items need: parameter, value, unit, ref_low, ref_high, issues.
    An item whose parameter is unknown is skipped; the caller reports those
    separately so nothing silently disappears.
    """
    findings: list[Finding] = []
    for item in extracted:
        pid = getattr(item, "parameter", None) or (item.get("parameter") if isinstance(item, dict) else None)
        if not pid:
            continue
        get = (lambda k, d=None: getattr(item, k, d)) if not isinstance(item, dict) else (lambda k, d=None: item.get(k, d))
        meta = table.meta(pid)
        if meta is None:
            continue
        value = float(get("value"))
        unit = get("unit") or meta.unit
        rep_low, rep_high = get("ref_low"), get("ref_high")

        if rep_low is not None or rep_high is not None:
            f = Finding(
                parameter=pid, display_name=meta.display_name, category=meta.category,
                value=value, unit=unit,
                status=classify(value, rep_low, rep_high),
                ref_low=rep_low, ref_high=rep_high, ref_origin="report",
                ref_source_name="Interval printed on the uploaded report",
                ref_source_url="",
                note="Compared with the interval printed on your own report.",
                issues=list(get("issues", []) or []),
            )
        else:
            row = table.row(pid, sex)
            if row is None or not row.verified:
                f = Finding(
                    parameter=pid, display_name=meta.display_name, category=meta.category,
                    value=value, unit=unit, status=UNKNOWN, ref_origin="none",
                    note=(row.source_text if row else
                          "No verified reference interval is held for this parameter."),
                    issues=list(get("issues", []) or []),
                )
            else:
                sex_note = "" if row.sex == "any" else f" ({row.sex} interval)"
                f = Finding(
                    parameter=pid, display_name=meta.display_name, category=meta.category,
                    value=value, unit=unit,
                    status=classify(value, row.low, row.high),
                    ref_low=row.low, ref_high=row.high, ref_origin="reference_table",
                    ref_source_name=row.source_name, ref_source_url=row.source_url,
                    note=(f"No interval was found on your report, so the stored reference "
                          f"interval was used{sex_note}. Intervals differ between "
                          f"laboratories; check the one printed on your own report."),
                    issues=list(get("issues", []) or []),
                )
        findings.append(f)
    return findings


def summarise(findings: list[Finding]) -> dict:
    return {
        "total": len(findings),
        "low": sum(1 for f in findings if f.status == LOW),
        "high": sum(1 for f in findings if f.status == HIGH),
        "normal": sum(1 for f in findings if f.status == NORMAL),
        "unknown": sum(1 for f in findings if f.status == UNKNOWN),
        "from_report_range": sum(1 for f in findings if f.ref_origin == "report"),
    }

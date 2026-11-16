"""Find parameters, values, units and printed reference intervals in report text.

Nothing here is accepted silently. Every reading carries a confidence and a list
of issues, and the application shows the user an editable table before any
analysis runs.

The patient's name is deliberately never extracted. Only age and sex are read,
because sex decides which reference interval applies and age decides whether the
adult intervals in this project are appropriate at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .data_normalizer import UnitConverter, canon_unit, is_implausible
from .reference_analyzer import ReferenceTable

NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"

UNIT_TOKENS = [
    "mL/min/1.73m2", "ml/min/1.73 m2", "ml/min/1.73m²", "mL/min",
    "10^3/uL", "10^3/µL", "10^3/μL", "10e3/uL", "x10^3/uL", "10*3/uL",
    "thou/uL", "thousand/uL", "K/uL", "k/ul",
    "10^6/uL", "10^6/µL", "10e6/uL", "x10^6/uL", "10*6/uL",
    "mill/uL", "million/uL", "M/uL", "millions/cumm", "mill/cumm",
    "cells/cumm", "cells/uL", "/cumm", "/uL", "cu mm", "cumm",
    "mg/dL", "mgs/dL", "mg/L", "g/dL", "gm/dL", "gms/dL", "g/L",
    "ug/dL", "mcg/dL", "µg/dL", "μg/dL", "ug/L", "mcg/L", "µg/L",
    "ng/mL", "ng/dL", "pg/mL", "pg",
    "mmol/L", "umol/L", "µmol/L", "μmol/L", "nmol/L",
    "uIU/mL", "µIU/mL", "μIU/mL", "mIU/L", "mIU/mL", "uU/mL",
    "IU/L", "U/L", "IU/mL", "fL", "fl", "%",
]
UNIT_RE = re.compile(
    "(" + "|".join(re.escape(u) for u in sorted(UNIT_TOKENS, key=len, reverse=True)) + ")",
    re.IGNORECASE)

RANGE_RE = re.compile(rf"({NUM})\s*(?:-|–|—|to)\s*({NUM})", re.IGNORECASE)
UPPER_RE = re.compile(rf"(?:<|less\s+than|upto|up\s+to|below)\s*({NUM})", re.IGNORECASE)
LOWER_RE = re.compile(rf"(?:>|greater\s+than|more\s+than|above)\s*({NUM})", re.IGNORECASE)
NUM_RE = re.compile(NUM)

SEX_RE = re.compile(r"\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f)\b", re.IGNORECASE)
AGE_RE = re.compile(r"\bage\s*[:\-]?\s*(\d{1,3})\s*(?:y|yr|yrs|years)?\b", re.IGNORECASE)


HEADER_WORDS = {"test", "tests", "result", "results", "unit", "units", "value",
                "values", "reference", "references", "range", "ranges", "normal",
                "interval", "intervals", "observed", "method", "parameter",
                "investigation", "bio. ref. interval", "ref. range", "ref range"}

# A cell that can belong to a record: a number, a unit, an interval, or a short
# qualifier such as "Low" or "H". Anything longer is the start of something else.
_VALUE_CELL = re.compile(
    r"^[\s<>=~+-]*(?:\d[\d,.\s]*)?(?:\s*(?:-|–|—|to)\s*\d[\d,.]*)?[\s]*$")


def _is_table_header(line: str) -> bool:
    return line.strip().lower().strip(":") in HEADER_WORDS


def _is_value_cell(line: str) -> bool:
    s = line.strip()
    if len(s) > 28:
        return False
    if _VALUE_CELL.match(s):
        return True
    # A bare unit, or a unit with an interval, counts as part of the record.
    without_units = UNIT_RE.sub(" ", s)
    return bool(UNIT_RE.search(s)) and _VALUE_CELL.match(without_units) is not None


_LEADING_PAREN = re.compile(r"^\s*\(([^)]*)\)")


def _strip_name_annotation(tail: str) -> str:
    """Drop a bracketed note that follows the test name.

    Reports write things like "ALT (SGPT) 26 U/L" and "Vitamin D (25-OH) 14".
    The bracketed part names the test, so its digits must not be read as the
    result. A bracket holding only numbers is left alone, because that is how
    some reports print the reference interval.
    """
    m = _LEADING_PAREN.match(tail)
    if m and re.search(r"[A-Za-z]", m.group(1)):
        return tail[m.end():]
    return tail


def _f(text: str) -> float:
    return float(text.replace(",", ""))


def _sig(x: float, digits: int = 4) -> float:
    """Round a converted number so it is not shown with false precision."""
    if x is None or x == 0 or not isinstance(x, (int, float)):
        return x
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + (digits - 1))


@dataclass
class Reading:
    parameter: str | None
    raw_name: str
    value: float
    unit: str | None
    ref_low: float | None = None
    ref_high: float | None = None
    confidence: float = 0.0
    source_line: str = ""
    issues: list = field(default_factory=list)
    conversion_note: str = ""

    @property
    def needs_review(self) -> bool:
        return self.confidence < 0.75 or bool(self.issues)

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter, "raw_name": self.raw_name,
            "value": self.value, "unit": self.unit,
            "ref_low": self.ref_low, "ref_high": self.ref_high,
            "confidence": round(self.confidence, 2),
            "issues": list(self.issues), "source_line": self.source_line,
            "conversion_note": self.conversion_note,
        }


@dataclass
class ExtractionResult:
    readings: list = field(default_factory=list)
    unmatched_lines: list = field(default_factory=list)
    sex: str | None = None
    age: int | None = None

    @property
    def low_confidence(self) -> list:
        return [r for r in self.readings if r.needs_review]


class ParameterExtractor:
    def __init__(self, table: ReferenceTable | None = None,
                 converter: UnitConverter | None = None):
        self.table = table or ReferenceTable()
        self.converter = converter or UnitConverter()
        index = self.table.alias_index()
        self._aliases = sorted(index.items(), key=lambda kv: len(kv[0]), reverse=True)
        self._patterns = [(re.compile(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])",
                                      re.IGNORECASE), pid)
                          for a, pid in self._aliases if len(a) >= 2]

    # -- patient context --------------------------------------------------
    def detect_patient(self, text: str):
        sex = age = None
        m = SEX_RE.search(text)
        if m:
            v = m.group(1).lower()
            sex = "male" if v in ("m", "male") else "female"
        m = AGE_RE.search(text)
        if m:
            try:
                a = int(m.group(1))
                age = a if 0 < a < 120 else None
            except ValueError:
                pass
        return sex, age

    # -- line parsing -----------------------------------------------------
    def match_parameter(self, line: str):
        """Longest alias that appears in the line, with where it ends."""
        best = None
        for pat, pid in self._patterns:
            m = pat.search(line)
            if m and (best is None or (m.end() - m.start()) > best[2]):
                best = (pid, m.end(), m.end() - m.start(), m.group(0))
        if best is None:
            return None, 0, ""
        return best[0], best[1], best[3]

    def parse_line(self, line: str, from_ocr: bool = False) -> Reading | None:
        clean = line.strip()
        if not clean or len(clean) > 300:
            return None
        pid, name_end, raw_name = self.match_parameter(clean)
        if pid is None:
            return None

        tail = _strip_name_annotation(clean[name_end:])
        issues: list[str] = []
        confidence = 0.9

        ref_low = ref_high = None
        consumed = []

        m = RANGE_RE.search(tail)
        if m:
            a, b = _f(m.group(1)), _f(m.group(2))
            ref_low, ref_high = (a, b) if a <= b else (b, a)
            consumed.append(m.span())
        else:
            mu = UPPER_RE.search(tail)
            if mu:
                ref_high = _f(mu.group(1))
                consumed.append(mu.span())
            ml = LOWER_RE.search(tail)
            if ml:
                ref_low = _f(ml.group(1))
                consumed.append(ml.span())

        value_zone = tail
        for s, e in sorted(consumed, reverse=True):
            value_zone = value_zone[:s] + " " * (e - s) + value_zone[e:]

        # Units such as 10^3/uL contain digits, so blank the unit tokens before
        # counting how many numbers the line really holds. Blanking keeps the
        # string the same length, so offsets into value_zone stay valid.
        masked = UNIT_RE.sub(lambda m: " " * len(m.group(0)), value_zone)
        nums = list(NUM_RE.finditer(masked))
        if not nums:
            return None
        value = _f(nums[0].group(0))
        if len(nums) > 1:
            issues.append("more than one number on this line; the first was taken as the result")
            confidence -= 0.15

        unit = None
        after = value_zone[nums[0].end():]
        mu = UNIT_RE.search(after[:24])
        if mu:
            unit = mu.group(1)
        else:
            mu = UNIT_RE.search(tail)
            if mu:
                unit = mu.group(1)
                confidence -= 0.05
        if unit is None:
            issues.append("no unit found; the stored reference unit was assumed")
            confidence -= 0.20

        # Unit conversion into the unit the reference table uses.
        ref_unit = self.table.unit_for(pid)
        conversion_note = ""
        if unit and ref_unit:
            new_value, new_unit, note, ok = self.converter.to_reference_unit(
                pid, value, unit, ref_unit)
            if ok:
                if note:
                    conversion_note = note
                    confidence -= 0.05
                    # Any interval printed on the report is in the same unit as
                    # the value, so it has to move with it.
                    factor, _ = self.converter.factor(pid, unit, ref_unit)
                    if factor is not None:
                        if ref_low is not None:
                            ref_low = _sig(ref_low * factor)
                        if ref_high is not None:
                            ref_high = _sig(ref_high * factor)
                value, unit = _sig(new_value), new_unit
            else:
                issues.append(note)
                confidence -= 0.25

        row = self.table.row(pid)
        if row is not None and is_implausible(value, row.low, row.high):
            issues.append("far outside any plausible range for this test; "
                          "check for a misread digit or decimal point")
            confidence -= 0.35

        if from_ocr:
            confidence -= 0.15

        return Reading(parameter=pid, raw_name=raw_name.strip(), value=value, unit=unit,
                       ref_low=ref_low, ref_high=ref_high,
                       confidence=max(0.05, min(1.0, confidence)),
                       source_line=clean, issues=issues,
                       conversion_note=conversion_note)

    # -- whole document ---------------------------------------------------
    def extract(self, text: str, from_ocr: bool = False) -> ExtractionResult:
        """Read every parameter in the document.

        Two layouts have to work. In some PDFs a whole table row comes out as
        one line. In others every cell comes out as its own line, so the test
        name, the value, the unit and the interval arrive one after another.
        When a line names a parameter but carries no number, the following
        lines are joined to it until the record is complete.
        """
        sex, age = self.detect_patient(text)
        result = ExtractionResult(sex=sex, age=age)
        seen: dict[str, Reading] = {}

        def keep(reading: Reading) -> None:
            prev = seen.get(reading.parameter)
            if prev is None or reading.confidence > prev.confidence:
                seen[reading.parameter] = reading

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]

            reading = self.parse_line(line, from_ocr=from_ocr)
            if reading is not None:
                keep(reading)
                i += 1
                continue

            # A test name can itself contain digits ("HbA1c", "Vitamin B12",
            # "Free T4"), so only the text after the name decides whether this
            # line already carries a result.
            pid, name_end, _raw = self.match_parameter(line)
            after_name = _strip_name_annotation(line[name_end:]) if pid else ""
            if pid is not None and not NUM_RE.search(after_name):
                joined, consumed = self._gather(lines, i, pid)
                if joined is not None:
                    reading = self.parse_line(joined, from_ocr=from_ocr)
                    if reading is not None and reading.parameter == pid:
                        keep(reading)
                        i += consumed
                        continue

            if NUM_RE.search(line) and len(line) > 3 and not _is_table_header(line):
                result.unmatched_lines.append(line)
            i += 1

        result.readings = list(seen.values())
        return result

    def _gather(self, lines: list, start: int, pid: str, lookahead: int = 4):
        """Join a parameter-name line with the cells that follow it."""
        buf = [lines[start]]
        best = None
        best_used = 1
        for step in range(1, lookahead + 1):
            j = start + step
            if j >= len(lines):
                break
            nxt = lines[j]
            if _is_table_header(nxt):
                break
            # Stop at the next test name, unless it is only a unit or a number.
            other, _e, _r = self.match_parameter(nxt)
            if other is not None and other != pid and not _is_value_cell(nxt):
                break
            if not _is_value_cell(nxt):
                break
            buf.append(nxt)
            candidate = " ".join(buf)
            parsed = self.parse_line(candidate)
            if parsed is not None and parsed.parameter == pid:
                best, best_used = candidate, step + 1
                if parsed.ref_low is not None or parsed.ref_high is not None:
                    break
        return best, best_used


def to_review_table(result: ExtractionResult, table: ReferenceTable) -> list[dict]:
    """Rows for the on-screen review grid the user confirms before analysis."""
    rows = []
    for r in sorted(result.readings, key=lambda x: x.confidence):
        meta = table.meta(r.parameter)
        rows.append({
            "parameter": r.parameter,
            "Test": meta.display_name if meta else r.raw_name,
            "Value": r.value,
            "Unit": r.unit or (meta.unit if meta else ""),
            "Report low": r.ref_low,
            "Report high": r.ref_high,
            "Confidence": round(r.confidence, 2),
            "Needs review": r.needs_review,
            "Issues": "; ".join(r.issues),
            "Read from": r.source_line,
        })
    return rows


def from_review_table(rows: list[dict]) -> list[Reading]:
    """Turn the confirmed (possibly edited) grid back into readings."""
    out = []
    for row in rows:
        if row.get("Value") in (None, ""):
            continue
        try:
            value = float(row["Value"])
        except (TypeError, ValueError):
            continue
        out.append(Reading(
            parameter=row.get("parameter"),
            raw_name=row.get("Test", ""),
            value=value,
            unit=row.get("Unit") or None,
            ref_low=_opt_float(row.get("Report low")),
            ref_high=_opt_float(row.get("Report high")),
            confidence=1.0,
            source_line=row.get("Read from", ""),
            issues=[],
        ))
    return out


def _opt_float(v):
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

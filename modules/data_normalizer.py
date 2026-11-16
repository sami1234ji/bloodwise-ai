"""Unit handling.

Conversion factors come from data/unit_conversions.csv, which is produced by
scripts/prepare_nhanes.py. Each factor there was derived from two NHANES
variables that report the same measurement in different units; none was taken
from memory. A conversion this project cannot justify is not performed: the
value is kept as it was read and flagged, so the user can see the problem
rather than being shown a silently wrong number.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def canon_unit(unit: str | None) -> str:
    """Normalise how a unit is written, without changing what it means."""
    if not unit:
        return ""
    u = unit.strip().lower()
    for a, b in (("µ", "u"), ("μ", "u"), ("^", ""), (" ", ""), ("·", "/"),
                 ("10e3", "103"), ("x103", "103"), ("10*3", "103"),
                 ("10e6", "106"), ("x106", "106"), ("10*6", "106"),
                 ("percent", "%"), ("iu/l", "u/l"), ("miu/l", "uiu/ml"),
                 ("mlu/ml", "uiu/ml"), ("uu/ml", "uiu/ml")):
        u = u.replace(a, b)
    aliases = {
        "g/dl": "g/dl", "gm/dl": "g/dl", "gms/dl": "g/dl", "g/l": "g/l",
        "mg/dl": "mg/dl", "mgs/dl": "mg/dl", "mg/l": "mg/l",
        "ug/dl": "ug/dl", "mcg/dl": "ug/dl", "ug/l": "ug/l", "mcg/l": "ug/l",
        "ng/ml": "ng/ml", "pg/ml": "pg/ml", "ng/dl": "ng/dl",
        "mmol/l": "mmol/l", "umol/l": "umol/l", "nmol/l": "nmol/l",
        "u/l": "u/l", "fl": "fl", "pg": "pg", "%": "%",
        "103/ul": "103/ul", "thou/ul": "103/ul", "k/ul": "103/ul",
        "thousand/ul": "103/ul", "103/ul": "103/ul",
        "106/ul": "106/ul", "mill/ul": "106/ul", "m/ul": "106/ul",
        "million/ul": "106/ul", "106/ul": "106/ul",
        "cells/ul": "/ul", "/ul": "/ul", "cumm": "/ul", "/cumm": "/ul",
        "uiu/ml": "uiu/ml", "miu/ml": "uiu/ml",
        "ml/min/1.73m2": "ml/min/1.73m2", "ml/min": "ml/min/1.73m2",
    }
    return aliases.get(u, u)


# How the reference table writes each unit, in canonical form.
REF_UNIT_CANON = {
    "g/dL": "g/dl", "mg/dL": "mg/dl", "ug/dL": "ug/dl", "ng/mL": "ng/ml",
    "pg/mL": "pg/ml", "ng/dL": "ng/dl", "U/L": "u/l", "fL": "fl", "pg": "pg",
    "%": "%", "10^3/uL": "103/ul", "million/uL": "106/ul", "uIU/mL": "uiu/ml",
    "mL/min/1.73m2": "ml/min/1.73m2",
}

# Count-based conversions that are a restatement of the same number in a
# different unit, not a chemical conversion. These are exact by definition.
COUNT_FACTORS = {
    ("/ul", "103/ul"): 0.001,
    ("103/ul", "/ul"): 1000.0,
    ("/ul", "106/ul"): 1e-6,
    ("106/ul", "/ul"): 1e6,
    ("g/l", "g/dl"): 0.1,
    ("g/dl", "g/l"): 10.0,
}


@dataclass
class Conversion:
    parameter: str
    from_unit: str
    to_unit: str
    multiply_by: float
    derivation: str
    source_name: str
    source_url: str


class UnitConverter:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DATA_DIR / "unit_conversions.csv"
        self.by_param: dict[str, list[Conversion]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("from_unit") or not r.get("to_unit"):
                    continue
                c = Conversion(
                    parameter=r["parameter"],
                    from_unit=canon_unit(r["from_unit"]),
                    to_unit=canon_unit(r["to_unit"]),
                    multiply_by=float(r["multiply_by"]),
                    derivation=r.get("derivation", ""),
                    source_name=r.get("source_name", ""),
                    source_url=r.get("source_url", ""),
                )
                self.by_param.setdefault(c.parameter, []).append(c)

    def factor(self, parameter: str, from_unit: str, to_unit: str):
        """Return (factor, explanation) or (None, reason it is not possible)."""
        a, b = canon_unit(from_unit), canon_unit(to_unit)
        if not a or not b or a == b:
            return 1.0, "same unit"
        if (a, b) in COUNT_FACTORS:
            return COUNT_FACTORS[(a, b)], "restatement of the same count in a different unit"
        for c in self.by_param.get(parameter, []):
            if (c.from_unit, c.to_unit) == (a, b):
                return c.multiply_by, c.derivation
            if (c.to_unit, c.from_unit) == (a, b) and c.multiply_by:
                return 1.0 / c.multiply_by, "inverse of: " + c.derivation
        return None, (f"no sourced conversion from '{from_unit}' to '{to_unit}' "
                      f"for {parameter}")

    def to_reference_unit(self, parameter: str, value: float, unit: str,
                          ref_unit: str):
        """Convert a reading into the unit the reference table uses.

        Returns (value, unit, note, ok). When no sourced factor exists the
        original value is returned unchanged with ok=False.
        """
        target = canon_unit(REF_UNIT_CANON.get(ref_unit, ref_unit))
        if not unit:
            return value, ref_unit, "no unit on the report; assumed to be " + ref_unit, True
        if canon_unit(unit) == target:
            return value, ref_unit, "", True
        f, why = self.factor(parameter, unit, ref_unit)
        if f is None:
            return value, unit, why, False
        converted = value * f
        note = (f"converted {value:g} {unit} to {converted:g} {ref_unit} "
                f"(x{f:g}; {why})")
        return converted, ref_unit, note, True


def plausibility_bounds(ref_low, ref_high, factor: float = 5.0):
    """A wide sanity window used only to catch misread digits.

    This is not a medical judgement. It flags readings far outside anything the
    reference interval would allow, which in practice means an OCR error such as
    a lost decimal point. The window is deliberately generous.
    """
    lo = (ref_low / factor) if ref_low not in (None, 0) else None
    hi = (ref_high * factor) if ref_high not in (None, 0) else None
    return lo, hi


def is_implausible(value: float, ref_low, ref_high) -> bool:
    lo, hi = plausibility_bounds(ref_low, ref_high)
    if lo is not None and value < lo:
        return True
    if hi is not None and value > hi:
        return True
    return False

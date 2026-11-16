"""Machine learning over a whole panel of results.

What this does and does not do
------------------------------
It looks at the combination of values in one report and says how unusual that
combination is compared with the training population. It can notice a set of
values that are each inside their own reference interval but rarely occur
together, which is the one thing comparing each value with its reference
interval cannot do.

It is **not** disease prediction, and the scores are not probabilities of
illness. They describe the training population only, which is a United States
health survey measured on that survey's analysers. See
docs/dataset_provenance.md.

Two detectors
-------------
`isolation_forest`
    Isolation Forest, as required by the project brief. It splits one parameter
    at a time, so it responds to rows that are extreme in several parameters at
    once.

`mahalanobis`
    Robust Mahalanobis distance from a minimum covariance determinant fit. It
    uses the covariance between parameters, so it responds when parameters that
    normally move together stop agreeing.

Both are scored and either one can raise the flag. reports/ml_evaluation.json
holds the measured detection rates for each, which is what justifies keeping
both rather than only the one the brief named.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "anomaly_model.joblib"
META_FILE = MODEL_DIR / "model_metadata.json"


class ModelNotTrained(RuntimeError):
    pass


def apply_log_transform(frame, log_features):
    """Log transform the strongly skewed columns.

    Several blood measurements are strongly right-skewed: in the training data
    the white cell count has a skew above 50, because a few participants have
    very high counts. Standardising such a column would leave one measured value
    worth dozens of standard deviations. The list of columns is decided during
    training from the training split's skew and stored in the model metadata, so
    exactly the same columns are transformed when scoring.
    """
    out = frame.copy()
    for f in log_features:
        if f in out.columns:
            out[f] = np.log1p(np.clip(out[f].astype(float), 0, None))
    return out


@dataclass
class AnomalyResult:
    available: bool
    is_unusual: bool = False
    mahalanobis: float | None = None
    mahalanobis_threshold: float | None = None
    isolation: float | None = None
    isolation_threshold: float | None = None
    triggered_by: list = field(default_factory=list)
    features_used: int = 0
    features_total: int = 0
    contributors: list = field(default_factory=list)
    message: str = ""
    training_note: str = ""

    # Kept so older callers that read `.score` still work.
    @property
    def score(self):
        return self.mahalanobis

    def to_dict(self) -> dict:
        return {
            "available": self.available, "is_unusual": self.is_unusual,
            "mahalanobis": self.mahalanobis,
            "mahalanobis_threshold": self.mahalanobis_threshold,
            "isolation": self.isolation,
            "isolation_threshold": self.isolation_threshold,
            "triggered_by": self.triggered_by,
            "features_used": self.features_used,
            "features_total": self.features_total,
            "contributors": self.contributors, "message": self.message,
        }


class AnomalyDetector:
    """Loads the trained detectors and scores one panel at a time."""

    def __init__(self, model_path: Path | None = None, meta_path: Path | None = None):
        self.model_path = Path(model_path) if model_path else MODEL_FILE
        self.meta_path = Path(meta_path) if meta_path else META_FILE
        self._bundle = None
        self._meta = None

    @property
    def available(self) -> bool:
        return self.model_path.exists() and self.meta_path.exists()

    def load(self):
        if self._bundle is not None:
            return
        if not self.available:
            raise ModelNotTrained(
                f"No trained model at {self.model_path}. Run scripts/train_model.py.")
        import joblib
        self._bundle = joblib.load(self.model_path)
        self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

    @property
    def features(self) -> list:
        self.load()
        return self._meta["features"]

    @property
    def metadata(self) -> dict:
        self.load()
        return self._meta

    def _row(self, values: dict):
        """Build the feature row, filling anything absent with the training median."""
        self.load()
        feats = self._meta["features"]
        medians = self._meta["medians"]
        log_features = set(self._meta.get("log_features", []))
        row, present = [], []
        for f in feats:
            v = values.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                x = float(medians[f])
            else:
                x = float(v)
                present.append(f)
            if f in log_features:
                x = float(np.log1p(max(x, 0.0)))
            row.append(x)
        # A DataFrame keeps the column names the scaler was fitted with.
        import pandas as pd
        return pd.DataFrame([row], columns=feats), present

    def score_panel(self, values: dict) -> AnomalyResult:
        """Score one report. `values` maps parameter id to value in reference units."""
        if not self.available:
            return AnomalyResult(
                available=False,
                message="The anomaly model has not been trained, so this check is "
                        "unavailable. Run scripts/train_model.py to enable it.")
        self.load()
        feats = self._meta["features"]
        min_needed = self._meta.get("min_features_to_score", 8)

        x, present = self._row(values)
        if len(present) < min_needed:
            return AnomalyResult(
                available=True, features_used=len(present), features_total=len(feats),
                message=(f"Only {len(present)} of the {len(feats)} parameters this model "
                         f"uses were found on the report. At least {min_needed} are needed "
                         f"before an unusual-combination check means anything."))

        scaler = self._bundle["scaler"]
        forest = self._bundle["forest"]
        cov = self._bundle["covariance"]
        xs = scaler.transform(x)

        mahal = float(np.sqrt(max(cov.mahalanobis(xs)[0], 0.0)))
        iso = float(-forest.score_samples(xs)[0])
        th_m = float(self._meta["mahalanobis_threshold"])
        th_i = float(self._meta["isolation_threshold"])

        triggered = []
        if mahal > th_m:
            triggered.append("correlation structure (Mahalanobis)")
        if iso > th_i:
            triggered.append("isolation forest")

        # Split the squared Mahalanobis distance across the parameters. The parts
        # add up to the total, so the largest ones are what drove the result.
        delta = (xs - cov.location_.reshape(1, -1)).ravel()
        parts = delta * (cov.get_precision() @ delta)
        contributors = []
        present_set = set(present)
        for f, d, p in zip(feats, delta, parts):
            if f in present_set:
                contributors.append({
                    "parameter": f,
                    "contribution": round(float(p), 3),
                    "direction": "above" if d > 0 else "below",
                    "z": round(float(d), 2),
                })
        contributors.sort(key=lambda c: abs(c["contribution"]), reverse=True)

        unusual = bool(triggered)
        if unusual:
            msg = ("This combination of values is less common in the training population "
                   "than most. That is a statistical observation about a survey "
                   "population, not a diagnosis.")
        else:
            msg = ("This combination of values sits within the common range for the "
                   "training population.")
        if len(present) < len(feats):
            msg += (f" {len(feats) - len(present)} of the {len(feats)} parameters were "
                    f"missing from the report and were filled with the population median, "
                    f"which pulls the result towards 'usual'.")

        return AnomalyResult(
            available=True, is_unusual=unusual,
            mahalanobis=round(mahal, 3), mahalanobis_threshold=round(th_m, 3),
            isolation=round(iso, 4), isolation_threshold=round(th_i, 4),
            triggered_by=triggered,
            features_used=len(present), features_total=len(feats),
            contributors=contributors[:5], message=msg,
            training_note=self._meta.get("training_note", ""))


def values_from_findings(findings) -> dict:
    """Pull a parameter -> value map out of a list of Finding objects."""
    out = {}
    for f in findings:
        pid = getattr(f, "parameter", None) if not isinstance(f, dict) else f.get("parameter")
        val = getattr(f, "value", None) if not isinstance(f, dict) else f.get("value")
        if pid is not None and val is not None:
            out[pid] = float(val)
    return out

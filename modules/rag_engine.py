"""Retrieval over the MedlinePlus knowledge base.

Two backends behind one interface:

* `embeddings` - Sentence Transformers for vectors and FAISS for the index, used
  when both are installed.
* `tfidf` - scikit-learn TF-IDF with cosine similarity, used otherwise.

The TF-IDF backend is not a stub. It is a complete retriever and is the default
when the heavier libraries are not present. Which backend ran is reported in the
interface, so results are never misattributed.

Every passage returned carries the title, URL and access date of the MedlinePlus
page it came from, and explanations shown to the user cite them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge_base"

SECTIONS = (("measures", "what it measures"), ("low", "low results"),
            ("high", "high results"), ("notes", "other factors"))


@dataclass
class Passage:
    parameter: str
    title: str
    section: str
    text: str
    source_url: str
    date_accessed: str
    score: float = 0.0

    def citation(self) -> str:
        return f"{self.title} - MedlinePlus ({self.source_url})"


class KnowledgeBase:
    def __init__(self, kb_dir: Path | None = None):
        self.dir = Path(kb_dir) if kb_dir else KB_DIR
        self.passages: list[Passage] = []
        self._load()

    def _load(self) -> None:
        for path in sorted(self.dir.glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            for key, label in SECTIONS:
                text = (doc.get(key) or "").strip()
                if text:
                    self.passages.append(Passage(
                        parameter=doc["parameter"], title=doc["title"], section=label,
                        text=text, source_url=doc["source_url"],
                        date_accessed=doc["date_accessed"]))

    def for_parameter(self, parameter: str, status: str | None = None) -> list[Passage]:
        """The passages for one parameter, optionally only those matching a status."""
        hits = [p for p in self.passages if p.parameter == parameter]
        if status == "low":
            want = ("what it measures", "low results", "other factors")
        elif status == "high":
            want = ("what it measures", "high results", "other factors")
        else:
            want = ("what it measures",)
        return [p for p in hits if p.section in want]

    def general(self) -> list[Passage]:
        return [p for p in self.passages if p.parameter == "_general"]


class Retriever:
    """Free-text search over the knowledge base."""

    def __init__(self, kb: KnowledgeBase | None = None, prefer_embeddings: bool = True):
        self.kb = kb or KnowledgeBase()
        self.backend = "none"
        self._tfidf = self._matrix = self._st = self._index = None
        if prefer_embeddings and self._try_embeddings():
            self.backend = "embeddings"
        elif self._try_tfidf():
            self.backend = "tfidf"

    @property
    def backend_description(self) -> str:
        return {
            "embeddings": "Sentence Transformers embeddings indexed with FAISS",
            "tfidf": "scikit-learn TF-IDF with cosine similarity",
            "none": "retrieval unavailable",
        }[self.backend]

    def _corpus(self) -> list:
        return [f"{p.title} {p.section} {p.text}" for p in self.kb.passages]

    def _try_embeddings(self) -> bool:
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except Exception:
            return False
        try:
            self._st = SentenceTransformer("all-MiniLM-L6-v2")
            vecs = self._st.encode(self._corpus(), normalize_embeddings=True)
            vecs = np.asarray(vecs, dtype="float32")
            self._index = faiss.IndexFlatIP(vecs.shape[1])
            self._index.add(vecs)
            return True
        except Exception:
            return False

    def _try_tfidf(self) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception:
            return False
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                      sublinear_tf=True)
        self._matrix = self._tfidf.fit_transform(self._corpus())
        return True

    def search(self, query: str, k: int = 4) -> list[Passage]:
        if self.backend == "embeddings":
            import numpy as np
            q = np.asarray(self._st.encode([query], normalize_embeddings=True),
                           dtype="float32")
            scores, idx = self._index.search(q, min(k, len(self.kb.passages)))
            out = []
            for s, i in zip(scores[0], idx[0]):
                p = self.kb.passages[int(i)]
                out.append(Passage(**{**p.__dict__, "score": float(s)}))
            return out
        if self.backend == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity
            qv = self._tfidf.transform([query])
            sims = cosine_similarity(qv, self._matrix)[0]
            order = sims.argsort()[::-1][:k]
            out = []
            for i in order:
                if sims[i] <= 0:
                    continue
                p = self.kb.passages[int(i)]
                out.append(Passage(**{**p.__dict__, "score": float(sims[i])}))
            return out
        return []

    def context_for(self, findings, limit_per_finding: int = 3) -> list[Passage]:
        """Passages relevant to the abnormal findings, deduplicated, in order."""
        chosen: list[Passage] = []
        seen = set()
        for f in findings:
            status = getattr(f, "status", None) or f.get("status")
            pid = getattr(f, "parameter", None) or f.get("parameter")
            if status not in ("low", "high"):
                continue
            direct = self.kb.for_parameter(pid, status)[:limit_per_finding]
            if not direct:
                name = getattr(f, "display_name", None) or f.get("display_name", pid)
                direct = self.search(f"{name} {status} result meaning",
                                     k=limit_per_finding)
            for p in direct:
                key = (p.parameter, p.section)
                if key not in seen:
                    seen.add(key)
                    chosen.append(p)
        return chosen

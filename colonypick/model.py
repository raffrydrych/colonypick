"""Colony taxon classifier (random forest) and its saved bundle."""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureBuilder

OTHER = "__other__"


@dataclass
class ColonyModel:
    """Random forest predicting a colony's taxon (ASV or genus) from morphology."""
    level: str = "asv"                       # "asv" or "genus"
    min_class_count: int = 10                # rarer taxa are merged into OTHER
    n_trees: int = 200
    min_samples_leaf: int = 10               # leaf 10 vs 2: model ~15x smaller (~0.5 GB), picking -0.3 ASV at n=50
    seed: int = 0
    use_context: bool = True
    classes: list = field(default_factory=list)
    builder: FeatureBuilder | None = None
    rf: object = None
    info: dict = field(default_factory=dict)
    taxon_info: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- labels
    def encode(self, labels: pd.Series, classes=None) -> np.ndarray:
        classes = classes if classes is not None else self.classes
        known = set(classes) - {OTHER}
        return np.array([v if v in known else OTHER for v in labels.fillna(OTHER).astype(str)])

    def _make_rf(self, seed):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)

    def _choose_classes(self, labels: pd.Series):
        counts = labels.value_counts()
        return sorted(set(counts[counts >= self.min_class_count].index) - {OTHER}) + [OTHER]

    # ---------------------------------------------------------------- training
    def fit(self, df: pd.DataFrame, ctx: pd.DataFrame | None = None, log=print) -> "ColonyModel":
        labels = df[self.level].astype(str)
        self.classes = self._choose_classes(labels)
        self.builder = FeatureBuilder(self.use_context).fit(df, ctx, log=log)
        X = self.builder.transform(df, ctx)
        y = self.encode(labels)
        self.rf = self._make_rf(self.seed).fit(X, y)
        self.taxon_info = self._taxon_info(df)
        self.info = {"trained": _dt.datetime.now().isoformat(timespec="seconds"), "n_isolates": int(len(df)),
                     "n_donors": int(df["donor"].nunique()), "n_classes": len(self.classes), "level": self.level,
                     "context_correction": bool(self.builder.context_coef), "media": self.builder.media}
        log(f"trained random forest: {len(df)} isolates, {len(self.classes) - 1} {self.level} classes + other")
        return self

    def _taxon_info(self, df: pd.DataFrame) -> dict:
        """Per class: most common genus and (for ASVs) the representative V4 sequence."""
        info = {}
        for c in self.classes:
            if c == OTHER:
                continue
            sub = df[df[self.level].astype(str) == c]
            d = {"n_isolates": int(len(sub))}
            if "genus" in sub and sub["genus"].notna().any():
                d["genus"] = str(sub["genus"].mode().iat[0])
            if self.level == "asv" and "asv_sequence" in sub and sub["asv_sequence"].notna().any():
                d["sequence"] = str(sub["asv_sequence"].mode().iat[0])
            info[c] = d
        return info

    def taxon_label(self, c: str) -> str:
        """Human-readable class name, e.g. 'ASV8 (Bacteroides)'."""
        g = getattr(self, "taxon_info", {}).get(c, {}).get("genus")
        return f"{c} ({g})" if g and self.level == "asv" else c

    def resolve_taxa(self, entries) -> list:
        """Map user-given taxa to model classes. An entry may be a class name ("ASV8"), a label as
        written in picks.csv ("ASV8 (Bacteroides)"), a genus (all classes of that genus) or an exact
        16S V4 sequence."""
        out = set()
        for e in entries or []:
            e = str(e).strip()
            if not e or e.startswith("#"):
                continue
            name = e.split(" (")[0]
            hits = [c for c in self.classes if c == name]
            hits += [c for c, d in self.taxon_info.items()
                     if d.get("genus", "").lower() == e.lower() or d.get("sequence") == e.upper()]
            if not hits:
                print(f"WARNING: excluded taxon '{e}' does not match any model class")
            out.update(hits)
        return sorted(out)

    def oof_predict(self, df: pd.DataFrame, ctx: pd.DataFrame | None = None, n_splits: int = 5, log=print,
                    train_frac: float = 1.0):
        """Out-of-fold probabilities with folds grouped by donor (each isolate is predicted by a model
        that never saw its donor). Returns (P, classes, y).

        `train_frac` < 1 trains each fold on a random fraction of its training isolates (learning curves)."""
        from sklearn.model_selection import GroupKFold
        labels = df[self.level].astype(str)
        classes = self._choose_classes(labels)
        y = self.encode(labels, classes)
        cidx = {c: i for i, c in enumerate(classes)}
        donors = df["donor"].to_numpy()
        n_splits = min(n_splits, len(np.unique(donors)))
        P = np.zeros((len(df), len(classes)))
        sub_rng = np.random.default_rng(self.seed + 12345)
        for k, (tr, te) in enumerate(GroupKFold(n_splits).split(np.zeros(len(df)), y, donors)):
            if train_frac < 1.0:
                tr = np.sort(sub_rng.choice(tr, max(50, int(round(train_frac * len(tr)))), replace=False))
            fb = FeatureBuilder(self.use_context).fit(df.iloc[tr], None if ctx is None else ctx.iloc[tr], log=lambda *_: None)
            rf = self._make_rf(self.seed + k).fit(fb.transform(df.iloc[tr], None if ctx is None else ctx.iloc[tr]), y[tr])
            pr = rf.predict_proba(fb.transform(df.iloc[te], None if ctx is None else ctx.iloc[te]))
            P[np.ix_(te, [cidx[c] for c in rf.classes_])] = pr
            log(f"  fold {k + 1}/{n_splits}: held-out donors {sorted(set(donors[te]))}")
        top1 = float((np.array(classes)[P.argmax(1)] == y).mean())
        log(f"out-of-fold top-1 accuracy ({self.level}): {top1:.3f}")
        return P, classes, y

    # ---------------------------------------------------------------- prediction
    def predict_proba(self, colonies: pd.DataFrame, ctx: pd.DataFrame | None = None, media=None) -> np.ndarray:
        X = self.builder.transform(colonies, ctx, media=media)
        pr = self.rf.predict_proba(X)
        P = np.zeros((len(colonies), len(self.classes)))
        cidx = {c: i for i, c in enumerate(self.classes)}
        P[:, [cidx[c] for c in self.rf.classes_]] = pr
        return P

    # ---------------------------------------------------------------- persistence
    def save(self, path):
        import joblib
        path = Path(path)
        joblib.dump(self, path, compress=3)
        path.with_suffix(".json").write_text(json.dumps({**self.info, "taxa": self.taxon_info}, indent=2))

    @staticmethod
    def load(path) -> "ColonyModel":
        import joblib
        return joblib.load(path)

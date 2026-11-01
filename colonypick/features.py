"""Classifier input features.

Per colony:
  * the 14 CAMII morphological features, with log(1 + x) for the size features;
  * optional plate-context correction: log size features are regressed on
    log(1 + neighbours within 50 px) and log(1 + colonies on the plate) and replaced by their
    residuals (+ mean). Crowded colonies grow smaller, so this removes part of the plate effect;
  * one-hot growth medium.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .detection import FEATURES

LOG_FEATURES = ["Area", "Perimeter", "Radius"]
NEIGHBOR_RADIUS_PX = 50


def log_features(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].to_numpy(dtype=float).copy()
    for i, f in enumerate(FEATURES):
        if f in LOG_FEATURES:
            X[:, i] = np.log1p(np.clip(X[:, i], 0, None))
    return X


def plate_context(colonies: pd.DataFrame, all_colonies: pd.DataFrame,
                  radius: float = NEIGHBOR_RADIUS_PX) -> pd.DataFrame:
    """Neighbour count (within `radius` px, excluding the colony itself) and colonies per plate.

    `colonies` and `all_colonies` need columns plate (or plate_barcode), X, Y in the same
    (cropped-image) coordinates."""
    pc = "plate" if "plate" in colonies else "plate_barcode"
    out = pd.DataFrame({"n_neighbors": np.nan, "plate_count": np.nan}, index=colonies.index)
    all_colonies = all_colonies.dropna(subset=["X", "Y"])
    for plate, sub in all_colonies.groupby("plate_barcode"):
        idx = colonies.index[(colonies[pc] == plate) & colonies["X"].notna()]
        if len(idx) == 0:
            continue
        tree = cKDTree(sub[["X", "Y"]].to_numpy(float))
        cnt = tree.query_ball_point(colonies.loc[idx, ["X", "Y"]].to_numpy(float), r=radius, return_length=True)
        out.loc[idx, "n_neighbors"] = np.maximum(np.asarray(cnt) - 1, 0)
        out.loc[idx, "plate_count"] = len(sub)
    return out


class FeatureBuilder:
    """Fits the context correction and the medium vocabulary on training data and applies both
    identically to new colonies."""

    def __init__(self, use_context: bool = True):
        self.use_context = use_context
        self.context_coef = None      # {feature: (intercept, b_neighbors, b_density, mean)}
        self.media = []               # medium names seen in training (lower case)

    def _context_matrix(self, ctx: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.log1p(ctx["n_neighbors"].to_numpy(float)),
                                np.log1p(ctx["plate_count"].to_numpy(float))])

    def fit(self, df: pd.DataFrame, ctx: pd.DataFrame | None, log=print) -> "FeatureBuilder":
        self.media = sorted(df["media"].dropna().astype(str).str.lower().unique().tolist())
        self.context_coef = None
        if self.use_context and ctx is not None and ctx["n_neighbors"].notna().mean() > 0.5:
            from sklearn.linear_model import LinearRegression
            X = log_features(df)
            ok = ctx["n_neighbors"].notna().to_numpy() & ctx["plate_count"].notna().to_numpy()
            Z = self._context_matrix(ctx)[ok]
            self.context_coef = {}
            for i, f in enumerate(FEATURES):
                if f in LOG_FEATURES:
                    lr = LinearRegression().fit(Z, X[ok, i])
                    self.context_coef[f] = (float(lr.intercept_), float(lr.coef_[0]), float(lr.coef_[1]),
                                            float(X[ok, i].mean()))
                    log(f"  context correction {f}: neighbours {lr.coef_[0]:+.3f}, plate density "
                        f"{lr.coef_[1]:+.3f}, R2 {lr.score(Z, X[ok, i]):.3f}")
        return self

    def transform(self, df: pd.DataFrame, ctx: pd.DataFrame | None = None, media=None) -> np.ndarray:
        X = log_features(df)
        if self.context_coef and ctx is not None:
            ok = ctx["n_neighbors"].notna().to_numpy() & ctx["plate_count"].notna().to_numpy()
            Z = self._context_matrix(ctx)
            for i, f in enumerate(FEATURES):
                if f in self.context_coef:
                    a, b1, b2, mu = self.context_coef[f]
                    X[ok, i] = X[ok, i] - (a + Z[ok, 0] * b1 + Z[ok, 1] * b2) + mu
        if media is None:
            media = df["media"] if "media" in df else pd.Series([""] * len(df), index=df.index)
        elif isinstance(media, str):
            media = pd.Series([media] * len(df), index=df.index)
        m = pd.Series(media, index=df.index).astype(str).str.strip().str.lower()
        M = np.column_stack([(m == level).to_numpy(float) for level in self.media]) if self.media \
            else np.zeros((len(df), 0))
        return np.hstack([X, M])

    @property
    def feature_names(self):
        return [f"log_{f}" if f in LOG_FEATURES else f for f in FEATURES] + [f"media={m}" for m in self.media]

"""Smoke tests on synthetic data (no CAMII data needed): python -m pytest tests"""
import numpy as np
import pandas as pd
import pytest

from colonypick.detection import FEATURES
from colonypick.features import plate_context
from colonypick.model import ColonyModel, OTHER
from colonypick.picking import expected_novelty_order, camii_maxmin_select


def synthetic_isolates(n_taxa=6, per_taxon=40, donors=("D1", "D2", "D3"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_taxa):
        centre = rng.normal(0, 1, len(FEATURES))
        for i in range(per_taxon):
            f = centre + rng.normal(0, 0.3, len(FEATURES))
            row = dict(zip(FEATURES, f))
            row.update(Area=float(np.exp(5 + f[0])), Perimeter=float(np.exp(4 + f[1])), Radius=float(np.exp(2 + f[2])),
                       asv=f"ASV{t + 1}", genus=f"Genus{t // 2}", donor=donors[i % len(donors)],
                       plate=f"P{i % 4}", X=float(rng.uniform(0, 1600)), Y=float(rng.uniform(0, 1000)),
                       media="mgam", group=f"{donors[i % len(donors)]}|mgam", is_random=True)
            rows.append(row)
    return pd.DataFrame(rows)


def test_expected_novelty_prefers_new_taxa():
    classes = ["A", "B", "C", OTHER]
    P = np.array([[0.9, 0.05, 0.05, 0], [0.85, 0.1, 0.05, 0], [0.05, 0.9, 0.05, 0], [0.05, 0.05, 0.9, 0]])
    order, gains = expected_novelty_order(P, classes, 3)
    assert set(order) == {0, 2, 3} or set(order) == {1, 2, 3}
    assert np.all(np.diff(gains) <= 1e-12)


def test_exclude_taxa():
    classes = ["A", "B", OTHER]
    P = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    order, _ = expected_novelty_order(P, classes, 1, exclude_taxa=["A"])
    assert order[0] == 1


def test_camii_maxmin_returns_unique_indices():
    X = np.random.default_rng(1).normal(size=(50, len(FEATURES)))
    sel = camii_maxmin_select(X, 10, seed=0)
    assert len(sel) == 10 and len(set(sel)) == 10


def test_plate_context_counts_neighbours():
    col = pd.DataFrame({"plate_barcode": ["P"] * 3, "X": [0, 10, 500], "Y": [0, 0, 0]})
    ctx = plate_context(col, col, radius=50)
    assert ctx["n_neighbors"].tolist() == [1, 1, 0]
    assert ctx["plate_count"].tolist() == [3, 3, 3]


@pytest.mark.parametrize("level", ["asv", "genus"])
def test_model_fit_predict_save(tmp_path, level):
    df = synthetic_isolates()
    m = ColonyModel(level=level, n_trees=30).fit(df, None, log=lambda *a: None)
    P = m.predict_proba(df.head(20), None, media="mGAM")
    assert P.shape == (20, len(m.classes))
    assert np.allclose(P.sum(1), 1)
    m.save(tmp_path / "m.joblib")
    m2 = ColonyModel.load(tmp_path / "m.joblib")
    assert np.allclose(m2.predict_proba(df.head(5), None, media="mgam"), m.predict_proba(df.head(5), None, media="mgam"))
    P_oof, classes, y = m.oof_predict(df, None, n_splits=3, log=lambda *a: None)
    assert (np.array(classes)[P_oof.argmax(1)] == y).mean() > 0.5   # well-separated synthetic taxa


def test_resolve_taxa():
    df = synthetic_isolates()
    m = ColonyModel(level="asv", n_trees=10).fit(df, None, log=lambda *a: None)
    assert m.resolve_taxa(["ASV1 (Genus0)"]) == ["ASV1"]
    assert m.resolve_taxa(["genus0"]) == ["ASV1", "ASV2"]


def test_hybrid_tail_fills_to_n():
    from colonypick.picking import hybrid_order
    df = synthetic_isolates()
    m = ColonyModel(level="asv", n_trees=20).fit(df, None, log=lambda *a: None)
    P = m.predict_proba(df, None, media="mgam")
    X = np.random.default_rng(0).normal(size=(len(df), 5))
    for tail in ("none", "maxmin", "anomaly", "anomaly_entropy"):
        order, gains, k = hybrid_order(P, m.classes, X, 60, tail=tail, switch_gain=0.5)
        assert len(order) == 60 and len(set(order)) == 60
        assert tail == "none" or k < 60


def test_classifier_metrics_and_facility_location():
    from colonypick.benchmark import classifier_metrics, facility_location_order
    df = synthetic_isolates()
    m = ColonyModel(level="asv", n_trees=20).fit(df, None, log=lambda *a: None)
    P = m.predict_proba(df, None, media="mgam")
    met = classifier_metrics(P, m.classes, m.encode(df["asv"]))
    assert 0 <= met["top-1 accuracy"] <= 1 and met["log-loss"] > 0 and 0.5 < met["ROC-AUC macro (OvR)"] <= 1
    order = facility_location_order(np.random.default_rng(0).normal(size=(50, 3)), 10)
    assert len(set(order)) == 10

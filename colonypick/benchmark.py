"""Full benchmark of every approach, with all metrics, written to an HTML report.

What is evaluated (all on the public CAMII data; classifier outputs are out-of-fold with folds grouped
by donor, so every colony is scored by a model that never saw its donor):

1. Classifiers (ASV with/without crowding correction, genus): accuracy (top-1, top-5, balanced),
   F1 (macro, weighted), log-loss, Brier score, ROC-AUC and PR-AUC (one-vs-rest, macro), calibration
   error (ECE).
2. Representations: ROC-AUC of "same taxon vs different taxon" for colony pairs within a picking group
   (the decision a diversity picker actually faces).
3. Anomaly detectors (Isolation Forest, LOF, kNN distance, classifier entropy, p(other)) as detectors
   of rare taxa: ROC-AUC, PR-AUC, precision / recall / F1 when the top 10 % are flagged.
4. Picking strategies, at two scales: pools of 200 colonies (up to 100 picks) and whole groups (90 %
   subsamples, up to 500 picks). Distinct ASVs, genera and rare ASVs vs number picked, with 95 %
   bootstrap intervals, normalised area under the curve, differences vs CAMII and groups won.
5. Learning curves: classifier metrics (training and held-out) and picking quality vs the number of
   training isolates.

Only randomly picked isolates are an unbiased sample of a plate, so picking is simulated inside pools
drawn from them (four sample x medium groups, three donors).
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .detection import FEATURES
from .features import FeatureBuilder, log_features
from .model import ColonyModel, OTHER
from .picking import (_zscore, anomaly_scores, camii_maxmin_select, entropy_scores, expected_novelty_order,
                      hybrid_order, maxmin_select, tail_fill)

SMALL_GRID = [5, 10, 20, 30, 50, 75, 100]
LARGE_GRID = [50, 100, 150, 200, 300, 400, 500]
METRICS = {"asv": "distinct ASVs", "genus": "distinct genera", "rare": "distinct rare ASVs"}

COLORS = {
    "random": "#7f7f7f", "oracle": "#000000", "CAMII (PCA2 + max-min)": "#d62728",
    "max-min (14 features)": "#9467bd", "facility location (14 features)": "#8c564b",
    "anomaly only (Isolation Forest x distance)": "#17becf",
    "expected novelty (ASV RF)": "#1f77b4", "expected novelty (ASV RF, no crowding corr.)": "#6baed6",
    "expected novelty (genus RF)": "#2ca02c",
}


# =====================================================================================
# 1. classifier metrics
# =====================================================================================
def classifier_metrics(P: np.ndarray, classes: list, y: np.ndarray) -> dict:
    """Multi-class metrics for probability matrix P (rows = colonies) and true labels y."""
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, log_loss,
                                 roc_auc_score)
    K = len(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    yi = np.array([cidx[v] for v in y])
    Q = np.clip(P, 1e-15, 1)
    Q = Q / Q.sum(1, keepdims=True)
    pred = Q.argmax(1)
    top5 = np.argsort(-Q, 1)[:, :5]
    onehot = np.zeros_like(Q)
    onehot[np.arange(len(yi)), yi] = 1
    aucs, aps, weights = [], [], []
    for k in range(K):
        pos = yi == k
        if 0 < pos.sum() < len(yi):
            aucs.append(roc_auc_score(pos, Q[:, k]))
            aps.append(average_precision_score(pos, Q[:, k]))
            weights.append(pos.sum())
    conf = Q.max(1)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == yi[m]).mean() - conf[m].mean())
    return {
        "top-1 accuracy": float((pred == yi).mean()),
        "top-5 accuracy": float((top5 == yi[:, None]).any(1).mean()),
        "balanced accuracy": float(balanced_accuracy_score(yi, pred)),
        "F1 macro": float(f1_score(yi, pred, average="macro", labels=range(K), zero_division=0)),
        "F1 weighted": float(f1_score(yi, pred, average="weighted", labels=range(K), zero_division=0)),
        "log-loss": float(log_loss(yi, Q, labels=range(K))),
        "Brier score": float(((Q - onehot) ** 2).sum(1).mean()),
        "ROC-AUC macro (OvR)": float(np.mean(aucs)),
        "ROC-AUC weighted (OvR)": float(np.average(aucs, weights=weights)),
        "PR-AUC macro (OvR)": float(np.mean(aps)),
        "ECE (calibration error)": float(ece),
        "classes": K,
        "chance top-1 (majority class)": float(np.bincount(yi, minlength=K).max() / len(yi)),
    }


# =====================================================================================
# 2. pairwise "same taxon?" AUC of representations
# =====================================================================================
def pairwise_auc(df, reps: dict, levels=("asv", "genus"), max_pairs: int = 100_000, seed: int = 0) -> pd.DataFrame:
    """ROC-AUC of similarity for predicting that two colonies of the same group share a taxon."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    rows = []
    for g, sub in df[df["is_random"]].groupby("group"):
        if len(sub) < 100:
            continue
        idx = sub.index.to_numpy()
        n = len(idx)
        m = min(max_pairs, n * (n - 1) // 2)
        a = rng.integers(0, n, m * 2)
        b = rng.integers(0, n, m * 2)
        keep = a != b
        a, b = a[keep][:m], b[keep][:m]
        for name, fn in reps.items():
            sim = fn(idx, a, b)
            for lev in levels:
                lab = df.loc[idx, lev].astype(str).to_numpy()
                same = lab[a] == lab[b]
                if 0 < same.sum() < len(same):
                    rows.append({"group": g, "representation": name, "level": lev,
                                 "ROC-AUC": roc_auc_score(same, sim), "share of same-taxon pairs": same.mean()})
    return pd.DataFrame(rows)


# =====================================================================================
# 3. anomaly detectors as rare-taxon detectors
# =====================================================================================
def anomaly_metrics(df, P, classes, level="asv", rare_max: int = 2, top_frac: float = 0.10, seed: int = 0):
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    ent = entropy_scores(P, classes)
    p_other = P[:, classes.index(OTHER)] if OTHER in classes else np.zeros(len(P))
    rows = []
    for g, sub in df[df["is_random"]].groupby("group"):
        if len(sub) < 100:
            continue
        idx = sub.index.to_numpy()
        X = _zscore(log_features(sub))
        cnt = sub[level].astype(str).map(sub[level].astype(str).value_counts()).to_numpy()
        target = (cnt <= rare_max).astype(int)
        scores = {
            "Isolation Forest": -IsolationForest(n_estimators=300, random_state=seed).fit(X).score_samples(X),
            "LOF (k=20)": -LocalOutlierFactor(20).fit(X).negative_outlier_factor_,
            "kNN distance (k=10)": NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)[0][:, 1:].mean(1),
            "classifier entropy": ent[idx],
            "classifier p(other)": p_other[idx],
        }
        k = max(1, int(round(top_frac * len(idx))))
        for name, s in scores.items():
            flag = np.zeros(len(idx), bool)
            flag[np.argsort(-s)[:k]] = True
            tp = int((flag & (target == 1)).sum())
            prec, rec = tp / k, tp / max(1, target.sum())
            rows.append({"group": g, "detector": name, "ROC-AUC": roc_auc_score(target, s),
                         "PR-AUC": average_precision_score(target, s), "base rate": target.mean(),
                         f"precision@{int(top_frac * 100)}%": prec, f"recall@{int(top_frac * 100)}%": rec,
                         f"F1@{int(top_frac * 100)}%": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})
    return pd.DataFrame(rows)


# =====================================================================================
# 4. picking strategies
# =====================================================================================
def facility_location_order(X, n):
    """Greedy facility location: maximise sum_i max_{j picked} similarity(i, j) (RBF kernel)."""
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    sigma = np.median(D[D > 0]) if (D > 0).any() else 1.0
    S = np.exp(-(D ** 2) / (2 * sigma ** 2))
    cur = np.zeros(len(X))
    order = []
    avail = np.ones(len(X), bool)
    for _ in range(min(n, len(X))):
        gain = np.maximum(S - cur[:, None], 0).sum(0)
        gain[~avail] = -np.inf
        j = int(np.argmax(gain))
        order.append(j)
        avail[j] = False
        cur = np.maximum(cur, S[:, j])
    return order


def _score(sel, asv, gen, rare):
    s = set(asv[sel])
    return len(s), len(set(gen[sel])), len(s & rare)


def simulate_picking(df, probs: dict, morph_log, morph_ctx, scenario: str, n_pools: int, pool_size=None,
                     pool_frac=None, grid=None, seed: int = 0, rare_max: int = 2, log=print) -> pd.DataFrame:
    """Run every strategy in every pool. probs: name -> (P, classes) for the novelty strategies."""
    rng = np.random.default_rng(seed)
    raw = df[FEATURES].to_numpy(float)
    rows = []
    main_P, main_classes = probs["expected novelty (ASV RF)"]
    for g, sub in df[df["is_random"]].groupby("group"):
        if len(sub) < 100:
            continue
        counts = sub["asv"].astype(str).value_counts()
        rare = set(counts[counts <= rare_max].index)
        t0 = time.time()
        for r in range(n_pools):
            size = pool_size if pool_size else int(round(pool_frac * len(sub)))
            size = min(size, len(sub))
            idx = np.sort(rng.choice(sub.index.to_numpy(), size, replace=False))
            asv = df.loc[idx, "asv"].astype(str).to_numpy()
            gen = df.loc[idx, "genus"].astype(str).to_numpy()
            g_grid = [n for n in grid if n <= len(idx)]
            nmax = max(g_grid)
            Xl, Xc = _zscore(morph_log[idx]), _zscore(morph_ctx[idx])
            nested = {}
            for name, (P, classes) in probs.items():
                nested[name] = expected_novelty_order(P[idx], classes, nmax)[0]
            nested["facility location (14 features)"] = facility_location_order(Xl, nmax)
            nested["anomaly only (Isolation Forest x distance)"] = tail_fill(
                Xc, [], nmax, weights=anomaly_scores(Xc, seed=r).argsort().argsort() / len(idx) + 1 / len(idx))
            for sg in (0.3, 0.1):
                for t in ("maxmin", "anomaly", "anomaly_entropy"):
                    nested[f"novelty + {t} tail @{sg:g}"] = hybrid_order(
                        main_P[idx], main_classes, Xc, nmax, tail=t, switch_gain=sg, seed=r)[0]
            from scipy.spatial.distance import pdist, squareform
            D14 = squareform(pdist(Xl))
            for n in g_grid:
                base = {"scenario": scenario, "group": g, "pool": r + 1, "n_picked": n, "pool_size": len(idx)}
                perms = [rng.permutation(len(idx))[:n] for _ in range(20)]
                sc = np.mean([_score(p, asv, gen, rare) for p in perms], 0)
                rows.append({**base, "strategy": "random", "asv": sc[0], "genus": sc[1], "rare": sc[2]})
                rows.append({**base, "strategy": "oracle", "asv": min(n, len(set(asv))),
                             "genus": min(n, len(set(gen))), "rare": min(n, len(set(asv) & rare))})
                sel = camii_maxmin_select(raw[idx], n, seed=int(rng.integers(1e9)))
                rows.append({**base, "strategy": "CAMII (PCA2 + max-min)", **dict(zip(("asv", "genus", "rare"), _score(sel, asv, gen, rare)))})
                sel = maxmin_select(D14, n, rng)
                rows.append({**base, "strategy": "max-min (14 features)", **dict(zip(("asv", "genus", "rare"), _score(sel, asv, gen, rare)))})
                for name, order in nested.items():
                    rows.append({**base, "strategy": name, **dict(zip(("asv", "genus", "rare"), _score(order[:n], asv, gen, rare)))})
        log(f"  [{scenario}] {g}: {n_pools} pools of {size} colonies ({time.time() - t0:.0f} s)")
    return pd.DataFrame(rows)


def _boot_mean(values, groups, B=1000, seed=0):
    """Mean and 95 % interval, bootstrap resampling pools within each group (stratified)."""
    rng = np.random.default_rng(seed)
    values, groups = np.asarray(values, float), np.asarray(groups)
    parts = [values[groups == g] for g in np.unique(groups)]
    boots = np.empty(B)
    for b in range(B):
        boots[b] = np.concatenate([p[rng.integers(0, len(p), len(p))] for p in parts]).mean()
    return values.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def summarise_picking(curves: pd.DataFrame, key_ns) -> dict:
    """Mean curves with CIs, normalised AUC, differences vs CAMII and group wins."""
    out = {}
    strategies = list(dict.fromkeys(curves["strategy"]))
    # mean + CI per strategy, n, metric
    ci_rows = []
    for (strat, n), sub in curves.groupby(["strategy", "n_picked"], sort=False):
        for m in METRICS:
            mu, lo, hi = _boot_mean(sub[m], sub["group"])
            ci_rows.append({"strategy": strat, "n_picked": n, "metric": m, "mean": mu, "ci_low": lo, "ci_high": hi})
    out["ci"] = pd.DataFrame(ci_rows)
    # normalised area under the curve, per pool: AUC(strategy) / AUC(oracle)
    auc_rows = []
    for (g, pool), sub in curves.groupby(["group", "pool"]):
        orc = sub[sub.strategy == "oracle"].sort_values("n_picked")
        for strat, s in sub.groupby("strategy"):
            s = s.sort_values("n_picked")
            row = {"group": g, "pool": pool, "strategy": strat}
            for m in METRICS:
                denom = np.trapezoid(orc[m], orc["n_picked"]) if hasattr(np, "trapezoid") else np.trapz(orc[m], orc["n_picked"])
                num = np.trapezoid(s[m], s["n_picked"]) if hasattr(np, "trapezoid") else np.trapz(s[m], s["n_picked"])
                row[m] = num / denom if denom else np.nan
            auc_rows.append(row)
    auc = pd.DataFrame(auc_rows)
    out["auc_per_pool"] = auc
    # ranking table per metric
    tables = {}
    for m in METRICS:
        rows = []
        cam = curves[curves.strategy == "CAMII (PCA2 + max-min)"].set_index(["group", "pool", "n_picked"])[m]
        rnd = curves[curves.strategy == "random"].set_index(["group", "pool", "n_picked"])[m]
        cam_auc = auc[auc.strategy == "CAMII (PCA2 + max-min)"].set_index(["group", "pool"])[m]
        for strat in strategies:
            a = auc[auc.strategy == strat]
            mu, lo, hi = _boot_mean(a[m], a["group"])
            row = {"strategy": strat, "AUC / oracle": mu, "AUC 95% CI": f"{lo:.3f}–{hi:.3f}"}
            if strat != "CAMII (PCA2 + max-min)":
                d = a.set_index(["group", "pool"])[m] - cam_auc
                dmu, dlo, dhi = _boot_mean(d.values, d.index.get_level_values(0))
                row["ΔAUC vs CAMII [95% CI]"] = f"{dmu:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
            s = curves[curves.strategy == strat].set_index(["group", "pool", "n_picked"])[m]
            for n in key_ns:
                sn = s.xs(n, level="n_picked")
                row[f"n={n}"] = sn.mean()
                if strat not in ("CAMII (PCA2 + max-min)",):
                    d = sn - cam.xs(n, level="n_picked")
                    dmu, dlo, dhi = _boot_mean(d.values, d.index.get_level_values(0))
                    row[f"Δ vs CAMII n={n} [95% CI]"] = f"{dmu:+.2f} [{dlo:+.2f}, {dhi:+.2f}]"
                    row[f"% vs random n={n}"] = 100 * (sn.mean() / rnd.xs(n, level="n_picked").mean() - 1)
            gm = a.groupby("group")[m].mean()
            cm = cam_auc.groupby(level=0).mean()
            row["groups beating CAMII (AUC)"] = ("–" if strat == "CAMII (PCA2 + max-min)"
                                                 else f"{int((gm > cm.reindex(gm.index)).sum())}/{len(gm)}")
            rows.append(row)
        tables[m] = pd.DataFrame(rows).sort_values("AUC / oracle", ascending=False).reset_index(drop=True)
    out["tables"] = tables
    out["per_group_auc"] = auc.pivot_table(index="strategy", columns="group", values="asv").round(3).rename_axis(columns=None)
    return out


def winner_statement(curves, auc, metric="asv", label="") -> tuple[str, str]:
    """Best strategy by normalised AUC (oracle excluded), compared with the best *different* strategy
    (variants with identical picks, e.g. a tail that never triggered, are reported as identical) and with
    CAMII. 'Clear winner' = 95 % CI of the paired difference excludes 0 and ahead in all or all-but-one groups."""
    a = auc[auc.strategy != "oracle"]
    wide = a.pivot_table(index=["group", "pool"], columns="strategy", values=metric)
    ranking = wide.mean().sort_values(ascending=False)
    best = ranking.index[0]
    same = [s for s in ranking.index[1:] if np.allclose(wide[s], wide[best])]
    others = [s for s in ranking.index[1:] if s not in same]
    ng = a["group"].nunique()

    def compare(other):
        d = wide[best] - wide[other]
        mu, lo, hi = _boot_mean(d.values, d.index.get_level_values(0))
        gw = int((wide[best].groupby(level=0).mean() > wide[other].groupby(level=0).mean()).sum())
        return mu, lo, hi, gw

    second = others[0]
    mu, lo, hi, gw = compare(second)
    clear = lo > 0 and gw >= ng - 1
    verdict = "clear winner" if clear else ("ahead, but not clearly" if lo > 0 else "statistically tied with the runner-up")
    txt = (f"{label}: <b>{best}</b> ranks first (AUC/oracle {ranking[best]:.3f})")
    if same:
        txt += f" (identical picks: {', '.join(same)})"
    txt += (f". Runner-up <b>{second}</b> ({ranking[second]:.3f}): difference {mu:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}], "
            f"better in {gw}/{ng} groups &rarr; <b>{verdict}</b>.")
    info = {"best": best, "second": second, "tied": not lo > 0, "clear": clear}
    cam = "CAMII (PCA2 + max-min)"
    if best != cam and cam in wide:
        mu, lo, hi, gw = compare(cam)
        txt += f" Versus CAMII: {mu:+.3f} [{lo:+.3f}, {hi:+.3f}], better in {gw}/{ng} groups."
        info["beats_camii"] = lo > 0
    return info, txt


# =====================================================================================
# 5. learning curves
# =====================================================================================
def learning_curve(df, ctx, fracs, pools_small, log=print, seed=0, **model_kwargs):
    """Classifier metrics (training and held-out) and picking quality vs number of training isolates."""
    from sklearn.model_selection import GroupKFold
    model = ColonyModel(level="asv", seed=seed, **model_kwargs)
    labels = df["asv"].astype(str)
    classes = model._choose_classes(labels)
    y = model.encode(labels, classes)
    cidx = {c: i for i, c in enumerate(classes)}
    donors = df["donor"].to_numpy()
    folds = list(GroupKFold(5).split(np.zeros(len(df)), y, donors))
    rows = []
    for f in fracs:
        rng = np.random.default_rng(seed + 1)
        P = np.zeros((len(df), len(classes)))
        tr_metrics, n_train = [], []
        for k, (tr, te) in enumerate(folds):
            if f < 1:
                tr = np.sort(rng.choice(tr, max(200, int(round(f * len(tr)))), replace=False))
            c_tr = None if ctx is None else ctx.iloc[tr]
            fb = FeatureBuilder(model.use_context).fit(df.iloc[tr], c_tr, log=lambda *a: None)
            Xtr = fb.transform(df.iloc[tr], c_tr)
            rf = model._make_rf(seed + k).fit(Xtr, y[tr])
            cols = [cidx[c] for c in rf.classes_]
            P[np.ix_(te, cols)] = rf.predict_proba(fb.transform(df.iloc[te], None if ctx is None else ctx.iloc[te]))
            Ptr = np.zeros((len(tr), len(classes)))
            Ptr[:, cols] = rf.predict_proba(Xtr)
            tr_metrics.append(classifier_metrics(Ptr, classes, y[tr]))
            n_train.append(len(tr))
        held = classifier_metrics(P, classes, y)
        train = {k: np.mean([m[k] for m in tr_metrics]) for k in tr_metrics[0]}
        # picking quality with these probabilities (pools of 200, n = 50)
        sim = _novelty_only_curve(df, P, classes, pools_small, seed=seed)
        row = {"train fraction": f, "training isolates (mean per fold)": int(np.mean(n_train))}
        for k in ("top-1 accuracy", "top-5 accuracy", "F1 macro", "log-loss", "ROC-AUC macro (OvR)", "PR-AUC macro (OvR)"):
            row[f"train {k}"] = train[k]
            row[f"held-out {k}"] = held[k]
        row["picking: distinct ASVs at n=50"] = sim
        rows.append(row)
        log(f"  learning curve: {row['training isolates (mean per fold)']} training isolates -> held-out top-1 "
            f"{held['top-1 accuracy']:.3f}, log-loss {held['log-loss']:.2f}, ASVs@50 {sim:.2f}")
    return pd.DataFrame(rows)


def _novelty_only_curve(df, P, classes, n_pools, pool_size=200, n=50, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for g, sub in df[df["is_random"]].groupby("group"):
        if len(sub) < 100:
            continue
        for _ in range(n_pools):
            idx = np.sort(rng.choice(sub.index.to_numpy(), min(pool_size, len(sub)), replace=False))
            order = expected_novelty_order(P[idx], classes, n)[0]
            vals.append(len(set(df.loc[idx[order], "asv"].astype(str))))
    return float(np.mean(vals))


# =====================================================================================
# report
# =====================================================================================
def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def _plot_curves(ci, strategies, title, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (m, lab) in zip(axes, METRICS.items()):
        for strat in strategies:
            s = ci[(ci.strategy == strat) & (ci.metric == m)].sort_values("n_picked")
            if s.empty:
                continue
            base = strat.split(" @")[0]
            color = COLORS.get(strat) or {"novelty + maxmin tail": "#bcbd22", "novelty + anomaly tail": "#ff7f0e",
                                          "novelty + anomaly_entropy tail": "#e377c2"}.get(base, None)
            ls = "--" if strat == "random" else (":" if strat == "oracle" or strat.endswith("@0.1") else "-")
            lw = 2.6 if strat == "expected novelty (ASV RF)" else 1.4
            ax.plot(s.n_picked, s["mean"], ls=ls, lw=lw, marker="o", ms=3, color=color, label=strat)
            if strat != "oracle":
                ax.fill_between(s.n_picked, s.ci_low, s.ci_high, color=color, alpha=0.12, lw=0)
        ax.set_xlabel("colonies picked")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return b64


def _plot_learning(lc, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = lc["training isolates (mean per fold)"]
    panels = [("top-1 accuracy", False), ("F1 macro", False), ("log-loss", False), ("ROC-AUC macro (OvR)", False)]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    for ax, (k, _) in zip(axes, panels):
        ax.plot(x, lc[f"train {k}"], marker="o", label="training (in-sample)", color="#ff7f0e")
        ax.plot(x, lc[f"held-out {k}"], marker="o", label="held-out donors", color="#1f77b4", lw=2.2)
        ax.set_xscale("log")
        ax.set_xlabel("training isolates")
        ax.set_title(k)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[4].plot(x, lc["picking: distinct ASVs at n=50"], marker="o", color="#1f77b4", lw=2.2, label="expected novelty")
    for name, val, c in (("CAMII", lc.attrs.get("camii50"), "#d62728"), ("random", lc.attrs.get("random50"), "#7f7f7f")):
        if val is not None:
            axes[4].axhline(val, color=c, ls="--", label=name)
    axes[4].set_xscale("log")
    axes[4].set_xlabel("training isolates")
    axes[4].set_title("picking: distinct ASVs at n = 50 (pool 200)")
    axes[4].legend(fontsize=8)
    axes[4].grid(alpha=0.3)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return b64


def _table_html(df, floatfmt="{:.3f}"):
    def f(v, ff=floatfmt):
        if isinstance(v, (float, np.floating)):
            return ff.format(v) if np.isfinite(v) else ""
        return str(v)
    fmt = {c: f for c in df.columns}
    return df.to_html(index=False, escape=False, formatters=fmt, border=0, classes="t")


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;color:#222;line-height:1.45}
h1{margin-bottom:4px} h2{margin-top:36px;border-bottom:2px solid #ddd;padding-bottom:4px} h3{margin-top:24px}
.verdict{background:#f3f8ff;border-left:5px solid #1f77b4;padding:12px 16px;margin:12px 0}
.note{color:#555;font-size:0.92em}
table.t{border-collapse:collapse;font-size:0.86em;margin:8px 0 16px} table.t th,table.t td{padding:4px 8px;border-bottom:1px solid #e3e3e3;text-align:right;white-space:nowrap}
table.t th{background:#fafafa} table.t td:first-child,table.t th:first-child{text-align:left}
img{max-width:100%} .scroll{overflow-x:auto}
"""


def run_benchmark(df, ctx, out_dir, n_pools_small: int = 20, n_pools_large: int = 10,
                  learning_fracs=(0.05, 0.1, 0.25, 0.5, 1.0), quick: bool = False, seed: int = 0, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if quick:
        n_pools_small, n_pools_large, learning_fracs = 3, 1, (0.1, 1.0)

    # ---------------------------------------------------------------- classifiers
    log("1/5 classifiers (out-of-fold, folds grouped by donor)")
    variants = {"expected novelty (ASV RF)": ("asv", ctx)}
    if ctx is not None:
        variants["expected novelty (ASV RF, no crowding corr.)"] = ("asv", None)
    variants["expected novelty (genus RF)"] = ("genus", ctx)
    probs, clf_rows = {}, []
    for name, (lev, c) in variants.items():
        m = ColonyModel(level=lev, seed=seed)
        P, classes, y = m.oof_predict(df, c, log=lambda *a: None)
        probs[name] = (P, classes)
        met = classifier_metrics(P, classes, y)
        clf_rows.append({"classifier": name.replace("expected novelty ", "").strip("()"), "level": lev,
                         "crowding correction": c is not None, **met})
        log(f"  {name}: top-1 {met['top-1 accuracy']:.3f}, F1 macro {met['F1 macro']:.3f}, "
            f"log-loss {met['log-loss']:.2f}, ROC-AUC {met['ROC-AUC macro (OvR)']:.3f}")
    clf = pd.DataFrame(clf_rows)
    clf.to_csv(out / "classifier_metrics.csv", index=False)
    P_main, cls_main = probs["expected novelty (ASV RF)"]

    # ---------------------------------------------------------------- representations
    log("2/5 representations: same-vs-different taxon AUC")
    morph_log = log_features(df)
    morph_ctx = FeatureBuilder(use_context=True).fit(df, ctx, log=lambda *a: None).transform(df, ctx)[:, :len(FEATURES)]
    raw = df[FEATURES].to_numpy(float)

    def dist_rep(M, pca2=False):
        def fn(idx, a, b):
            Z = _zscore(M[idx])
            if pca2:
                from sklearn.decomposition import PCA
                Z = PCA(2).fit_transform(Z)
            return -np.sqrt(((Z[a] - Z[b]) ** 2).sum(1))
        return fn

    def prob_rep(P):
        return lambda idx, a, b: (P[idx][a] * P[idx][b]).sum(1)

    reps = {"CAMII: PCA2 of z-scored features": dist_rep(raw, pca2=True),
            "14 features (log, z-scored)": dist_rep(morph_log),
            "14 features + crowding correction": dist_rep(morph_ctx),
            "classifier probabilities (ASV RF)": prob_rep(P_main),
            "classifier probabilities (genus RF)": prob_rep(probs["expected novelty (genus RF)"][0])}
    pw = pairwise_auc(df, reps, seed=seed)
    pw.to_csv(out / "pairwise_auc_per_group.csv", index=False)
    pw_tab = pw.pivot_table(index="representation", columns="level", values="ROC-AUC", aggfunc="mean").reindex(list(reps))
    pw_tab.columns = [f"ROC-AUC same {c.upper() if c == 'asv' else c}" for c in pw_tab.columns]
    pw_rng = pw.groupby(["representation", "level"])["ROC-AUC"].agg(["min", "max"]).unstack()
    for lev in ("asv", "genus"):
        pw_tab[f"range over groups ({lev})"] = [f"{pw_rng.loc[r, ('min', lev)]:.3f}–{pw_rng.loc[r, ('max', lev)]:.3f}"
                                                for r in pw_tab.index]
    pw_tab = pw_tab.reset_index()

    # ---------------------------------------------------------------- anomaly detectors
    log("3/5 anomaly detectors as rare-taxon detectors")
    an = anomaly_metrics(df, P_main, cls_main, seed=seed)
    an.to_csv(out / "anomaly_metrics_per_group.csv", index=False)
    an_tab = an.drop(columns="group").groupby("detector", sort=False).mean().reset_index()

    # ---------------------------------------------------------------- picking
    log("4/5 picking simulations")
    small = simulate_picking(df, probs, morph_log, morph_ctx, "pool of 200", n_pools_small, pool_size=200,
                             grid=SMALL_GRID, seed=seed, log=log)
    large = simulate_picking(df, probs, morph_log, morph_ctx, "whole group", n_pools_large, pool_frac=0.9,
                             grid=LARGE_GRID if not quick else [50, 100, 200], seed=seed + 1, log=log)
    small.to_csv(out / "picking_curves_pool200_per_pool.csv", index=False)
    large.to_csv(out / "picking_curves_wholegroup_per_pool.csv", index=False)
    S = summarise_picking(small, key_ns=[20, 50, 100])
    L = summarise_picking(large, key_ns=[n for n in (100, 300, 500) if n in set(large.n_picked)])
    for tag, R in (("pool200", S), ("wholegroup", L)):
        R["ci"].to_csv(out / f"picking_mean_ci_{tag}.csv", index=False)
        for m, t in R["tables"].items():
            t.to_csv(out / f"ranking_{tag}_{m}.csv", index=False)

    # ---------------------------------------------------------------- learning curve
    log("5/5 learning curves")
    lc = learning_curve(df, ctx, learning_fracs, pools_small=max(3, n_pools_small // 4), log=log, seed=seed)
    lc.attrs["camii50"] = S["ci"].query("strategy == 'CAMII (PCA2 + max-min)' and n_picked == 50 and metric == 'asv'")["mean"].iat[0]
    lc.attrs["random50"] = S["ci"].query("strategy == 'random' and n_picked == 50 and metric == 'asv'")["mean"].iat[0]
    lc.to_csv(out / "learning_curve.csv", index=False)

    # ---------------------------------------------------------------- figures + verdict
    core = ["oracle", "random", "CAMII (PCA2 + max-min)", "max-min (14 features)", "facility location (14 features)",
            "anomaly only (Isolation Forest x distance)"] + list(probs)
    tails = ["oracle", "random", "CAMII (PCA2 + max-min)", "expected novelty (ASV RF)"] + \
            [s for s in dict.fromkeys(small.strategy) if s.startswith("novelty +")]
    figs = {
        "s_core": _plot_curves(S["ci"], core, "Pools of 200 colonies: all approaches", out / "fig_pool200_core.png"),
        "s_tail": _plot_curves(S["ci"], tails, "Pools of 200 colonies: novelty + tails", out / "fig_pool200_tails.png"),
        "l_core": _plot_curves(L["ci"], core, "Whole groups: all approaches", out / "fig_wholegroup_core.png"),
        "l_tail": _plot_curves(L["ci"], tails, "Whole groups: novelty + tails", out / "fig_wholegroup_tails.png"),
        "lc": _plot_learning(lc, out / "fig_learning_curves.png"),
    }
    verdicts, infos = [], []
    for R, lab in ((S, "Pools of 200 (up to 100 picks)"), (L, "Whole groups (up to 500 picks)")):
        for m in ("asv", "genus", "rare"):
            info, txt = winner_statement(None, R["auc_per_pool"], m, f"{lab}, {METRICS[m]}")
            verdicts.append(txt)
            infos.append(info)
    learned = lambda x: x.startswith("expected novelty") or x.startswith("novelty +")
    k_learned = sum(learned(i["best"]) for i in infos)
    k_cam = sum(i.get("beats_camii", False) for i in infos)
    k_tied = sum(i["tied"] or (learned(i["best"]) and learned(i["second"]) and not i["clear"]) for i in infos)
    main_wins = sum(i["best"] == "expected novelty (ASV RF)" for i in infos)
    summary = (f"<p><b>Overall.</b> A learned approach (expected novelty, alone or followed by a tail) ranks first in "
               f"{k_learned}/{len(infos)} comparisons (3 metrics &times; 2 scales) and beats CAMII with a 95% interval "
               f"excluding zero in {k_cam}/{len(infos)}. The first place is not clear-cut in {k_tied}/{len(infos)} "
               f"comparisons, where the leader is statistically tied with, or only marginally ahead of, another "
               f"variant. Plain expected novelty (ASV RF) is first in {main_wins}/{len(infos)}. "
               f"Where variants are tied, the simplest one (plain expected novelty) is the recommended default.</p>")

    # ---------------------------------------------------------------- HTML
    def ranking_block(R, title):
        h = f"<h3>{title}</h3>"
        for m, lab in METRICS.items():
            h += f"<h4>{lab}</h4><div class='scroll'>{_table_html(R['tables'][m], '{:.2f}')}</div>"
        return h

    n_groups = df[df["is_random"]].groupby("group").size()
    n_groups = n_groups[n_groups >= 100]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>colonypick benchmark</title><style>{CSS}</style></head><body>
<h1>colonypick: full benchmark</h1>
<p class="note">{time.strftime('%Y-%m-%d %H:%M')} &middot; {len(df)} sequenced isolates, {df['donor'].nunique()} donors,
{df['asv'].nunique()} ASVs, {df['genus'].nunique()} genera &middot; test groups with random picks:
{', '.join(f'{g} ({n})' for g, n in n_groups.items())} &middot; crowding correction: {'yes' if ctx is not None else 'no'}
&middot; run time {(time.time() - t_start) / 60:.0f} min{' &middot; QUICK MODE (few pools, indicative only)' if quick else ''}</p>

<h2>Verdict</h2>
<div class="verdict">{summary}<ul>{''.join('<li>' + v + '</li>' for v in verdicts)}</ul></div>
<p class="note">Ranking by area under the picking curve divided by the oracle's area (1.0 = every pick a new taxon
as long as possible). "Clear winner" = 95% interval of the paired difference to the runner-up excludes 0 and the
winner is ahead in all or all-but-one groups. Intervals come from resampling pools within groups; with only four
groups (three donors) they understate between-sample uncertainty, so the number of groups won matters as much.</p>

<h2>1. Classifiers</h2>
<p>Out-of-fold predictions on held-out donors. Classes: taxa with &ge; 10 isolates plus "other". Higher is better
except log-loss, Brier score and ECE. Accuracy is low because many taxa look alike; for picking what matters is
whether the probabilities rank taxa well (ROC-AUC, PR-AUC, log-loss).</p>
<div class="scroll">{_table_html(clf)}</div>

<h2>2. Representations: can they tell whether two colonies are the same taxon?</h2>
<p>ROC-AUC over random colony pairs within each test group (0.5 = no information). CAMII's 2-D PCA is the
representation CAMII's max-min spreads over.</p>
<div class="scroll">{_table_html(pw_tab)}</div>

<h2>3. Anomaly detectors as rare-taxon detectors</h2>
<p>Target: colonies of ASVs with &le; 2 isolates in their group. ROC-AUC 0.5 = chance; PR-AUC should be compared
with the base rate; precision / recall / F1 when the top 10% most anomalous colonies are flagged.</p>
<div class="scroll">{_table_html(an_tab)}</div>

<h2>4. Picking</h2>
<p>Each strategy picks colonies from the same pools; we count distinct ASVs, distinct genera and distinct rare
ASVs (&le; 2 isolates in the group). Bands = 95% bootstrap intervals.</p>
<h3>Pools of 200 colonies</h3>
<img src="data:image/png;base64,{figs['s_core']}"><img src="data:image/png;base64,{figs['s_tail']}">
{ranking_block(S, 'Ranking, pools of 200')}
<h3>Whole groups (90% subsamples of each group's random picks)</h3>
<img src="data:image/png;base64,{figs['l_core']}"><img src="data:image/png;base64,{figs['l_tail']}">
{ranking_block(L, 'Ranking, whole groups')}
<h3>Per group (AUC/oracle, distinct ASVs, pools of 200)</h3>
<div class="scroll">{_table_html(S['per_group_auc'].reset_index())}</div>
<h3>Per group (AUC/oracle, distinct ASVs, whole groups)</h3>
<div class="scroll">{_table_html(L['per_group_auc'].reset_index())}</div>

<h2>5. Learning curves</h2>
<p>The ASV classifier trained on growing random fractions of each fold's training isolates. Orange = performance
on its own training data, blue = on held-out donors. A widening gap means overfitting; a blue curve still rising at
the right edge means more labelled isolates would help.</p>
<img src="data:image/png;base64,{figs['lc']}">
<div class="scroll">{_table_html(lc)}</div>

<h2>Strategies</h2>
<ul>
<li><b>random</b>: average of 20 random orders. <b>oracle</b>: knows every taxon (upper bound).</li>
<li><b>CAMII (PCA2 + max-min)</b>: original CAMII: z-score, PCA to 2-D, max-min spread (10 restarts).</li>
<li><b>max-min / facility location (14 features)</b>: spread or coverage in the full 14-feature space.</li>
<li><b>anomaly only</b>: Isolation Forest anomaly rank &times; distance to already picked colonies.</li>
<li><b>expected novelty</b>: greedy maximisation of the expected number of distinct taxa from classifier
probabilities (ASV or genus classifier; with or without crowding correction).</li>
<li><b>novelty + tail @s</b>: expected novelty until a pick adds &lt; s expected new taxa, then max-min, anomaly or
anomaly &times; entropy.</li>
</ul>
<p class="note">Image embeddings (DINOv2) and a CNN on colony crops were tested earlier
(<code>experiments/camii_picking_eval.py</code>) and gave no gain beyond noise; they need the raw images and PyTorch
and are not part of this benchmark.</p>
</body></html>"""
    (out / "report.html").write_text(html, encoding="utf-8")
    log(f"\nreport: {out / 'report.html'}  ({(time.time() - t_start) / 60:.1f} min)")
    import re as _re
    for v in [summary] + verdicts:
        log("  " + _re.sub(r"<[^>]+>", "", v).replace("&rarr;", "->").replace("&times;", "x"))
    return out / "report.html"

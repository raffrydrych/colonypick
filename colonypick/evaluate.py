"""Offline evaluation of picking strategies on the public CAMII data.

Only randomly picked isolates are an unbiased sample of the colonies on a plate, so strategies are
compared inside candidate pools drawn from them: from each picking group (sample x medium) with
random picks, `n_pools` pools of `pool_size` colonies are drawn; each strategy picks n colonies
from a pool and we count the distinct taxa among them. Classifier probabilities are out-of-fold
with folds grouped by donor, so a pool is always scored by a model that never saw its donor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from .features import log_features
from .detection import FEATURES
from .model import ColonyModel
from .picking import expected_novelty_order, camii_maxmin_select, maxmin_select, _zscore, hybrid_order

GRID = [5, 10, 20, 30, 50, 75, 100]


def _unique_curve(order, labels, grid):
    seen, cum = set(), []
    for j in order:
        seen.add(labels[j])
        cum.append(len(seen))
    return [cum[n - 1] if n <= len(cum) else np.nan for n in grid]


def evaluate(df: pd.DataFrame, ctx: pd.DataFrame | None, out_dir, level: str = "asv", n_pools: int = 20,
             pool_size: int = 200, random_repeats: int = 50, seed: int = 0, log=print,
             **model_kwargs) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ColonyModel(level=level, seed=seed, **model_kwargs)
    log(f"out-of-fold classifier ({level}), folds grouped by donor:")
    P, classes, _ = model.oof_predict(df, ctx, log=log)

    rng = np.random.default_rng(seed)
    pools = []
    for g, sub in df[df["is_random"]].groupby("group"):
        idx = sub.index.to_numpy()
        if len(idx) < 15:
            continue
        if len(idx) > pool_size:
            pools += [(f"{g}#pool{r + 1}", g, np.sort(rng.choice(idx, pool_size, replace=False)))
                      for r in range(n_pools)]
        else:
            pools.append((g, g, idx))
    log(f"{len(pools)} candidate pools from {df[df['is_random']]['group'].nunique()} groups with random picks")

    raw = df[FEATURES].to_numpy(float)
    logf = log_features(df)
    rows = []
    for name, group, idx in pools:
        labels = df.loc[idx, level].astype(str).to_numpy()
        grid = [n for n in GRID if n <= len(idx)]
        nmax = max(grid)
        res = {
            "random": np.nanmean([_unique_curve(rng.permutation(len(idx)), labels, grid)
                                  for _ in range(random_repeats)], axis=0),
            "oracle": [min(n, len(set(labels))) for n in grid],
            "CAMII (PCA2 + max-min)": [len(set(labels[camii_maxmin_select(raw[idx], n, seed=int(rng.integers(1e9)))]))
                                       for n in grid],
        }
        D = squareform(pdist(_zscore(logf[idx])))
        res["max-min (14 features)"] = [len(set(labels[maxmin_select(D, n, rng)])) for n in grid]
        order, _ = expected_novelty_order(P[idx], classes, nmax)
        res["expected novelty (RF)"] = _unique_curve(order, labels, grid)
        for strat, vals in res.items():
            for n, v in zip(grid, vals):
                rows.append({"pool": name, "group": group, "donor": df.loc[idx[0], "donor"],
                             "strategy": strat, "n_picked": n, "n_unique": v})
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "curves_per_pool.csv", index=False)

    summary = curves.groupby(["strategy", "n_picked"])["n_unique"].mean().unstack("n_picked").round(2)
    summary.to_csv(out / "summary.csv")
    by_group = curves[curves.n_picked == 50].pivot_table(index="group", columns="strategy", values="n_unique").round(2)
    by_group.to_csv(out / "by_group_n50.csv")
    log("\nmean unique taxa per pool:\n" + summary.to_string())
    log("\nper group, n = 50:\n" + by_group.to_string())
    _plot(summary, level, out / "picking_curves.png")
    return summary


def _plot(summary: pd.DataFrame, level: str, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    style = {"oracle": dict(color="black", ls=":"), "random": dict(color="grey", ls="--"),
             "CAMII (PCA2 + max-min)": dict(color="#d62728"), "max-min (14 features)": dict(color="#9467bd"),
             "expected novelty (RF)": dict(color="#1f77b4", lw=2.4)}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strat, row in summary.iterrows():
        ax.plot(row.index, row.values, marker="o", ms=3, label=strat, **style.get(strat, {}))
    ax.set_xlabel("colonies picked from a pool of 200")
    ax.set_ylabel(f"distinct {level.upper() if level == 'asv' else level}s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


TAIL_GRID = [50, 100, 150, 200, 300, 400, 500]


def evaluate_tail(df: pd.DataFrame, ctx: pd.DataFrame | None, out_dir, level: str = "asv", n_pools: int = 10,
                  pool_frac: float = 0.9, n_max: int = 500, switch_gains=(0.3, 0.1), rare_max: int = 2,
                  seed: int = 0, log=print, **model_kwargs) -> pd.DataFrame:
    """Benchmark for LARGE pick numbers, where the classifier runs out of known taxa.

    Pools are whole picking groups with random picks (each pool a random `pool_frac` subsample of
    the group's randomly picked isolates), and up to `n_max` colonies are picked. Compared:
    random, CAMII, expected novelty alone, and expected novelty followed by each tail strategy
    (max-min, Isolation Forest anomaly, anomaly x classifier entropy). Reported: distinct taxa
    and distinct RARE taxa (at most `rare_max` isolates in the whole group) among the picks."""
    from .features import FeatureBuilder
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ColonyModel(level=level, seed=seed, **model_kwargs)
    log(f"out-of-fold classifier ({level}), folds grouped by donor:")
    P, classes, _ = model.oof_predict(df, ctx, log=log)
    # morphology for the tail: log features with the (unsupervised) plate-context correction
    morph = FeatureBuilder(use_context=True).fit(df, ctx, log=lambda *a: None).transform(df, ctx)[:, :len(FEATURES)]
    raw = df[FEATURES].to_numpy(float)
    rng = np.random.default_rng(seed)
    switch_gains = [float(x) for x in np.atleast_1d(switch_gains)]
    tails = ("maxmin", "anomaly", "anomaly_entropy")
    strategies = ["random", "CAMII (PCA2 + max-min)", "expected novelty"] + \
                 [f"novelty + {t} tail @{sg:g}" for sg in switch_gains for t in tails]
    rows, switches = [], []
    for g, sub in df[df["is_random"]].groupby("group"):
        if len(sub) < 100:
            continue
        counts = sub[level].astype(str).value_counts()
        rare_taxa = set(counts[counts <= rare_max].index)
        log(f"group {g}: {len(sub)} random picks, {len(counts)} taxa, {len(rare_taxa)} rare (<= {rare_max} isolates)")
        for r in range(n_pools):
            idx = np.sort(rng.choice(sub.index.to_numpy(), int(round(pool_frac * len(sub))), replace=False))
            labels = df.loc[idx, level].astype(str).to_numpy()
            grid = [n for n in TAIL_GRID if n <= min(n_max, len(idx))]
            nmax = max(grid)
            X = _zscore(morph[idx])
            orders = {"random": None,
                      "CAMII (PCA2 + max-min)": camii_maxmin_select(raw[idx], nmax, seed=int(rng.integers(1e9)))}
            orders["expected novelty"] = hybrid_order(P[idx], classes, X, nmax, tail="none")[0]
            for sg in switch_gains:
                for t in tails:
                    o, _, k = hybrid_order(P[idx], classes, X, nmax, tail=t, switch_gain=sg, seed=r)
                    orders[f"novelty + {t} tail @{sg:g}"] = o
                switches.append({"group": g, "pool": r + 1, "switch_gain": sg, "phase1_picks": k,
                                 "switched": k < nmax})
            for strat in strategies:
                for n in grid:
                    if strat == "random":
                        sel = [rng.permutation(len(idx))[:n] for _ in range(20)]
                        u = np.mean([len(set(labels[s])) for s in sel])
                        ur = np.mean([len(set(labels[s]) & rare_taxa) for s in sel])
                    elif strat.startswith("CAMII"):
                        sel = camii_maxmin_select(raw[idx], n, seed=int(rng.integers(1e9)))
                        u, ur = len(set(labels[sel])), len(set(labels[sel]) & rare_taxa)
                    else:
                        sel = orders[strat][:n]
                        u, ur = len(set(labels[sel])), len(set(labels[sel]) & rare_taxa)
                    rows.append({"group": g, "pool": r + 1, "strategy": strat, "n_picked": n,
                                 "n_unique": u, "n_unique_rare": ur, "pool_size": len(idx),
                                 "taxa_in_pool": len(set(labels)), "rare_in_pool": len(set(labels) & rare_taxa)})
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "tail_curves_per_pool.csv", index=False)
    sw = pd.DataFrame(switches)
    sw.to_csv(out / "tail_switch_points.csv", index=False)
    summ = curves.groupby(["strategy", "n_picked"])["n_unique"].mean().unstack().round(2).reindex(strategies)
    summ_r = curves.groupby(["strategy", "n_picked"])["n_unique_rare"].mean().unstack().round(2).reindex(strategies)
    summ.to_csv(out / "tail_summary_all_taxa.csv")
    summ_r.to_csv(out / "tail_summary_rare_taxa.csv")
    by_group = curves[curves.n_picked == curves.groupby("group").n_picked.transform("max")].pivot_table(
        index="group", columns="strategy", values=["n_unique", "n_unique_rare"]).round(1)
    by_group.to_csv(out / "tail_by_group_at_max_n.csv")
    log("\npicks made by expected novelty before switching to the tail (= largest n: never switched):\n"
        + sw.pivot_table(index="group", columns="switch_gain", values="phase1_picks").round(0).to_string())
    log("\nmean distinct taxa:\n" + summ.to_string())
    log("\nmean distinct RARE taxa (<= %d isolates in the group):\n" % rare_max + summ_r.to_string())
    log("\nper group at the largest n (n_unique = all taxa, n_unique_rare = rare taxa):\n" + by_group.to_string())
    _plot_tail(summ, summ_r, level, out / "tail_curves.png")
    return summ


def _plot_tail(summ, summ_r, level, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    style = {"random": dict(color="grey", ls="--"), "CAMII (PCA2 + max-min)": dict(color="#d62728"),
             "expected novelty": dict(color="#1f77b4", lw=2.2), "novelty + maxmin tail": dict(color="#9467bd"),
             "novelty + anomaly tail": dict(color="#2ca02c", lw=2.2),
             "novelty + anomaly_entropy tail": dict(color="#ff7f0e")}
    fig_strats = list(summ.index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, s, t in zip(axes, (summ, summ_r), (f"distinct {level}s", f"distinct rare {level}s")):
        for strat, row in s.iterrows():
            base, _, sg = strat.partition(" @")
            st = dict(style.get(base, {}))
            if sg and sg != max((x.partition(" @")[2] for x in fig_strats if " @" in x), key=float):
                st["ls"] = ":"   # lower switch threshold dotted
            ax.plot(row.index, row.values, marker="o", ms=3, label=strat, **st)
        ax.set_xlabel("colonies picked from the whole group")
        ax.set_ylabel(t)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

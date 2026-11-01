"""Colony selection strategies.

`expected_novelty_order` is the recommended strategy: it greedily picks the colony that adds the
largest expected number of taxa not yet covered,

    gain(i) = sum_k  p_ik * prod_{j already picked} (1 - p_jk),

where p_ik is the classifier probability that colony i belongs to taxon k (the catch-all
"other" class is ignored). Taxa already isolated before (e.g. in an existing biobank) can be
excluded up front.

`camii_maxmin_select` reproduces the original CAMII strategy (z-score -> PCA to 2D ->
farthest-point max-min with 10 random restarts) for comparison.
"""
from __future__ import annotations

import numpy as np

OTHER = "__other__"


def expected_novelty_order(P: np.ndarray, classes: list, n: int, exclude_taxa=None, weights=None):
    """Return (order, gains): indices of the n colonies to pick, in pick order, and the expected
    number of new taxa each one added at the moment it was picked."""
    K = len(classes)
    w = np.ones(K)
    if weights:
        for k, c in enumerate(classes):
            w[k] = weights.get(c, 1.0)
    if OTHER in classes:
        w[classes.index(OTHER)] = 0.0
    remaining = np.ones(K)
    if exclude_taxa:
        for k, c in enumerate(classes):
            if c in set(exclude_taxa):
                remaining[k] = 0.0
    avail = np.ones(len(P), bool)
    order, gains = [], []
    for _ in range(min(n, len(P))):
        g = (P * w * remaining).sum(1)
        g[~avail] = -np.inf
        j = int(np.argmax(g))
        order.append(j)
        gains.append(float(g[j]))
        avail[j] = False
        remaining *= (1 - P[j])
    return order, gains


def _zscore(X):
    X = np.asarray(X, float)
    sd = X.std(0)
    sd[sd < 1e-12] = 1.0
    return (X - X.mean(0)) / sd


def camii_maxmin_select(features: np.ndarray, n: int, restarts: int = 10, max_iters: int = 50, seed: int = 0):
    """Original CAMII selection: PCA(2) of z-scored features, then farthest-point max-min search."""
    from scipy.spatial.distance import pdist, squareform
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    N = len(features)
    if n >= N:
        return list(range(N))
    Z = PCA(min(2, N, features.shape[1])).fit_transform(_zscore(features))
    D = squareform(pdist(Z))
    return maxmin_select(D, n, rng, restarts, max_iters)


def _top2(D, r, rows=None):
    """Smallest and second-smallest distance from each point (or `rows`) to the selected set
    `r`, with the positions in `r` that give them."""
    sub = D[:, r] if rows is None else D[np.ix_(rows, r)]
    if sub.shape[1] == 1:
        a1 = np.zeros(len(sub), int)
        return sub[:, 0].copy(), a1, np.full(len(sub), np.inf), np.full(len(sub), -1)
    idx = np.argpartition(sub, 1, axis=1)[:, :2]
    v = np.take_along_axis(sub, idx, 1)
    swap = v[:, 1] < v[:, 0]
    idx[swap] = idx[swap][:, ::-1]
    v[swap] = v[swap][:, ::-1]
    return v[:, 0].copy(), idx[:, 0].copy(), v[:, 1].copy(), idx[:, 1].copy()


def maxmin_select(D: np.ndarray, n: int, rng, restarts: int = 10, max_iters: int = 50):
    """Local-search max-min selection on a distance matrix (best of `restarts`).

    Each colony in turn is replaced by the point farthest from all other selected colonies,
    until nothing changes. For every point the two nearest selected colonies are tracked, so one
    replacement costs O(N) instead of O(N * n); results are identical to the direct algorithm."""
    N = D.shape[0]
    if n >= N:
        return list(range(N))
    if n == 1:
        return [int(rng.integers(N))]
    best, best_val = None, -np.inf
    for _ in range(restarts):
        r = np.asarray(rng.choice(N, n, replace=False))
        sel = np.zeros(N, bool)
        sel[r] = True
        d1, a1, d2, a2 = _top2(D, r)
        for _it in range(max_iters):
            changed = False
            for i in range(n):
                md = np.where(a1 == i, d2, d1)          # distance to the selected set without r[i]
                md[sel] = -np.inf
                md[r[i]] = d2[r[i]] if a1[r[i]] == i else d1[r[i]]
                new = int(np.argmax(md))
                if new == r[i]:
                    continue
                changed = True
                sel[r[i]] = False
                r[i] = new
                sel[new] = True
                # points whose nearest or 2nd-nearest selected colony was position i: recompute
                stale = np.flatnonzero((a1 == i) | (a2 == i))
                if len(stale):
                    d1[stale], a1[stale], d2[stale], a2[stale] = _top2(D, r, stale)
                # all other points: the new colony at position i may be nearer
                rest = np.ones(N, bool)
                rest[stale] = False
                dn = D[:, new]
                closer1 = rest & (dn < d1)
                closer2 = rest & ~closer1 & (dn < d2)
                d2[closer1], a2[closer1] = d1[closer1], a1[closer1]
                d1[closer1], a1[closer1] = dn[closer1], i
                d2[closer2], a2[closer2] = dn[closer2], i
            if not changed:
                break
        r = [int(x) for x in r]
        val = D[np.ix_(r, r)][np.triu_indices(n, 1)].min()
        if val > best_val:
            best_val, best = val, list(r)
    return best


# --------------------------------------------------------------------------------------
# Hybrid: expected novelty first, then a morphology-based "tail" for large pick numbers
# --------------------------------------------------------------------------------------
TAILS = ("none", "maxmin", "anomaly", "anomaly_entropy")


def _rank01(s):
    """Scores -> ranks scaled to (0, 1]; makes differently scaled scores comparable."""
    from scipy.stats import rankdata
    return rankdata(s) / len(s)


def anomaly_scores(X, seed: int = 0, n_trees: int = 300):
    """Isolation Forest anomaly score per colony (higher = more unusual within this group)."""
    from sklearn.ensemble import IsolationForest
    return -IsolationForest(n_estimators=n_trees, random_state=seed).fit(X).score_samples(X)


def entropy_scores(P, classes):
    """Entropy of the classifier's distribution over known taxa (higher = less like any known taxon)."""
    keep = [k for k, c in enumerate(classes) if c != OTHER]
    Q = np.clip(P[:, keep], 1e-12, 1)
    Q = Q / Q.sum(1, keepdims=True)
    return -(Q * np.log(Q)).sum(1)


def tail_fill(X, picked, n_total: int, weights=None):
    """Greedy farthest-point fill: repeatedly add the colony maximising
    weight * distance to the nearest already-picked colony, until n_total colonies are picked."""
    X = np.asarray(X, float)
    N = len(X)
    picked = list(picked)
    taken = np.zeros(N, bool)
    taken[picked] = True
    if picked:
        d = np.full(N, np.inf)
        for j in picked:
            d = np.minimum(d, np.sqrt(((X - X[j]) ** 2).sum(1)))
    else:
        d = np.sqrt(((X - X.mean(0)) ** 2).sum(1))
    w = np.ones(N) if weights is None else np.asarray(weights, float)
    out = []
    while len(picked) + len(out) < min(n_total, N):
        v = d * w
        v[taken] = -np.inf
        j = int(np.argmax(v))
        out.append(j)
        taken[j] = True
        d = np.minimum(d, np.sqrt(((X - X[j]) ** 2).sum(1)))
    return out


def hybrid_order(P, classes, X, n: int, tail: str = "anomaly", switch_gain: float = 0.1,
                 exclude_taxa=None, seed: int = 0):
    """Expected novelty while a pick still adds >= `switch_gain` expected new taxa, then fill the
    remaining picks with the chosen tail strategy on morphology features X (z-scored):

      none             expected novelty all the way (default behaviour)
      maxmin           farthest from the already picked colonies (CAMII-like spread)
      anomaly          Isolation Forest anomaly rank x distance to the already picked colonies
      anomaly_entropy  as anomaly, also x rank of classifier entropy (unlike any known taxon)

    Returns (order, gains, n_phase1); gains are NaN for tail picks."""
    if tail not in TAILS:
        raise ValueError(f"tail must be one of {TAILS}")
    order, gains = expected_novelty_order(P, classes, n, exclude_taxa=exclude_taxa)
    if tail == "none":
        return order, gains, len(order)
    below = [i for i, g in enumerate(gains) if g < switch_gain]
    k = below[0] if below else len(order)
    if k >= len(order):
        return order, gains, k
    w = None
    if tail in ("anomaly", "anomaly_entropy"):
        w = _rank01(anomaly_scores(X, seed=seed))
        if tail == "anomaly_entropy":
            w = w * _rank01(entropy_scores(P, classes))
    rest = tail_fill(X, order[:k], n, weights=w)
    return order[:k] + rest, list(gains[:k]) + [np.nan] * len(rest), k

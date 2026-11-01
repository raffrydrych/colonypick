"""Loading the public CAMII datasets into one training table.

Expected files (downloadable from the CAMII data portal, microbial-culturomics.com):
    CAMII_isolates_metadata.tsv            isolateID, sourcePlate, sourceSample, media, isolateStrategy, ...
    CAMII_isolates_morphological_features.tsv
                                           isolateID, X, Y, 14 features, X_rawImage, Y_rawImage
    CAMII_isolates_16SV4_result.tsv        isolateID, taxonomy (domain..species), purity, sequence
    colony_all.metadata.<date>/            optional; one CSV per plate with ALL detected colonies
                                           (X, Y, features, plate_barcode, colony_index)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .detection import FEATURES

MISSING = {"", "na", "nan", "none", "null", "-", "unclassified", "unknown", "n/a"}


def _read(path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab", ".txt") else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _clean(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.where(~s.str.lower().isin(MISSING))


def donor_from_sample(sample: str) -> str | float:
    """'H5t1 (feces from H5)' -> 'H5'; 'H12_F (Pre-selected culture of H12t1)' -> 'H12'; 'H1t4' -> 'H1'."""
    if not isinstance(sample, str):
        return np.nan
    m = re.search(r"\((?:.*?\b(?:from|of)\s+)(.+?)\)\s*$", sample)
    core = m.group(1).strip() if m else re.sub(r"\s*\(.*\)\s*$", "", sample).strip()
    core = re.sub(r"^(?:CRE sample|BA patient|Fstore sample)\s+", "", core, flags=re.I)
    return re.sub(r"t\d+$", "", core.split()[0]) if core else np.nan


def denoise_sequences(seqs: pd.Series, max_mismatch: int = 1, skew: float = 2.0) -> pd.Series:
    """Greedy UNOISE-like clustering: a sequence joins a >= `skew` x more abundant centroid of the
    same length that differs in <= `max_mismatch` positions; otherwise it becomes a new centroid."""
    counts = seqs.dropna().value_counts()
    centroids, mapping = [], {}
    for seq, c in counts.items():
        arr = np.frombuffer(seq.encode(), dtype=np.uint8)
        target = seq
        for cseq, carr, ccount in centroids:
            if ccount >= skew * c and len(carr) == len(arr) and int((carr != arr).sum()) <= max_mismatch:
                target = cseq
                break
        if target == seq:
            centroids.append((seq, arr, c))
        mapping[seq] = target
    return seqs.map(mapping)


def load_camii_isolates(metadata, morphology, s16, min_purity: float = 50.0, denoise: bool = True,
                        log=print) -> pd.DataFrame:
    """One row per sequenced isolate with features, labels (asv, genus), donor, group and plate."""
    meta = _read(metadata)
    mor = _read(morphology)
    seq = _read(s16)
    for d in (meta, mor, seq):
        d["isolateID"] = _clean(d["isolateID"])
    df = meta.merge(mor, on="isolateID", how="inner").merge(seq, on="isolateID", how="inner")
    df = df.drop_duplicates("isolateID").reset_index(drop=True)

    df["purity"] = pd.to_numeric(df["purity"], errors="coerce")
    if min_purity is not None:
        before = len(df)
        df = df[~(df["purity"] < min_purity)].reset_index(drop=True)
        log(f"purity >= {min_purity}%: kept {len(df)} of {before} isolates")

    s = _clean(df["sequence"]).str.upper()
    if denoise:
        n_raw = s.nunique()
        s = denoise_sequences(s)
        log(f"ASV: {n_raw} unique V4 sequences -> {s.nunique()} after denoising")
    codes = {v: f"ASV{i + 1}" for i, v in enumerate(s.dropna().value_counts().index)}
    df["asv"] = s.map(codes)
    df["asv_sequence"] = s
    for k in ("genus", "family", "species"):
        if k in df:
            df[k] = _clean(df[k]).str.replace(r"^[a-z]__", "", regex=True)
    df["genus"] = df["genus"].where(df["genus"].notna(), df.get("family") + "_unclassified")

    df["plate"] = _clean(df["sourcePlate"])
    df["sample"] = _clean(df["sourceSample"]).str.replace(r"\s*\(.*\)\s*$", "", regex=True)
    df["donor"] = _clean(df["sourceSample"]).map(donor_from_sample)
    df["media"] = _clean(df["media"]).str.replace(r"\s+", " ", regex=True).str.lower()
    df["group"] = df["sample"].fillna("NA") + "|" + df["media"].fillna("NA")
    df["is_random"] = df["isolateStrategy"].astype(str).str.contains("random", case=False)
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna(subset=["asv", "donor"]).reset_index(drop=True)
    n_na = int(df[FEATURES].isna().any(axis=1).sum())
    if n_na:
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
        log(f"{n_na} isolates with missing feature values (filled with medians)")
    log(f"isolates: {len(df)} | donors: {df['donor'].nunique()} | ASVs: {df['asv'].nunique()} | "
        f"genera: {df['genus'].nunique()} | random-picked: {int(df['is_random'].sum())}")
    return df


def load_colony_table(path) -> pd.DataFrame:
    """All detected colonies: a folder of per-plate CSVs or a single CSV/TSV."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    col = pd.concat([_read(f) for f in files], ignore_index=True)
    col["plate_barcode"] = _clean(col["plate_barcode"])
    for c in ["X", "Y"] + FEATURES:
        col[c] = pd.to_numeric(col[c], errors="coerce")
    return col.dropna(subset=["plate_barcode", "X", "Y"]).reset_index(drop=True)

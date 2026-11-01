#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camii_picking_eval.py
=====================
Offline evaluation of colony REPRESENTATIONS and colony-PICKING STRATEGIES
on the public CAMII datasets (Huang et al., Nat Biotechnol 2023).

What it does
------------
1. Loads the isolate tables (metadata, morphology, 16S), joins them by isolate ID,
   derives donor / picking group / ASV labels, keeps randomly-picked isolates as an
   unbiased pool for simulation.
2. Builds several colony representations:
     - CAMII_PCA2      : the original method (z-score within group -> PCA to 2D)
     - full14_z        : all 14 features, z-scored within group
     - full14_log_z    : same, with log-transformed size features
     - context_corr_z  : size features corrected for neighbour count / plate density
                         (only if the all-colonies table is provided)
     - embedding       : DINOv2 image embeddings of colony crops (optional, needs torch)
     - clf_prob        : out-of-fold classifier probabilities (random forest,
                         grouped cross-validation by DONOR)
3. Evaluates representations: ROC-AUC of "same vs different ASV" for colony pairs
   within the same picking group (the decision the picker actually faces).
4. Simulates picking strategies inside each group's random pool and draws
   "unique ASVs vs number of picked colonies" curves:
     random, CAMII (PCA2 + max-min), max-min on full features, facility location,
     facility location on classifier similarity, expected-novelty greedy, oracle.

How to use (Visual Studio / VS Code)
------------------------------------
1. pip install numpy pandas scipy scikit-learn matplotlib openpyxl pillow
   (optional, for image embeddings: pip install torch torchvision)
2. Set the paths in the CONFIG section below.
3. Run once with MODE = "inspect" -> check that columns were recognised.
   If something is not found, put the exact column name in COLUMN_OVERRIDES.
4. Set MODE = "run" and run again. Results go to PATHS["output_dir"].

You can also override the mode from the command line:  python camii_picking_eval.py --inspect
"""

# =============================================================================
# CONFIG  - SET YOUR PATHS HERE
# =============================================================================

PATHS = {
    # Metadata of isolates (Isolate ID, Source plate, Source sample, Culturing media,
    # Isolation strategy, taxonomy ...). CSV / TSV / XLSX.
    "isolate_metadata":   r"C:\data\CAMII\CAMII_isolates_metadata.tsv",

    # Morphological features of isolates (Isolate ID + Area, Perimeter, ...).
    # Can be the same file as isolate_metadata.
    "isolate_morphology": r"C:\data\CAMII\CAMII_isolates_morphological_features.tsv",

    # 16S results of isolates (Isolate ID + ASV / taxonomy / purity).
    # Set to None if taxonomy is already in isolate_metadata.
    "isolate_16s":        r"C:\data\CAMII\CAMII_isolates_16SV4_result.tsv",

    # OPTIONAL: metadata of ALL visually captured colonies (plate barcode, X, Y, features).
    # A single file, or a folder of per-plate *_Metadata_all.csv files. None = skip.
    "colony_metadata":    r"C:\data\CAMII\colony_all.metadata.20220714",

    # OPTIONAL: folder with raw BMP plate images named <barcode>_<time>.bmp
    # (trans-illuminated first, epi-illuminated second). None = skip image embeddings.
    "raw_image_dir":      r"C:\data\CAMII\00_raw_image_plate",

    # Output folder (created if missing)
    "output_dir":         r"C:\data\CAMII\results",
}

# Per-file reading options: header row index (0 = first row), Excel sheet name/index.
FILE_OPTIONS = {
    "isolate_metadata":   {"header": 0, "sheet": 0},
    "isolate_morphology": {"header": 0, "sheet": 0},
    "isolate_16s":        {"header": 0, "sheet": 0},
    "colony_metadata":    {"header": 0, "sheet": 0},
}

MODE = "run"              # "inspect" (check columns)  or  "run" (full analysis)

# Experiments run one after another in MODE="run". Each entry overrides CONFIG settings below;
# results go to <output_dir>/<name>/ and a comparison table to <output_dir>/experiments_comparison.csv.
# Image embeddings are computed once and cached in <output_dir> (later runs reuse them).
# Every experiment states its settings explicitly, so the CONFIG values below do not leak into it.
_BASE = {"DIVERSITY_LEVEL": "asv", "CLASSIFIER_LEVEL": "asv", "NOVELTY_OTHER_WEIGHT": 0.0,
         "USE_IMAGE_EMBEDDINGS": False, "REQUIRE_IMAGES": True, "USE_CNN": False,
         "NEW_STRATEGIES": False}
EXPERIMENTS = [
    {**_BASE, "name": "11_asv_imgset"},
    {**_BASE, "name": "12_genus_imgset",              "DIVERSITY_LEVEL": "genus", "CLASSIFIER_LEVEL": "genus"},
    {**_BASE, "name": "13_asv_emb_norm",     "USE_IMAGE_EMBEDDINGS": True},
    {**_BASE, "name": "14_genus_emb_norm",   "DIVERSITY_LEVEL": "genus", "CLASSIFIER_LEVEL": "genus",
                                               "USE_IMAGE_EMBEDDINGS": True},
    {**_BASE, "name": "21_genus_cnn",          "DIVERSITY_LEVEL": "genus", "CLASSIFIER_LEVEL": "genus",
                                               "USE_CNN": True},
    # strategies 1-3 on all four random-pick groups (H1t1 included: no images needed)
    {**_BASE, "name": "31_asv_new",             "NEW_STRATEGIES": True, "REQUIRE_IMAGES": False},
    {**_BASE, "name": "32_asv_div_genus_clf",   "NEW_STRATEGIES": True, "REQUIRE_IMAGES": False,
                                               "CLASSIFIER_LEVEL": "genus"},
    {**_BASE, "name": "33_genus_new",           "NEW_STRATEGIES": True, "REQUIRE_IMAGES": False,
                                               "DIVERSITY_LEVEL": "genus", "CLASSIFIER_LEVEL": "genus"},
]
SKIP_EXPERIMENTS = ["11_asv_imgset", "12_genus_imgset", "13_asv_emb_norm", "14_genus_emb_norm",
                    "21_genus_cnn"]  # done; [] = all
# EXPERIMENTS = []   # <- uncomment to run a single analysis with the settings below

# If auto-detection picks the wrong column or finds nothing, give the exact name here.
# A list means "first of these that exists in the file".
# Example: {"strategy": "Isolation strategy", "media": ["Culturing media", "Media"]}
COLUMN_OVERRIDES = {}

# --- analysis settings -------------------------------------------------------
DIVERSITY_LEVEL   = "asv"     # what counts as "a different taxon" in curves/AUC: "asv" | "species" | "genus"
CLASSIFIER_LEVEL  = "asv"     # what the classifier predicts:                       "asv" | "species" | "genus"
GROUP_COLS        = ["sample", "media"]   # canonical columns defining a picking group
                                          # (CAMII grouped plates of one sample+medium). Alternative: ["plate"]
RANDOM_STRATEGY_PATTERN = r"random"       # regex (case-insensitive) on Isolation strategy marking random picks
DONOR_FROM_SAMPLE_REGEX = r"t\d+$"        # removed from sample ID to get donor (H1t4 -> H1)
                                          # CAMII format "H5t1 (feces from H5)" -> donor taken from "from H5"
MIN_PURITY        = 50        # drop isolates with 16S purity < this % (NaN purity is kept); None = no filter
ASV_DENOISE       = True      # merge rare V4 sequences 1 mismatch away from a >=2x more abundant one
MIN_POOL          = 15        # skip groups whose random pool is smaller than this
POOL_SUBSAMPLE_SIZE    = 200  # CAMII has only ~4 random-picked groups of ~1000 isolates: draw many
POOL_SUBSAMPLE_REPEATS = 20   # candidate pools of this size from each (None = use the whole group once)
N_GRID            = [5, 10, 20, 30, 50, 75, 100]
RANDOM_REPEATS    = 50
MAXMIN_RESTARTS   = 10        # CAMII used 10
MAXMIN_MAX_ITERS  = 50
LOG_FEATURES      = ["Area", "Perimeter", "Radius"]

# Classifier
CLF_N_SPLITS      = 5         # GroupKFold by donor (use a large number, e.g. 99, for leave-one-donor-out)
CLF_MIN_CLASS_COUNT = 10      # rarer labels are merged into "__other__"
CLF_N_TREES       = 300
NOVELTY_OTHER_WEIGHT = 0.5    # value of a colony predicted "__other__" in expected-novelty picking
TARGET_WEIGHTS    = None      # e.g. {"Bifidobacterium": 5.0} (keys = classifier label names); None = pure diversity

# Pair-AUC evaluation
PAIR_EVAL_USE_ALL_ISOLATES = True   # True: all labelled isolates of a group; False: only random pool
MAX_PAIRS_PER_GROUP = 20000

# Neighbour / plate-density correction (needs colony_metadata and isolate X/Y)
NEIGHBOR_RADIUS_PX = 50

# Image embeddings (needs raw_image_dir, isolate X/Y and torch)
USE_IMAGE_EMBEDDINGS = False
DINO_MODEL        = "dinov2_vits14"
CROP_X_OFFSET     = 150       # cropXMin from CAMII configure (X/Y are in cropped coordinates)
CROP_Y_OFFSET     = 150       # cropYMin
CROP_HALF         = 48        # crop = 96 x 96 px around colony centre (fixed scale!)
MASK_RADIUS_FACTOR = 1.4      # mask outside Radius*factor (None = no mask)
EMBED_PCA_DIMS    = 32
EMBED_BATCH       = 64        # crops per batch (GPU with >=6 GB: 256)
EMBED_INPUT_SIZE  = 112       # crops are resized to this (multiple of 14). 112 is ~4x faster than 224
                              # and enough for 96-px colony crops
# Extra strategies (experiments 31-33)
NEW_STRATEGIES    = False     # calibration, hybrid, rarity, adaptive rounds
ROUND_SIZE        = 10        # colonies per picking round (sequencing between rounds)
ADAPTIVE_ALPHA    = 0.5       # weight of the donor-specific duplicate model vs classifier novelty
HYBRID_LAMBDA     = 0.3       # weight of distance-to-chosen in the hybrid strategy
RARITY_K          = 10        # neighbours for the colony-rarity score

# Supervised CNN on crops (experiment 21)
USE_CNN           = False
CNN_INPUT_SIZE    = 64        # crop (2*CROP_HALF px) resized to this
CNN_WIDTH         = 32        # channels of first block (blocks: w, 2w, 3w, 4w)
CNN_EPOCHS        = 15
CNN_BATCH         = 128
CNN_LR            = 3e-3
CNN_MASK          = True      # blank out everything beyond Radius*MASK_RADIUS_FACTOR (removes neighbours)
CROP_NORMALIZE    = True      # per-plate contrast stretch (1st..99.8th percentile) of trans and each epi channel
REQUIRE_IMAGES    = False     # True: analyse only isolates whose plate has a trans+epi image pair
                              # (makes runs with and without embeddings comparable)
DEVICE            = "auto"    # "auto" = GPU if torch supports it, else CPU; "cuda" = GPU or error; "cpu"
USE_FP16          = True      # half precision on GPU: ~2x faster, same embeddings for this purpose

SEED = 0

# =============================================================================
# Column aliases (normalised: lowercase, only letters/digits are compared)
# =============================================================================
COLUMN_ALIASES = {
    "isolate_id":   ["Isolate ID", "isolate_id", "IsolateID", "Isolate", "isolate name"],
    "plate":        ["Source plate", "source_plate", "plate_barcode", "plate barcode", "barcode", "plate"],
    "source_index": ["Source index", "source_index", "colony index", "colony_index"],
    "sample":       ["Source sample", "source_sample", "sampleID", "sample", "sample of origin"],
    "media":        ["Culturing media", "media", "medium", "plateType", "plate type", "plateTyte"],
    "strategy":     ["isolateStrategy", "Isolation strategy", "isolate strategy", "strategy"],
    "donor":        ["Individual of origin", "individual", "donor", "subject", "host"],
    "asv":          ["ASV", "ASV ID", "ASV (16SV4)", "asv_id", "ASVs"],
    "seq16s":       ["16SV4 sequence", "16S sequence", "sequence", "16SV4"],
    "species":      ["Species"],
    "genus":        ["Genus"],
    "family":       ["Family"],
    "purity":       ["Purity (%)", "Purity", "purity_pct"],
    "x":            ["X", "x_coord", "X coordinate"],
    "y":            ["Y", "y_coord", "Y coordinate"],
    "x_raw":        ["X_rawImage", "x_raw"],
    "y_raw":        ["Y_rawImage", "y_raw"],
    "colony_index": ["colony index", "colony_index", "index", "Unnamed: 0", "Source index"],
}

FEATURES = ["Area", "Perimeter", "Radius", "Circularity", "Convexity", "Inertia",
            "Graymean", "Graystd", "Repimean", "Repistd", "Gepimean", "Gepistd",
            "Bepimean", "Bepistd"]
FEATURE_ALIASES = {
    "Area": ["Area"], "Perimeter": ["Perimeter"], "Radius": ["Radius"],
    "Circularity": ["Circularity"], "Convexity": ["Convexity"], "Inertia": ["Inertia", "InertiaRatio"],
    "Graymean": ["Graymean", "Gray(mean)", "Gray mean", "Grey mean"],
    "Graystd":  ["Graystd", "Gray(sd)", "Gray sd", "Gray std"],
    "Repimean": ["Repimean", "R(mean)", "Red mean", "Rmean"],
    "Repistd":  ["Repistd", "R(sd)", "Red sd", "Rstd"],
    "Gepimean": ["Gepimean", "G(mean)", "Green mean", "Gmean"],
    "Gepistd":  ["Gepistd", "G(sd)", "Green sd", "Gstd"],
    "Bepimean": ["Bepimean", "B(mean)", "Blue mean", "Bmean"],
    "Bepistd":  ["Bepistd", "B(sd)", "Blue sd", "Bstd"],
}

# =============================================================================
# Imports
# =============================================================================
import os
import re
import sys
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, squareform, pdist
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

MISSING_TOKENS = {"", "na", "nan", "none", "null", "-", "unclassified", "unknown", "n/a"}


def log(msg=""):
    print(msg, flush=True)


# =============================================================================
# Reading tables and finding columns
# =============================================================================
def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def read_table(path, header=0, sheet=0):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File/folder not found: {p}")
    if p.is_dir():
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in (".csv", ".tsv", ".txt")])
        if not files:
            raise FileNotFoundError(f"No CSV/TSV files in folder {p}")
        frames = []
        for f in files:
            d = read_table(f, header=header, sheet=sheet)
            d["__source_file"] = f.stem
            frames.append(d)
        return pd.concat(frames, ignore_index=True)
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(p, sheet_name=sheet, header=header)
    if ext in (".tsv", ".tab"):
        return pd.read_csv(p, sep="\t", header=header, low_memory=False)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(p)
    # .csv / .txt / unknown: sniff separator
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    sep = "\t" if first.count("\t") > first.count(",") else ("," if first.count(",") >= first.count(";") else ";")
    return pd.read_csv(p, sep=sep, header=header, low_memory=False, encoding_errors="replace")


def load_named(key):
    path = PATHS.get(key)
    if not path:
        return None
    opt = FILE_OPTIONS.get(key, {})
    df = read_table(path, header=opt.get("header", 0), sheet=opt.get("sheet", 0))
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df, key, aliases=None, required=False):
    if df is None:
        return None
    if key in COLUMN_OVERRIDES and COLUMN_OVERRIDES[key]:
        cand = COLUMN_OVERRIDES[key]
        cand = [cand] if isinstance(cand, str) else list(cand)
        for c in cand:
            if c in df.columns:
                return c
        if required:
            raise KeyError(f"Override for '{key}' ({cand}) not found. Columns: {list(df.columns)}")
        return None
    normmap = {}
    for c in df.columns:
        normmap.setdefault(norm(c), c)
    for a in (aliases or COLUMN_ALIASES.get(key, [key])):
        if norm(a) in normmap:
            return normmap[norm(a)]
    if required:
        raise KeyError(f"Could not find column for '{key}'. Add it to COLUMN_OVERRIDES. "
                       f"Available columns: {list(df.columns)}")
    return None


def canon(df, keys):
    """Return a DataFrame with canonical column names for the keys that exist."""
    out = pd.DataFrame(index=df.index)
    found = {}
    for k in keys:
        c = find_col(df, k)
        if c is not None:
            out[k] = df[c].values
            found[k] = c
    return out, found


def canon_features(df):
    out = pd.DataFrame(index=df.index)
    found = {}
    for f in FEATURES:
        c = find_col(df, "feat_" + f, aliases=FEATURE_ALIASES[f])
        if c is not None:
            out[f] = pd.to_numeric(df[c], errors="coerce").values
            found[f] = c
    return out, found


def clean_str(s):
    s = s.astype(str).str.strip()
    return s.where(~s.str.lower().isin(MISSING_TOKENS), np.nan)


def donor_from_sample(x):
    """'H5t1 (feces from H5)' -> 'H5'; 'H12_F (Pre-selected culture of H12t1)' -> 'H12'; 'H1t4' -> 'H1'."""
    if not isinstance(x, str):
        return np.nan
    m = re.search(r"\((?:.*?\b(?:from|of)\s+)(.+?)\)\s*$", x)
    core = m.group(1).strip() if m else re.sub(r"\s*\(.*\)\s*$", "", x).strip()
    core = re.sub(r"^(?:CRE sample|BA patient|Fstore sample)\s+", "", core, flags=re.I)
    return re.sub(DONOR_FROM_SAMPLE_REGEX, "", core.split()[0]) if core else np.nan


def denoise_sequences(seqs, max_mismatch=1, skew=2.0):
    """Greedy UNOISE-like: a sequence joins a >=skew x more abundant same-length centroid
    within max_mismatch differences; otherwise it becomes a new centroid."""
    vc = seqs.dropna().value_counts()
    centroids, mapping = [], {}
    for seq, c in vc.items():
        a = np.frombuffer(seq.encode(), dtype=np.uint8)
        target = seq
        for cseq, carr, ccount in centroids:
            if ccount >= skew * c and len(carr) == len(a) and int((carr != a).sum()) <= max_mismatch:
                target = cseq
                break
        if target == seq:
            centroids.append((seq, a, c))
        mapping[seq] = target
    return seqs.map(mapping), len(vc), len(centroids)


def clean_taxon(s):
    s = clean_str(s)
    s = s.str.replace(r"^[a-z]__", "", regex=True)   # g__Bacteroides -> Bacteroides
    return s.where(~s.fillna("").str.lower().isin(MISSING_TOKENS), np.nan)


# =============================================================================
# INSPECT mode
# =============================================================================
def inspect():
    log("=" * 78)
    log("INSPECT MODE — checking files and column recognition")
    log("=" * 78)
    for key in ["isolate_metadata", "isolate_morphology", "isolate_16s", "colony_metadata"]:
        path = PATHS.get(key)
        log(f"\n### {key}: {path}")
        if not path:
            log("   (not set — skipped)")
            continue
        try:
            df = load_named(key)
        except Exception as e:
            log(f"   !! cannot read: {e}")
            continue
        log(f"   shape: {df.shape}")
        log(f"   columns: {list(df.columns)}")
        log(df.head(3).to_string(max_colwidth=40))
        keys = list(COLUMN_ALIASES.keys())
        _, found = canon(df, keys)
        _, ffound = canon_features(df)
        log(f"   recognised columns: {found}")
        log(f"   recognised features ({len(ffound)}/14): {ffound}")
        if "strategy" in found:
            log(f"   Isolation strategy values: {df[found['strategy']].value_counts(dropna=False).head(15).to_dict()}")
        if "media" in found:
            log(f"   media values: {df[found['media']].value_counts(dropna=False).head(15).to_dict()}")
        if "sample" in found:
            vals = df[found['sample']].dropna().astype(str)
            donors = vals.map(donor_from_sample)
            log(f"   samples: {vals.nunique()}  -> donors (after regex): {sorted(donors.unique())[:40]}")
    log("\nIf something is missing or wrong, set COLUMN_OVERRIDES / FILE_OPTIONS and re-run.")
    log("When everything looks right, set MODE = 'run'.")


# =============================================================================
# Build the isolate dataset
# =============================================================================
def merge_prefer_left(left, right, on="isolate_id"):
    if right is None or right.empty:
        return left
    right = right.drop_duplicates(subset=on)
    m = left.merge(right, on=on, how="left", suffixes=("", "__r"))
    for c in list(m.columns):
        if c.endswith("__r"):
            base = c[:-3]
            m[base] = m[base].where(m[base].notna(), m[c])
            m.drop(columns=c, inplace=True)
    return m


def build_dataset():
    keys = list(COLUMN_ALIASES.keys())
    meta_raw = load_named("isolate_metadata")
    meta, f1 = canon(meta_raw, keys)
    feats1, ff1 = canon_features(meta_raw)
    if "isolate_id" not in meta:
        find_col(meta_raw, "isolate_id", required=True)
    df = pd.concat([meta, feats1], axis=1)
    log(f"isolate_metadata: {len(df)} rows, columns {f1}")

    if PATHS.get("isolate_morphology") and PATHS["isolate_morphology"] != PATHS["isolate_metadata"]:
        mo_raw = load_named("isolate_morphology")
        mo, f2 = canon(mo_raw, keys)
        feats2, ff2 = canon_features(mo_raw)
        find_col(mo_raw, "isolate_id", required=True)
        mo = pd.concat([mo, feats2], axis=1)
        log(f"isolate_morphology: {len(mo)} rows, features found {len(ff2)}/14")
        df["isolate_id"] = clean_str(df["isolate_id"])
        mo["isolate_id"] = clean_str(mo["isolate_id"])
        df = merge_prefer_left(df, mo)

    if PATHS.get("isolate_16s") and PATHS["isolate_16s"] not in (PATHS["isolate_metadata"], PATHS.get("isolate_morphology")):
        s_raw = load_named("isolate_16s")
        s, f3 = canon(s_raw, keys)
        find_col(s_raw, "isolate_id", required=True)
        log(f"isolate_16s: {len(s)} rows, columns {f3}")
        df["isolate_id"] = clean_str(df["isolate_id"])
        s["isolate_id"] = clean_str(s["isolate_id"])
        df = merge_prefer_left(df, s)

    df["isolate_id"] = clean_str(df["isolate_id"])
    df = df.dropna(subset=["isolate_id"]).drop_duplicates(subset="isolate_id").reset_index(drop=True)

    # --- features
    present = [f for f in FEATURES if f in df.columns and df[f].notna().any()]
    if len(present) < 4:
        raise RuntimeError(f"Only {len(present)} morphological features found ({present}). "
                           f"Check isolate_morphology path / FEATURE_ALIASES.")
    missing = [f for f in FEATURES if f not in present]
    if missing:
        log(f"WARNING: features not found, analysis continues without them: {missing}")

    # --- labels
    for k in ["genus", "family", "species"]:
        if k in df:
            df[k] = clean_taxon(df[k])
    if "asv" in df and clean_str(df["asv"]).notna().any():
        df["asv"] = clean_str(df["asv"])
    elif "seq16s" in df and clean_str(df["seq16s"]).notna().any():
        seq = clean_str(df["seq16s"]).str.upper()
        if MIN_PURITY is not None and "purity" in df:   # denoise on the isolates that will be kept
            pur = pd.to_numeric(df["purity"], errors="coerce")
            pur = pur * 100 if pur.max(skipna=True) <= 1.5 else pur
            seq = seq.where(~(pur < MIN_PURITY), np.nan)
        if ASV_DENOISE:
            seq, n_raw, n_cent = denoise_sequences(seq)
            log(f"ASV denoising: {n_raw} unique sequences -> {n_cent} ASVs (1-mismatch, 2x abundance skew)")
        codes = {s: f"ASVseq{i+1}" for i, s in enumerate(seq.dropna().value_counts().index)}
        df["asv"] = seq.map(codes)
        vc = df["asv"].value_counts()
        log(f"ASV derived from 16S sequences: {len(codes)} ASVs ({int((vc == 1).sum())} seen only once)")
    else:
        df["asv"] = np.nan

    def level_labels(level):
        if level == "asv":
            if df["asv"].notna().any():
                return df["asv"]
            log("WARNING: no ASV/sequence column -> falling back to genus")
            level = "genus"
        if level == "species" and "species" in df:
            return df["species"]
        if level in ("genus", "species"):
            g = df["genus"] if "genus" in df else pd.Series(np.nan, index=df.index)
            if "family" in df:
                g = g.where(g.notna(), df["family"] + "_unclassified")
            return g
        raise ValueError(level)

    df["label_div"] = level_labels(DIVERSITY_LEVEL)
    df["label_clf"] = level_labels(CLASSIFIER_LEVEL)

    # --- purity
    if MIN_PURITY is not None and "purity" in df:
        pur = pd.to_numeric(df["purity"], errors="coerce")
        if pur.max(skipna=True) <= 1.5:
            pur = pur * 100
        before = len(df)
        df = df[~(pur < MIN_PURITY)].reset_index(drop=True)
        log(f"purity filter (<{MIN_PURITY}%): removed {before - len(df)} isolates")

    # --- donor, group, pool
    if "donor" not in df or df["donor"].isna().all():
        if "sample" not in df:
            raise RuntimeError("Neither donor nor sample column found — set COLUMN_OVERRIDES['sample'].")
        df["donor"] = clean_str(df["sample"]).map(donor_from_sample)
    df["donor"] = clean_str(df["donor"])
    if "sample" in df:
        df["sample"] = clean_str(df["sample"]).str.replace(r"\s*\(.*\)\s*$", "", regex=True)
    if "media" in df:
        df["media"] = clean_str(df["media"]).str.replace(r"\s+", " ", regex=True).str.lower()

    gcols = [c for c in GROUP_COLS if c in df]
    if not gcols:
        raise RuntimeError(f"None of GROUP_COLS {GROUP_COLS} found.")
    if len(gcols) < len(GROUP_COLS):
        log(f"WARNING: group columns missing, using only {gcols}")
    df["group"] = df[gcols].fillna("NA").astype(str).agg("|".join, axis=1)

    if "strategy" not in df:
        raise RuntimeError("Isolation strategy column not found — needed to find the random pool.")
    df["is_random"] = df["strategy"].astype(str).str.contains(RANDOM_STRATEGY_PATTERN, case=False, regex=True)

    before = len(df)
    df = df.dropna(subset=["label_div", "donor"]).reset_index(drop=True)
    log(f"dropped {before - len(df)} isolates without label/donor")

    # impute missing feature values with median
    for f in present:
        n_na = df[f].isna().sum()
        if n_na:
            df[f] = df[f].fillna(df[f].median())
            log(f"  {f}: {n_na} missing values imputed with median")

    log(f"\nDATASET: {len(df)} isolates | donors {df['donor'].nunique()} | groups {df['group'].nunique()} "
        f"| {DIVERSITY_LEVEL} labels {df['label_div'].nunique()} | random-picked {int(df['is_random'].sum())}")
    log(f"strategy values: {df['strategy'].value_counts().to_dict()}")
    return df, present


# =============================================================================
# Optional: link to all-colonies table -> neighbour counts, X/Y
# =============================================================================
def attach_colony_context(df):
    if not PATHS.get("colony_metadata"):
        return df, False
    try:
        col_raw = load_named("colony_metadata")
    except Exception as e:
        log(f"colony_metadata could not be read ({e}) — context correction skipped")
        return df, False
    c, found = canon(col_raw, ["plate", "x", "y", "colony_index"])
    feats, _ = canon_features(col_raw)
    c = pd.concat([c, feats], axis=1)
    if "plate" not in c and "__source_file" in col_raw:
        c["plate"] = col_raw["__source_file"].astype(str).str.split("_").str[0].values
    if not {"plate", "x", "y"} <= set(c.columns):
        log(f"colony_metadata lacks plate/X/Y (found {found}) — context correction skipped")
        return df, False
    c["plate"] = clean_str(c["plate"])
    c["x"] = pd.to_numeric(c["x"], errors="coerce")
    c["y"] = pd.to_numeric(c["y"], errors="coerce")
    c = c.dropna(subset=["plate", "x", "y"])
    log(f"colony_metadata: {len(c)} colonies on {c['plate'].nunique()} plates")
    global _COLONY_TABLE
    _COLONY_TABLE = c

    if "plate" not in df:
        log("isolates have no plate column — context correction skipped")
        return df, False
    df["plate"] = clean_str(df["plate"])

    # If isolates lack X/Y, try linking via source index (test 0- and 1-based)
    if not ({"x", "y"} <= set(df.columns)) or df["x"].isna().all():
        if "source_index" in df and "colony_index" in c:
            best = None
            for off in (0, 1):
                cc = c.copy()
                cc["colony_index"] = pd.to_numeric(cc["colony_index"], errors="coerce") + off
                tmp = df[["isolate_id", "plate", "source_index"]].copy()
                tmp["source_index"] = pd.to_numeric(tmp["source_index"], errors="coerce")
                m = tmp.merge(cc, left_on=["plate", "source_index"], right_on=["plate", "colony_index"], how="left")
                score = m["x"].notna().mean()
                if "Area" in df and "Area" in m:
                    a = df.set_index("isolate_id")["Area"].reindex(m["isolate_id"]).values
                    ok = m["Area"].notna().values
                    if ok.sum() > 10:
                        score = np.corrcoef(a[ok], m["Area"].values[ok])[0, 1]
                if best is None or score > best[0]:
                    best = (score, off, m)
            score, off, m = best
            df["x"] = m["x"].values
            df["y"] = m["y"].values
            log(f"linked isolates to colonies via source index (offset {off}); "
                f"match/Area-agreement score = {score:.3f}, matched {np.isfinite(df['x']).mean():.1%}")
        else:
            log("isolates have no X/Y and no source index to link — context correction skipped")
            return df, False

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["n_neighbors"] = np.nan
    df["plate_count"] = np.nan
    for plate, sub in c.groupby("plate"):
        idx = df.index[(df["plate"] == plate) & df["x"].notna()]
        if len(idx) == 0:
            continue
        tree = cKDTree(sub[["x", "y"]].values)
        cnt = tree.query_ball_point(df.loc[idx, ["x", "y"]].values, r=NEIGHBOR_RADIUS_PX, return_length=True)
        df.loc[idx, "n_neighbors"] = np.maximum(np.asarray(cnt) - 1, 0)
        df.loc[idx, "plate_count"] = len(sub)
    ok = df["n_neighbors"].notna().mean()
    log(f"neighbour counts computed for {ok:.1%} of isolates (radius {NEIGHBOR_RADIUS_PX}px)")
    return df, ok > 0.5


# =============================================================================
# Representations
# =============================================================================
def zscore(X):
    X = np.asarray(X, float)
    sd = X.std(0)
    sd[sd < 1e-12] = 1.0
    return (X - X.mean(0)) / sd


def log_features(df, feats):
    X = df[feats].values.astype(float).copy()
    for i, f in enumerate(feats):
        if f in LOG_FEATURES:
            X[:, i] = np.log1p(np.clip(X[:, i], 0, None))
    return X


def context_corrected(df, feats):
    """Regress log size features on log(neighbours) and log(plate density); keep residual + mean."""
    from sklearn.linear_model import LinearRegression
    X = log_features(df, feats)
    ok = df["n_neighbors"].notna().values
    Z = np.column_stack([np.log1p(df["n_neighbors"].fillna(0).values),
                         np.log1p(df["plate_count"].fillna(df["plate_count"].median()).values)])
    for i, f in enumerate(feats):
        if f in LOG_FEATURES:
            lr = LinearRegression().fit(Z[ok], X[ok, i])
            log(f"  context correction {f}: coef neighbours={lr.coef_[0]:+.3f}, plate density={lr.coef_[1]:+.3f}, "
                f"R2={lr.score(Z[ok], X[ok, i]):.3f}")
            X[ok, i] = X[ok, i] - lr.predict(Z[ok]) + X[ok, i].mean()
    return X


def group_rep(name, idx, store):
    """Distance-space representation for a subset (one picking group)."""
    if name == "CAMII_PCA2":
        from sklearn.decomposition import PCA
        Xz = zscore(store["raw"][idx])
        k = min(2, Xz.shape[0], Xz.shape[1])
        return PCA(k).fit_transform(Xz)
    if name == "full14_z":
        return zscore(store["raw"][idx])
    if name == "full14_log_z":
        return zscore(store["log"][idx])
    if name == "context_corr_z":
        return zscore(store["ctx"][idx])
    if name == "embedding":
        return store["emb"][idx]
    if name == "cnn_feat":
        return zscore(store["cnn"][idx])
    if name == "embedding+morph":
        return np.hstack([store["emb"][idx], zscore(store["log"][idx])])
    raise KeyError(name)


# =============================================================================
# Optional image embeddings (DINOv2)
# =============================================================================
def extract_crop(img, cx, cy, half, radius=None, fill=0):
    H, W = img.shape[:2]
    out_shape = (2 * half, 2 * half) + img.shape[2:]
    out = np.full(out_shape, fill, dtype=img.dtype)
    x0, y0 = cx - half, cy - half
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(cx + half, W), min(cy + half, H)
    if xe > xs and ye > ys:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    if radius is not None:
        yy, xx = np.mgrid[:2 * half, :2 * half]
        m = (xx - half) ** 2 + (yy - half) ** 2 > radius ** 2
        out[m] = fill
    return out


def index_plate_images(raw_dir):
    """Map plate barcode -> list of BMP paths. Searches recursively. Barcode = file-name part before
    the first '_' (CAMII: <barcode>_<time>.bmp); if the file sits in a folder named after the plate,
    that folder name is also registered."""
    idx = {}
    files = [p for p in Path(raw_dir).rglob("*") if p.suffix.lower() == ".bmp"]
    for p in files:
        keys = {p.stem.split("_")[0], p.stem.split(".")[0]}
        if p.parent != Path(raw_dir):
            keys.add(p.parent.name)
        for k in keys:
            idx.setdefault(k, []).append(p)
    log(f"found {len(files)} BMP files under {raw_dir} ({len(idx)} candidate plate keys)")
    if files[:3]:
        log("  examples: " + ", ".join(f.name for f in files[:3]))
    return idx


def classify_illumination(path):
    """'trans' if the image is red-dominated (CAMII trans-illumination), else 'epi'."""
    from PIL import Image
    with Image.open(path) as im:
        im.draft("RGB", (200, 200))
        a = np.asarray(im.convert("RGB").resize((160, 110)), dtype=float)
    r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
    return "trans" if r > 1.8 * max(g, b, 1.0) else "epi"


def pick_image_pair(paths):
    """Return (trans_path, epi_path). With several time points take the last one of each kind
    (files sorted by name = by CAMII timestamp)."""
    trans, epi = [], []
    for p in sorted(set(paths)):
        (trans if classify_illumination(p) == "trans" else epi).append(p)
    if not trans or not epi:
        return None, None
    return trans[-1], epi[-1]


def _stretch(a, lo_p=1.0, hi_p=99.8):
    """Linear contrast stretch using percentiles of the plate interior (edges/frame excluded)."""
    h, w = a.shape[:2]
    inner = a[int(h * 0.12):int(h * 0.88), int(w * 0.08):int(w * 0.92)]
    lo, hi = np.percentile(inner, [lo_p, hi_p])
    if hi - lo < 1:
        return a
    return np.clip((a.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def load_trans_epi(trans_path, epi_path):
    from PIL import Image
    t = np.asarray(Image.open(trans_path).convert("RGB"))
    t = t[..., int(np.argmax(t.reshape(-1, 3).mean(0)))]   # dominant (red) channel as grey image
    e = np.asarray(Image.open(epi_path).convert("RGB"))
    if CROP_NORMALIZE:
        t = _stretch(t)
        e = np.stack([_stretch(e[..., c]) for c in range(3)], axis=-1)
    return np.ascontiguousarray(t), np.ascontiguousarray(e)


def plates_with_images(cache_dir):
    """Set of plate barcodes that have a usable trans+epi pair (cached in cache_dir)."""
    import json
    f = Path(cache_dir) / "plates_with_images.json"
    if f.exists():
        return set(json.loads(f.read_text()))
    if not PATHS.get("raw_image_dir"):
        return None
    log("checking which plates have a trans+epi image pair (one-time, cached) ...")
    index = index_plate_images(PATHS["raw_image_dir"])
    ok = set()
    for k, paths in index.items():
        if len(paths) >= 2:
            tp, ep = pick_image_pair(paths)
            if tp is not None:
                ok.add(k)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(sorted(ok)))
    return ok


def isolate_centre(r):
    if "x_raw" in r and pd.notna(r.get("x_raw")) and pd.notna(r.get("y_raw")):
        return int(round(float(r["x_raw"]))), int(round(float(r["y_raw"])))
    return int(round(float(r["x"]))) + CROP_X_OFFSET, int(round(float(r["y"]))) + CROP_Y_OFFSET


def save_crop_montage(examples, path):
    """examples: list of (label, trans_crop, epi_crop). Saves a PNG grid for visual QC."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(examples)
    if n == 0:
        return
    cols = 8
    rows = int(np.ceil(n / cols)) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")
    for i, (lab, ct, ce) in enumerate(examples):
        rr, cc = (i // cols) * 2, i % cols
        axes[rr, cc].imshow(ct, cmap="gray")
        axes[rr, cc].set_title(str(lab)[:18], fontsize=6)
        axes[rr + 1, cc].imshow(ce)
    fig.suptitle("Colony crops used for embeddings (row pairs: trans / epi). Colony must be centred.", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def compute_embeddings(df, cache_dir):
    """DINOv2 embeddings of trans + epi crops for every isolate in df. Cached in cache_dir and
    extended incrementally, so several experiments reuse the same file."""
    if not USE_IMAGE_EMBEDDINGS:
        return None
    if not PATHS.get("raw_image_dir"):
        log("USE_IMAGE_EMBEDDINGS=True but raw_image_dir not set — skipped")
        return None
    if "plate" not in df or not ({"x_raw", "y_raw"} <= set(df.columns) or {"x", "y"} <= set(df.columns)):
        log("image embeddings need plate + coordinates per isolate — skipped")
        return None
    tag = f"{DINO_MODEL}_crop{CROP_HALF}_in{EMBED_INPUT_SIZE}{'_norm' if CROP_NORMALIZE else ''}"
    cache = Path(cache_dir) / f"embeddings_{tag}.npz"
    have = None
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have = pd.DataFrame(z["emb"], index=pd.Index(z["ids"].astype(str)))
        have = have[~have.index.duplicated()]
    todo = df[~df["isolate_id"].isin(have.index if have is not None else [])]
    todo = todo.dropna(subset=["plate"])
    tried_file = Path(cache_dir) / f"embeddings_{tag}_unavailable.txt"
    if tried_file.exists():   # isolates whose plate image was not found in a previous run
        skip = set(tried_file.read_text().split())
        todo = todo[~todo["isolate_id"].isin(skip)]
    if len(todo):
        log(f"embedding {len(todo)} isolates (cached: {0 if have is None else len(have)})")
        new, unavailable = _embed_isolates(todo, Path(cache_dir))
        if new is not None:
            have = new if have is None else pd.concat([have, new])
            np.savez(cache, emb=have.values, ids=np.array(have.index, dtype=object))
        if unavailable:
            with open(tried_file, "a") as fh:
                fh.write("\n".join(unavailable) + "\n")
    if have is None:
        return None
    emb = have.reindex(df["isolate_id"].astype(str)).values
    log(f"embeddings: dim {emb.shape[1]}, coverage {np.isfinite(emb[:, 0]).mean():.1%} of isolates")
    return emb


def get_device(torch):
    if DEVICE == "cpu":
        log(f"using CPU ({torch.get_num_threads()} threads)")
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        major, minor = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        supported = any(a.startswith("sm_") and int(a[3:]) // 10 == major and int(a[3:]) % 10 <= minor
                        for a in archs if a[3:].isdigit())
        if supported:
            log(f"GPU: {name} ({mem:.1f} GB, CC {major}.{minor}), torch {torch.__version__}, CUDA {torch.version.cuda}")
            torch.backends.cudnn.benchmark = True
            return "cuda"
        msg = (f"GPU {name} has compute capability {major}.{minor}, but this torch build supports only "
               f"{', '.join(archs)}. ")
        if DEVICE == "cuda":
            raise RuntimeError(msg + "Set DEVICE = 'auto' or 'cpu'.")
        log("NOTE: " + msg + f"Using CPU ({torch.get_num_threads()} threads).")
        return "cpu"
    msg = ("CUDA GPU not available to torch. Most often torch was installed in its CPU-only build "
           f"(torch {torch.__version__}, CUDA build: {torch.version.cuda}). Install the GPU build, e.g.:\n"
           "   1) check the GPU driver:  nvidia-smi   (top-right shows the max CUDA version)\n"
           "   2) pip uninstall -y torch torchvision\n"
           "   3) install the CUDA build with the command from https://pytorch.org/get-started/locally/\n"
           "      (e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128)\n"
           "and check with:  python -c \"import torch; print(torch.cuda.is_available())\"")
    if DEVICE == "cuda":
        raise RuntimeError(msg + "\nOr set DEVICE = 'auto' to fall back to CPU (much slower).")
    log("WARNING: " + msg + "\nFalling back to CPU.")
    return "cpu"


def _embed_isolates(todo, cache_dir):
    try:
        import torch
        from PIL import Image  # noqa: F401
    except ImportError:
        log("torch / pillow not installed — image embeddings skipped (pip install torch torchvision pillow)")
        return None, []
    dev = get_device(torch)
    log(f"loading {DINO_MODEL} on {dev} (first time downloads weights from torch.hub)")
    model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL).to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    use_amp = dev.startswith("cuda") and USE_FP16

    def embed(batch_np):
        t = torch.from_numpy(np.stack(batch_np)).to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        t = torch.nn.functional.interpolate(t, size=(EMBED_INPUT_SIZE, EMBED_INPUT_SIZE), mode="bilinear",
                                            align_corners=False)
        t = (t - mean) / std
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(t)
        return out.float().cpu().numpy()

    index = index_plate_images(PATHS["raw_image_dir"])
    ids, vecs, unavailable, montage = [], [], [], []
    missing_plates = []
    plates = todo.groupby("plate")
    t0 = time.time()
    for pi, (plate, sub) in enumerate(plates):
        paths = index.get(str(plate), [])
        tp, ep = pick_image_pair(paths) if len(paths) >= 2 else (None, None)
        if tp is None:
            missing_plates.append(str(plate))
            unavailable.extend(sub["isolate_id"].astype(str).tolist())
            continue
        trans, epi = load_trans_epi(tp, ep)
        crops_t, crops_e, pid = [], [], []
        for _, r in sub.iterrows():
            try:
                cx, cy = isolate_centre(r)
            except (TypeError, ValueError):
                unavailable.append(str(r["isolate_id"]))
                continue
            rad = None
            if MASK_RADIUS_FACTOR and pd.notna(r.get("Radius", np.nan)):
                rad = max(float(r["Radius"]) * MASK_RADIUS_FACTOR, 4.0)
            ct = extract_crop(trans, cx, cy, CROP_HALF, rad)
            ce = extract_crop(epi, cx, cy, CROP_HALF, rad)
            if len(montage) < 48 and len(crops_t) < 3:
                montage.append((f"{plate}:{r['isolate_id']}", extract_crop(trans, cx, cy, CROP_HALF),
                                extract_crop(epi, cx, cy, CROP_HALF)))
            crops_t.append(np.repeat(ct[:, :, None], 3, axis=2))
            crops_e.append(ce)
            pid.append(str(r["isolate_id"]))
        for b in range(0, len(pid), EMBED_BATCH):
            et = embed(crops_t[b:b + EMBED_BATCH])
            ee = embed(crops_e[b:b + EMBED_BATCH])
            vecs.append(np.hstack([et, ee]))
            ids.extend(pid[b:b + EMBED_BATCH])
        if pi % 10 == 0:
            el = time.time() - t0
            log(f"  plate {pi + 1}/{plates.ngroups}: {len(ids)} isolates embedded ({el:.0f}s, "
                f"~{el / (pi + 1) * (plates.ngroups - pi - 1) / 60:.0f} min left)")
    if missing_plates:
        log(f"WARNING: no trans+epi BMP pair found for {len(missing_plates)} plates "
            f"(e.g. {missing_plates[:5]}); {len(unavailable)} isolates without embedding")
    save_crop_montage(montage, cache_dir / "embedding_crops_check.png")
    if montage:
        log(f"crop check image: {cache_dir / 'embedding_crops_check.png'} — open it and confirm colonies are centred")
    if not ids:
        return None, unavailable
    return pd.DataFrame(np.vstack(vecs), index=pd.Index(ids)), unavailable


def reduce_embeddings(emb):
    from sklearn.decomposition import PCA
    ok = np.isfinite(emb).all(1)
    out = np.full((emb.shape[0], EMBED_PCA_DIMS), np.nan)
    p = PCA(min(EMBED_PCA_DIMS, ok.sum() - 1), random_state=SEED).fit(zscore(emb[ok]))
    out[ok, :p.n_components_] = zscore(p.transform(zscore(emb[ok])))
    return out[:, :p.n_components_], ok


# =============================================================================
# Classifier (out-of-fold, grouped by donor)
# =============================================================================
def oof_classifier(df, X, outdir):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score, classification_report

    y_raw = df["label_clf"].fillna("__other__").astype(str).values
    counts = pd.Series(y_raw).value_counts()
    keep = set(counts[counts >= CLF_MIN_CLASS_COUNT].index) - {"__other__"}
    y = np.array([v if v in keep else "__other__" for v in y_raw])
    classes = sorted(keep) + ["__other__"]
    cidx = {c: i for i, c in enumerate(classes)}
    donors = df["donor"].values
    n_splits = min(CLF_N_SPLITS, len(np.unique(donors)))
    log(f"classifier: {len(classes) - 1} classes (+__other__), {n_splits}-fold GroupKFold by donor, "
        f"{X.shape[1]} input features")

    P = np.zeros((len(y), len(classes)))
    for k, (tr, te) in enumerate(GroupKFold(n_splits).split(X, y, donors)):
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=CLF_N_TREES, min_samples_leaf=2, max_features="sqrt",
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=SEED + k)
        rf.fit(X[tr], y[tr])
        pr = rf.predict_proba(X[te])
        cols = [cidx[c] for c in rf.classes_]
        P[np.ix_(te, cols)] = pr
        log(f"  fold {k + 1}/{n_splits}: test donors {sorted(set(donors[te]))} ({time.time() - t0:.0f}s)")

    pred = np.array(classes)[P.argmax(1)]
    top5 = np.argsort(-P, 1)[:, :5]
    yi = np.array([cidx[v] for v in y])
    metrics = {
        "top1_accuracy": float((pred == y).mean()),
        "top5_accuracy": float((top5 == yi[:, None]).any(1).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n_classes": len(classes),
    }
    rep = classification_report(y, pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(Path(outdir) / "classifier_per_class_report.csv")
    log(f"classifier OOF: top1={metrics['top1_accuracy']:.3f} top5={metrics['top5_accuracy']:.3f} "
        f"balanced={metrics['balanced_accuracy']:.3f}")
    return P, classes, metrics


# =============================================================================
# Supervised CNN on colony crops (trans + epi), out-of-fold by donor
# =============================================================================
def build_cnn_crops(df, cache_dir):
    """4 x S x S uint8 crops (trans grey + epi RGB) for every isolate in df; cached and extended."""
    from PIL import Image
    S = CNN_INPUT_SIZE
    tag = f"cnncrops_h{CROP_HALF}_s{S}{'_norm' if CROP_NORMALIZE else ''}{'_mask' if CNN_MASK else ''}"
    cache = Path(cache_dir) / f"{tag}.npz"
    ids_have, X_have = np.array([], dtype=object), np.zeros((0, 4, S, S), np.uint8)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        ids_have, X_have = z["ids"].astype(str), z["X"]
    todo = df[~df["isolate_id"].astype(str).isin(set(ids_have))].dropna(subset=["plate"])
    if len(todo):
        log(f"extracting CNN crops for {len(todo)} isolates (cached: {len(ids_have)})")
        index = index_plate_images(PATHS["raw_image_dir"])
        new_ids, new_X = [], []
        t0 = time.time()
        plates = todo.groupby("plate")
        for pi, (plate, sub) in enumerate(plates):
            paths = index.get(str(plate), [])
            tp, ep = pick_image_pair(paths) if len(paths) >= 2 else (None, None)
            if tp is None:
                continue
            trans, epi = load_trans_epi(tp, ep)
            for _, r in sub.iterrows():
                try:
                    cx, cy = isolate_centre(r)
                except (TypeError, ValueError):
                    continue
                rad = None
                if CNN_MASK and MASK_RADIUS_FACTOR and pd.notna(r.get("Radius", np.nan)):
                    rad = max(float(r["Radius"]) * MASK_RADIUS_FACTOR, 4.0)
                ct = extract_crop(trans, cx, cy, CROP_HALF, rad)
                ce = extract_crop(epi, cx, cy, CROP_HALF, rad)
                ct = np.asarray(Image.fromarray(ct).resize((S, S), Image.BILINEAR))
                ce = np.asarray(Image.fromarray(ce).resize((S, S), Image.BILINEAR))
                new_X.append(np.concatenate([ct[None], ce.transpose(2, 0, 1)], axis=0))
                new_ids.append(str(r["isolate_id"]))
            if pi % 50 == 0:
                log(f"  crops: plate {pi + 1}/{plates.ngroups} ({time.time() - t0:.0f}s)")
        if new_ids:
            ids_have = np.concatenate([ids_have, np.array(new_ids, dtype=object)])
            X_have = np.concatenate([X_have, np.stack(new_X)], axis=0)
            np.savez(cache, ids=np.array(ids_have, dtype=object), X=X_have)
    pos = pd.Series(np.arange(len(ids_have)), index=pd.Index(ids_have.astype(str)))
    pos = pos[~pos.index.duplicated()]
    p = pos.reindex(df["isolate_id"].astype(str)).values
    ok = ~np.isnan(p)
    X = np.zeros((len(df), 4, S, S), np.uint8)
    X[ok] = X_have[p[ok].astype(int)]
    log(f"CNN crops: {ok.sum()} of {len(df)} isolates ({ok.mean():.1%}), {S}x{S} px, 4 channels")
    return X, ok


def _make_cnn(torch, n_classes):
    nn = torch.nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True), nn.MaxPool2d(2))

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            w = CNN_WIDTH
            self.features = nn.Sequential(block(4, w), block(w, 2 * w), block(2 * w, 3 * w), block(3 * w, 4 * w))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(4 * w, n_classes)

        def forward(self, x):
            h = self.pool(self.features(x)).flatten(1)
            return self.fc(self.drop(h)), h

    return SmallCNN()


def oof_cnn(df, crops, ok, classes, cache_dir, outdir):
    """Out-of-fold CNN class probabilities + penultimate features, same GroupKFold-by-donor as the RF."""
    try:
        import torch
    except ImportError:
        log("torch not installed — CNN skipped")
        return None, None, None
    from sklearn.model_selection import GroupKFold
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    cidx = {c: i for i, c in enumerate(classes)}
    y_raw = df["label_clf"].fillna("__other__").astype(str).values
    y = np.array([cidx.get(v, cidx["__other__"]) for v in y_raw])
    donors = df["donor"].values
    K = len(classes)
    n_splits = min(CLF_N_SPLITS, len(np.unique(donors)))
    tag = (f"cnn_oof_{CLASSIFIER_LEVEL}_K{K}_s{CNN_INPUT_SIZE}_w{CNN_WIDTH}_e{CNN_EPOCHS}"
           f"{'_norm' if CROP_NORMALIZE else ''}{'_mask' if CNN_MASK else ''}_n{len(df)}")
    cache = Path(cache_dir) / f"{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["classes"]) == list(classes) and list(z["ids"].astype(str)) == list(df["isolate_id"].astype(str)):
            log(f"loaded cached CNN out-of-fold predictions ({cache.name})")
            return z["P"], z["F"], {"top1_accuracy": float(z["top1"]), "top5_accuracy": float(z["top5"])}
    dev = get_device(torch)
    P = np.full((len(df), K), np.nan)
    F = np.full((len(df), 4 * CNN_WIDTH), np.nan)
    folds = list(GroupKFold(n_splits).split(np.zeros(len(df)), y, donors))
    t_all = time.time()
    for k, (tr, te) in enumerate(folds):
        tr = tr[ok[tr]]
        te_ok = te[ok[te]]
        torch.manual_seed(SEED + k)
        rng = np.random.default_rng(SEED + k)
        samp = crops[rng.choice(tr, min(len(tr), 3000), replace=False)].astype(np.float32) / 255.0
        mean_t = torch.tensor(samp.mean(axis=(0, 2, 3)), dtype=torch.float32).view(1, 4, 1, 1)
        std_t = torch.tensor(samp.std(axis=(0, 2, 3)) + 1e-6, dtype=torch.float32).view(1, 4, 1, 1)
        del samp
        Xtr = torch.from_numpy(crops[tr])          # uint8, normalised per batch
        ytr = torch.from_numpy(y[tr]).long()
        Xte = torch.from_numpy(crops[te_ok])

        def prep(xb_u8):
            return (xb_u8.float() / 255.0 - mean_t) / std_t
        counts = np.bincount(y[tr], minlength=K).astype(float)
        w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
        w = w / w[counts > 0].mean()
        model = _make_cnn(torch, K).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
        steps = CNN_EPOCHS * int(np.ceil(len(tr) / CNN_BATCH))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CNN_LR, total_steps=steps)
        lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev),
                                          label_smoothing=0.05)

        def augment(xb):
            kk = int(rng.integers(4))
            if kk:
                xb = torch.rot90(xb, kk, dims=(2, 3))
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=(3,))
            sx, sy = (int(v) for v in rng.integers(-3, 4, size=2))
            xb = torch.roll(xb, shifts=(sy, sx), dims=(2, 3))
            gain = 1 + 0.1 * torch.randn(xb.shape[0], xb.shape[1], 1, 1)
            bias = 0.1 * torch.randn(xb.shape[0], xb.shape[1], 1, 1)
            return xb * gain + bias

        # quick sanity check: one forward/backward pass before the long run
        if k == 0:
            xb = augment(prep(Xtr[:8])).to(dev)
            out, h = model(xb)
            assert out.shape == (8, K) and h.shape == (8, 4 * CNN_WIDTH), (out.shape, h.shape)
            lossf(out, ytr[:8].to(dev)).backward()
            opt.zero_grad()
            log(f"CNN sanity check OK: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters, "
                f"{K} classes, device {dev}")

        for ep in range(CNN_EPOCHS):
            model.train()
            t0 = time.time()
            perm = torch.from_numpy(rng.permutation(len(tr)))
            tot, nb = 0.0, 0
            for b in range(0, len(tr), CNN_BATCH):
                idx = perm[b:b + CNN_BATCH]
                xb = augment(prep(Xtr[idx])).to(dev)
                out, _ = model(xb)
                loss = lossf(out, ytr[idx].to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss)
                nb += 1
            if ep == 0 or (ep + 1) % 3 == 0 or ep == CNN_EPOCHS - 1:
                el = time.time() - t_all
                done = k * CNN_EPOCHS + ep + 1
                log(f"  fold {k + 1}/{n_splits} epoch {ep + 1}/{CNN_EPOCHS}: loss {tot / max(nb, 1):.3f} "
                    f"({time.time() - t0:.0f}s/epoch, ~{el / done * (n_splits * CNN_EPOCHS - done) / 60:.0f} min left)")
        # predict held-out donors with 4-rotation test-time augmentation
        model.eval()
        probs, feats = [], []
        with torch.inference_mode():
            for b in range(0, len(te_ok), 256):
                xb = prep(Xte[b:b + 256]).to(dev)
                ps, hs = 0, 0
                for kk in range(4):
                    out, h = model(torch.rot90(xb, kk, dims=(2, 3)))
                    ps = ps + torch.softmax(out, 1)
                    hs = hs + h
                probs.append((ps / 4).cpu().numpy())
                feats.append((hs / 4).cpu().numpy())
        if len(te_ok):
            P[te_ok] = np.vstack(probs)
            F[te_ok] = np.vstack(feats)
        acc = (P[te_ok].argmax(1) == y[te_ok]).mean() if len(te_ok) else np.nan
        log(f"  fold {k + 1}/{n_splits}: test donors {sorted(set(donors[te]))}, top-1 {acc:.3f}")
    # isolates without a crop: uniform probabilities, mean features
    P[~ok] = 1.0 / K
    F[~ok] = np.nanmean(F[ok], axis=0)
    top1 = float((P[ok].argmax(1) == y[ok]).mean())
    top5 = float((np.argsort(-P[ok], 1)[:, :5] == y[ok][:, None]).any(1).mean())
    log(f"CNN OOF: top1={top1:.3f} top5={top5:.3f} ({(time.time() - t_all) / 60:.0f} min)")
    np.savez(cache, P=P, F=F, classes=np.array(classes, dtype=object),
             ids=np.array(df["isolate_id"].astype(str), dtype=object), top1=top1, top5=top5)
    return P, F, {"top1_accuracy": top1, "top5_accuracy": top5}


# =============================================================================
# Picking strategies
# =============================================================================
def n_unique_curve(order, labels, grid):
    seen, cum = set(), []
    for j in order:
        seen.add(labels[j])
        cum.append(len(seen))
    return [cum[n - 1] if n <= len(cum) else np.nan for n in grid]


def maxmin_select(D, n, rng):
    """CAMII farthest-points local search (best of MAXMIN_RESTARTS)."""
    N = D.shape[0]
    if n >= N:
        return np.arange(N)
    if n == 1:
        return np.array([rng.integers(N)])
    best, best_val = None, -np.inf
    for _ in range(MAXMIN_RESTARTS):
        r = list(rng.choice(N, n, replace=False))
        for _it in range(MAXMIN_MAX_ITERS):
            changed = False
            for i in range(n):
                others = r[:i] + r[i + 1:]
                md = D[:, others].min(1)
                md[others] = -np.inf
                new = int(np.argmax(md))
                if new != r[i]:
                    r[i] = new
                    changed = True
            if not changed:
                break
        sub = D[np.ix_(r, r)]
        val = sub[np.triu_indices(n, 1)].min()
        if val > best_val:
            best_val, best = val, list(r)
    return np.array(best)


def facility_order(S, nmax):
    """Greedy facility location: maximise sum_i max_{j in S} sim(i,j)."""
    N = S.shape[0]
    b = np.zeros(N)
    avail = np.ones(N, bool)
    order = []
    for _ in range(min(nmax, N)):
        gains = np.maximum(S - b[:, None], 0).sum(0)
        gains[~avail] = -np.inf
        j = int(np.argmax(gains))
        order.append(j)
        avail[j] = False
        b = np.maximum(b, S[:, j])
    return order


def rbf_similarity(Xg):
    D = squareform(pdist(Xg))
    sigma = np.median(D[np.triu_indices(len(D), 1)]) if len(D) > 1 else 1.0
    sigma = sigma if sigma > 0 else 1.0
    return np.exp(-(D ** 2) / (2 * (0.5 * sigma) ** 2)), D


def novelty_order(Pg, classes, nmax):
    """Greedy expected number of distinct taxa: gain_i = sum_k w_k p_ik prod_{j in S}(1-p_jk)."""
    K = len(classes)
    other = classes.index("__other__")
    w = np.ones(K)
    if TARGET_WEIGHTS:
        for k, c in enumerate(classes):
            w[k] = TARGET_WEIGHTS.get(c, 0.1)
    w[other] = 0.0
    remaining = np.ones(K)
    avail = np.ones(len(Pg), bool)
    order = []
    for _ in range(min(nmax, len(Pg))):
        gain = (Pg * w * remaining).sum(1) + NOVELTY_OTHER_WEIGHT * Pg[:, other]
        gain[~avail] = -np.inf
        j = int(np.argmax(gain))
        order.append(j)
        avail[j] = False
        remaining *= (1 - Pg[j])
    return order


# =============================================================================
# Extra picking strategies: calibration, hybrid, colony rarity, adaptive rounds
# =============================================================================
_COLONY_TABLE = None      # all visually captured colonies (set in attach_colony_context)


def calibrate_power(P, y_idx, folds, grid=None):
    """Honest probability calibration: p -> p^g / sum, with g for each fold chosen on the OTHER folds'
    out-of-fold predictions (maximum log-likelihood). Returns calibrated P and the chosen exponents."""
    grid = grid if grid is not None else np.round(np.arange(0.1, 3.01, 0.05), 2)
    eps = 1e-6
    Pc = P.copy()
    gammas = []

    def powered(M, g):
        Q = np.power(M + eps, g)
        return Q / Q.sum(1, keepdims=True)

    for k, (_, te) in enumerate(folds):
        others = np.concatenate([folds[j][1] for j in range(len(folds)) if j != k])
        ll = [np.mean(np.log(powered(P[others], g)[np.arange(len(others)), y_idx[others]])) for g in grid]
        g = float(grid[int(np.argmax(ll))])
        gammas.append(g)
        Pc[te] = powered(P[te], g)
    return Pc, gammas


def novelty_gain_vector(Pg, classes, remaining):
    w = np.ones(len(classes))
    if TARGET_WEIGHTS:
        for k, c in enumerate(classes):
            w[k] = TARGET_WEIGHTS.get(c, 0.1)
    other = classes.index("__other__")
    w[other] = 0.0
    return (Pg * w * remaining).sum(1) + NOVELTY_OTHER_WEIGHT * Pg[:, other]


def hybrid_order(Pg, classes, D, nmax, lam):
    """(1-lam) * normalised expected novelty + lam * normalised distance to the closest already chosen colony."""
    N = len(Pg)
    remaining = np.ones(len(classes))
    avail = np.ones(N, bool)
    mind = np.full(N, np.inf)
    order = []
    for _ in range(min(nmax, N)):
        nov = novelty_gain_vector(Pg, classes, remaining)
        nov_n = nov / (nov[avail].max() + 1e-12)
        if order:
            md = np.where(np.isfinite(mind), mind, 0)
            md_n = md / (md[avail].max() + 1e-12)
        else:
            md_n = np.zeros(N)
        gain = (1 - lam) * nov_n + lam * md_n
        gain[~avail] = -np.inf
        j = int(np.argmax(gain))
        order.append(j)
        avail[j] = False
        remaining *= (1 - Pg[j])
        mind = np.minimum(mind, D[:, j])
    return order


def rarity_scores(df, feats):
    """kNN distance of each random-picked isolate to all captured colonies of the same picking group
    (z-scored log features). Large = morphologically rare on its own plates. NaN if unavailable."""
    out = np.full(len(df), np.nan)
    col = _COLONY_TABLE
    if col is None or "plate" not in df:
        return out
    present = [f for f in feats if f in col.columns]
    if len(present) < 4:
        return out
    plate2group = df.dropna(subset=["plate"]).groupby("plate")["group"].agg(lambda s: s.mode().iat[0])
    c = col.assign(group=col["plate"].map(plate2group)).dropna(subset=["group"])
    for g, sub in df[df["is_random"]].groupby("group"):
        cg = c[c["group"] == g]
        if len(cg) < RARITY_K + 5:
            continue
        Xc = log_features(cg, present)
        ok = np.isfinite(Xc).all(1)
        Xc = Xc[ok]
        mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
        tree = cKDTree((Xc - mu) / sd)
        Xi = (log_features(sub, present) - mu) / sd
        Xi = np.nan_to_num(Xi)
        dist, _ = tree.query(Xi, k=RARITY_K + 1)
        dist = np.where(dist[:, :1] < 1e-6, dist[:, 1:], dist[:, :-1])   # drop the colony itself
        out[sub.index.values] = dist.mean(1)
    return out


def rarity_order(r, D, Pg, classes, nmax, mode):
    """mode 'maxmin': gain = rarity_rank * distance to closest chosen; 'novelty': gain = novelty * rarity_rank."""
    N = len(r)
    rank = (pd.Series(r).rank(pct=True).fillna(0.5)).values
    avail = np.ones(N, bool)
    mind = np.full(N, np.inf)
    remaining = np.ones(len(classes)) if Pg is not None else None
    order = []
    for _ in range(min(nmax, N)):
        if mode == "novelty":
            gain = novelty_gain_vector(Pg, classes, remaining) * rank
        else:
            md = np.where(np.isfinite(mind), mind, 1.0)
            gain = rank * md
        gain[~avail] = -np.inf
        j = int(np.argmax(gain))
        order.append(j)
        avail[j] = False
        mind = np.minimum(mind, D[:, j])
        if remaining is not None:
            remaining *= (1 - Pg[j])
    return order


def _fit_same_vs_distance(D, picked, labels):
    """Logistic P(same taxon | feature distance) fitted on revealed pairs of THIS donor's picks."""
    if len(picked) < 4:
        return None
    from sklearn.linear_model import LogisticRegression
    p = np.array(picked)
    iu = np.triu_indices(len(p), 1)
    d = D[p[iu[0]], p[iu[1]]]
    same = (labels[p[iu[0]]] == labels[p[iu[1]]]).astype(int)
    if same.sum() == 0 or same.sum() == len(same):
        return None
    lr = LogisticRegression(C=1.0).fit(d[:, None], same)
    a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    if b >= 0:            # farther must not mean more similar; no usable signal
        return None
    return a, b


def adaptive_order(D, Pg, classes, labels_div, labels_clf_idx, nmax, use_clf=True):
    """Picks in rounds of ROUND_SIZE. After each round the taxa of the picked colonies are revealed:
    - classifier part: revealed classes count as caught (hard), unrevealed picks discount softly;
    - donor part: P(same | distance) is re-fitted on this donor's revealed pairs and used to estimate
      the chance that a candidate duplicates any already chosen colony.
    gain = novelty^(1-alpha) * (1 - p_duplicate)^alpha (donor part only once it can be fitted)."""
    N = len(D)
    K = len(classes)
    other = classes.index("__other__")
    picked, model = [], None
    while len(picked) < min(nmax, N):
        round_sel = []
        for _ in range(min(ROUND_SIZE, nmax - len(picked), N - len(picked))):
            chosen = picked + round_sel
            avail = np.ones(N, bool)
            avail[chosen] = False
            if model is not None and chosen:
                a, b = model
                ps = 1.0 / (1.0 + np.exp(-(a + b * D[:, chosen])))
                p_new_donor = np.prod(1 - ps, axis=1)
            else:
                p_new_donor = None
            if use_clf:
                remaining = np.ones(K)
                caught = {labels_clf_idx[j] for j in picked} - {other}
                if caught:
                    remaining[list(caught)] = 0.0
                for j in round_sel:
                    remaining *= (1 - Pg[j])
                nov = novelty_gain_vector(Pg, classes, remaining)
                gain = nov if p_new_donor is None else \
                    np.power(nov + 1e-12, 1 - ADAPTIVE_ALPHA) * np.power(p_new_donor, ADAPTIVE_ALPHA)
            else:
                if p_new_donor is not None:
                    gain = p_new_donor
                elif chosen:
                    gain = D[:, chosen].min(1)
                else:
                    gain = np.abs(D).mean(1)            # start from the most atypical colony
            gain = np.asarray(gain, float).copy()
            gain[~avail] = -np.inf
            j = int(np.argmax(gain))
            round_sel.append(j)
        picked += round_sel
        model = _fit_same_vs_distance(D, picked, labels_div)   # reveal round, re-fit donor model
    return picked


def simulate(df, store, reps, probs, classes, outdir):
    rng = np.random.default_rng(SEED)
    rows = []
    groups = df[df["is_random"]].groupby("group")
    n_used = 0
    pools = []
    for g, sub in groups:
        idx_all = sub.index.values
        if len(idx_all) < MIN_POOL:
            continue
        if POOL_SUBSAMPLE_SIZE and len(idx_all) > POOL_SUBSAMPLE_SIZE:
            for rr in range(POOL_SUBSAMPLE_REPEATS or 1):
                pools.append((f"{g}#pool{rr + 1}", np.sort(rng.choice(idx_all, POOL_SUBSAMPLE_SIZE, replace=False))))
        else:
            pools.append((g, idx_all))
    t0 = time.time()
    for g, idx in pools:
        sub = df.loc[idx]
        N = len(idx)
        n_used += 1
        if n_used % 10 == 0:
            log(f"  pool {n_used}/{len(pools)} ({time.time() - t0:.0f}s)")
        labels = df.loc[idx, "label_div"].values
        n_true = len(set(labels))
        grid = [n for n in N_GRID if n <= N]
        nmax = max(grid)
        res = {}

        # random
        curves = [n_unique_curve(rng.permutation(N), labels, grid) for _ in range(RANDOM_REPEATS)]
        res["random"] = np.nanmean(np.array(curves, float), 0)
        # oracle
        res["oracle"] = [min(n, n_true) for n in grid]

        # max-min (CAMII and variants)
        for rep in [r for r in reps if r in ("CAMII_PCA2", "full14_log_z", "embedding", "cnn_feat")]:
            Xg = group_rep(rep, idx, store)
            if not np.isfinite(Xg).all():
                continue
            D = squareform(pdist(Xg))
            vals = []
            for n in grid:
                sel = maxmin_select(D, n, rng)
                vals.append(len(set(labels[sel])))
            name = "CAMII (PCA2 + max-min)" if rep == "CAMII_PCA2" else f"max-min [{rep}]"
            res[name] = vals

        # facility location on feature spaces
        for rep in [r for r in reps if r in ("full14_log_z", "context_corr_z", "embedding", "embedding+morph")]:
            Xg = group_rep(rep, idx, store)
            if not np.isfinite(Xg).all():
                continue
            S, _ = rbf_similarity(Xg)
            res[f"facility [{rep}]"] = n_unique_curve(facility_order(S, nmax), labels, grid)

        # classifier-based
        for pname, Pm in (probs or {}).items():
            Pg = Pm[idx]
            if pname == "clf":
                res["facility [clf similarity]"] = n_unique_curve(facility_order(Pg @ Pg.T, nmax), labels, grid)
            res[f"expected novelty [{pname}]"] = n_unique_curve(novelty_order(Pg, classes, nmax), labels, grid)

        if NEW_STRATEGIES and "clf" in (probs or {}):
            D = squareform(pdist(group_rep("full14_log_z", idx, store)))
            Pg = probs["clf"][idx]          # raw RF probabilities (calibration is tested separately)
            res["hybrid [clf + distance]"] = n_unique_curve(
                hybrid_order(Pg, classes, D, nmax, HYBRID_LAMBDA), labels, grid)
            if "rarity" in store and np.isfinite(store["rarity"][idx]).mean() > 0.8:
                r = store["rarity"][idx]
                res["rarity x novelty [clf]"] = n_unique_curve(
                    rarity_order(r, D, Pg, classes, nmax, "novelty"), labels, grid)
                res["rarity + max-min"] = n_unique_curve(rarity_order(r, D, None, classes, nmax, "maxmin"),
                                                         labels, grid)
            lab_clf = store["y_clf_idx"][idx]
            res["adaptive rounds [clf + donor]"] = n_unique_curve(
                adaptive_order(D, Pg, classes, labels, lab_clf, nmax, use_clf=True), labels, grid)
            res["adaptive rounds [donor only]"] = n_unique_curve(
                adaptive_order(D, Pg, classes, labels, lab_clf, nmax, use_clf=False), labels, grid)

        donor = sub["donor"].iloc[0]
        for strat, vals in res.items():
            for n, v in zip(grid, vals):
                rows.append({"group": g, "donor": donor, "pool_size": N, "n_unique_in_pool": n_true,
                             "strategy": strat, "n_picked": n, "n_unique": v})
    log(f"simulation: {n_used} candidate pools from {df[df['is_random']]['group'].nunique()} groups with random picks"
        f"{' (subsampled to ' + str(POOL_SUBSAMPLE_SIZE) + ')' if POOL_SUBSAMPLE_SIZE else ''}")
    out = pd.DataFrame(rows)
    out.to_csv(Path(outdir) / "picking_curves_per_group.csv", index=False)
    return out


# =============================================================================
# Pair AUC (same vs different taxon within a group)
# =============================================================================
def pair_auc(df, store, reps, probs, outdir):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    sel = df if PAIR_EVAL_USE_ALL_ISOLATES else df[df["is_random"]]
    probs = probs or {}
    collected = {r: ([], []) for r in reps + [f"{p}_prob" for p in probs]}
    donors_per_pair = []
    for g, sub in sel.groupby("group"):
        idx = sub.index.values
        if len(idx) < 5:
            continue
        lab = df.loc[idx, "label_div"].values
        iu = np.triu_indices(len(idx), 1)
        diff = (lab[iu[0]] != lab[iu[1]]).astype(int)
        if diff.min() == diff.max():
            continue
        take = np.arange(len(diff))
        if len(take) > MAX_PAIRS_PER_GROUP:
            take = rng.choice(len(take), MAX_PAIRS_PER_GROUP, replace=False)
        donors_per_pair.extend([sub["donor"].iloc[0]] * len(take))
        for r in reps:
            Xg = group_rep(r, idx, store)
            if not np.isfinite(Xg).all():
                d = np.full(len(take), np.nan)
            else:
                d = pdist(Xg)[take]  # pdist order == triu order
            collected[r][0].append(d)
            collected[r][1].append(diff[take])
        for pname, Pm in probs.items():
            Pg = Pm[idx]
            same = (Pg @ Pg.T)[iu][take]
            collected[f"{pname}_prob"][0].append(1 - same)
            collected[f"{pname}_prob"][1].append(diff[take])
    rows = []
    donors_per_pair = np.array(donors_per_pair)
    for r, (scores, ys) in collected.items():
        if not scores:
            continue
        s = np.concatenate(scores)
        y = np.concatenate(ys)
        ok = np.isfinite(s)
        if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
            continue
        auc = roc_auc_score(y[ok], s[ok])
        per_d = []
        for d in np.unique(donors_per_pair):
            m = ok & (donors_per_pair == d)
            if m.sum() > 50 and len(np.unique(y[m])) == 2:
                per_d.append(roc_auc_score(y[m], s[m]))
        rows.append({"representation": r, "pair_AUC": auc, "mean_donor_AUC": np.mean(per_d) if per_d else np.nan,
                     "sd_donor_AUC": np.std(per_d) if per_d else np.nan, "n_pairs": int(ok.sum()),
                     "frac_pairs_different": float(y[ok].mean())})
    out = pd.DataFrame(rows).sort_values("pair_AUC", ascending=False)
    out.to_csv(Path(outdir) / "representation_pair_auc.csv", index=False)
    log("\nREPRESENTATIONS — ROC-AUC 'different taxon' from distance (pairs within group):")
    log(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return out


# =============================================================================
# Summaries and plots
# =============================================================================
def summarise(curves, outdir):
    oracle = curves[curves.strategy == "oracle"][["group", "n_picked", "n_unique"]].rename(columns={"n_unique": "oracle"})
    c = curves.merge(oracle, on=["group", "n_picked"])
    c["frac_of_oracle"] = c["n_unique"] / c["oracle"]
    summ = (c.groupby(["strategy", "n_picked"])
              .agg(mean_unique=("n_unique", "mean"), mean_frac_oracle=("frac_of_oracle", "mean"),
                   n_groups=("group", "nunique"))
              .reset_index())
    summ.to_csv(Path(outdir) / "picking_curves_summary.csv", index=False)

    # paired comparison vs CAMII and vs random at each n
    tests = []
    for ref in ["CAMII (PCA2 + max-min)", "random"]:
        if ref not in set(c.strategy):
            continue
        r = c[c.strategy == ref][["group", "n_picked", "n_unique"]].rename(columns={"n_unique": "ref"})
        for strat in sorted(set(c.strategy) - {ref, "oracle"}):
            m = c[c.strategy == strat][["group", "n_picked", "n_unique"]].merge(r, on=["group", "n_picked"])
            for n, mm in m.groupby("n_picked"):
                d = (mm["n_unique"] - mm["ref"]).values
                if len(d) < 5:
                    continue
                # NOTE: subsampled pools from the same group are not independent -> p is optimistic
                p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                tests.append({"strategy": strat, "vs": ref, "n_picked": n, "n_groups": len(d),
                              "mean_gain_unique": d.mean(), "wilcoxon_p": p})
    tests = pd.DataFrame(tests)
    tests.to_csv(Path(outdir) / "picking_paired_tests.csv", index=False)

    log("\nPICKING — mean unique taxa (fraction of oracle) per strategy:")
    show_n = [n for n in [10, 20, 30, 50] if n in set(summ.n_picked)]
    tab = summ[summ.n_picked.isin(show_n)].pivot(index="strategy", columns="n_picked", values="mean_unique")
    frac = summ[summ.n_picked.isin(show_n)].pivot(index="strategy", columns="n_picked", values="mean_frac_oracle")
    tab = tab.sort_values(show_n[-1] if show_n else tab.columns[-1], ascending=False)
    frac = frac.reindex(tab.index)
    pretty = tab.copy().astype(object)
    for col in tab.columns:
        pretty[col] = [f"{a:.1f} ({b:.0%})" for a, b in zip(tab[col], frac[col])]
    pretty.columns = [f"n={n}" for n in pretty.columns]
    log(pretty.to_string())
    ngr = summ[summ.n_picked.isin(show_n)].groupby("n_picked")["n_groups"].max().to_dict()
    log(f"(groups contributing per n: {ngr})")
    if not tests.empty:
        t = tests[(tests.vs == "CAMII (PCA2 + max-min)") & tests.n_picked.isin(show_n)]
        if not t.empty:
            log("\nGain vs CAMII (mean extra unique taxa, Wilcoxon p):")
            log(t.pivot(index="strategy", columns="n_picked", values="mean_gain_unique").round(2).to_string())
    return summ, tests


def plots(summ, auc, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed — no plots")
        return
    style = {"oracle": dict(color="black", ls=":", lw=1.5), "random": dict(color="grey", ls="--", lw=1.5),
             "CAMII (PCA2 + max-min)": dict(color="#d62728", lw=2.2)}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ci = 0
    for strat, s in summ.groupby("strategy"):
        s = s[s.n_groups >= 3]
        if strat in style:
            kw = style[strat]
        else:
            kw = dict(color=palette[ci % len(palette)], lw=1.6)
            ci += 1
        axes[0].plot(s.n_picked, s.mean_unique, marker="o", ms=3, label=strat, **kw)
        axes[1].plot(s.n_picked, s.mean_frac_oracle, marker="o", ms=3, label=strat, **kw)
    axes[0].set_xlabel("colonies picked per group")
    axes[0].set_ylabel(f"mean unique {DIVERSITY_LEVEL.upper()}s")
    axes[1].set_xlabel("colonies picked per group")
    axes[1].set_ylabel("fraction of oracle")
    axes[1].set_ylim(0, 1.02)
    for a in axes:
        a.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Offline picking simulation within random-picked pools (n plotted where ≥3 groups)")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "picking_curves.png", dpi=150)
    plt.close(fig)

    if auc is not None and not auc.empty:
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(auc) + 1.5))
        a = auc.sort_values("pair_AUC")
        cols = ["#d62728" if r == "CAMII_PCA2" else "#1f77b4" for r in a.representation]
        ax.barh(a.representation, a.pair_AUC, xerr=a.sd_donor_AUC.fillna(0), color=cols)
        ax.axvline(0.5, color="grey", ls="--")
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("ROC-AUC: distance separates different-taxon pairs (error = SD across donors)")
        fig.tight_layout()
        fig.savefig(Path(outdir) / "representation_pair_auc.png", dpi=150)
        plt.close(fig)
    log(f"\nplots saved to {outdir}")


# =============================================================================
# RUN
# =============================================================================
def run(outdir=None, cache_dir=None):
    t_start = time.time()
    outdir = Path(outdir or PATHS["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir or outdir)
    log("=" * 78)
    log("RUN — building dataset")
    log("=" * 78)
    df, feats = build_dataset()
    df, has_ctx = attach_colony_context(df)
    if REQUIRE_IMAGES:
        ok_plates = plates_with_images(cache_dir)
        if ok_plates is None:
            log("REQUIRE_IMAGES=True but raw_image_dir not set — using all isolates")
        else:
            keep = df["plate"].astype(str).isin(ok_plates) if "plate" in df else pd.Series(True, index=df.index)
            dropped = df.loc[~keep]
            if len(dropped):
                lost = dropped[dropped["is_random"]].groupby("group").size().to_dict()
                log(f"REQUIRE_IMAGES: dropping {len(dropped)} isolates on {dropped['plate'].nunique()} plates "
                    f"without images; random-pick isolates dropped per group: {lost or 'none'}")
            df = df.loc[keep].reset_index(drop=True)

    store = {"raw": df[feats].values.astype(float), "log": log_features(df, feats)}
    reps = ["CAMII_PCA2", "full14_z", "full14_log_z"]
    if has_ctx:
        log("context correction:")
        store["ctx"] = context_corrected(df, feats)
        reps.append("context_corr_z")

    emb = compute_embeddings(df, cache_dir)
    if emb is not None and np.isfinite(emb).any():
        E, emb_ok = reduce_embeddings(emb)
        log(f"embedding PCA: {E.shape[1]} components; isolates without image get the mean (0) "
            f"— {int((~emb_ok).sum())} of {len(emb_ok)}")
        store["emb"] = np.nan_to_num(E)
        reps += ["embedding", "embedding+morph"]

    # classifier inputs: log features (+ context) (+ media one-hot) (+ embedding PCs)
    Xc = [zscore(store["ctx"] if has_ctx else store["log"])]
    if "media" in df:
        Xc.append(pd.get_dummies(df["media"].astype(str)).values.astype(float))
    if "emb" in store:
        Xc.append(np.nan_to_num(store["emb"]))
    Xc = np.hstack(Xc)

    log("\n" + "=" * 78)
    log("CLASSIFIER (out-of-fold by donor)")
    log("=" * 78)
    P, classes, clf_metrics = oof_classifier(df, Xc, outdir)
    probs = {"clf": P}
    if NEW_STRATEGIES:
        from sklearn.model_selection import GroupKFold
        cidx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([cidx.get(v, cidx["__other__"]) for v in df["label_clf"].fillna("__other__").astype(str)])
        folds = list(GroupKFold(min(CLF_N_SPLITS, df["donor"].nunique())).split(
            np.zeros(len(df)), y_idx, df["donor"].values))
        P_cal, gammas = calibrate_power(P, y_idx, folds)
        ll0 = np.mean(np.log(P[np.arange(len(P)), y_idx] + 1e-6))
        ll1 = np.mean(np.log(P_cal[np.arange(len(P)), y_idx] + 1e-6))
        log(f"calibration exponents per fold: {gammas}; mean log-likelihood {ll0:.3f} -> {ll1:.3f}")
        probs["clf_cal"] = P_cal
        store["y_clf_idx"] = y_idx
        store["rarity"] = rarity_scores(df, feats)
        rr = df[df["is_random"]].copy()
        rr["rarity"] = store["rarity"][rr.index.values]
        rr = rr.dropna(subset=["rarity"])
        if len(rr) > 30:
            rr["label_freq"] = rr.groupby(["group", "label_div"])["label_div"].transform("size") / \
                rr.groupby("group")["label_div"].transform("size")
            rho = stats.spearmanr(rr["rarity"], rr["label_freq"]).correlation
            log(f"colony rarity: computed for {len(rr)} random-pick isolates; Spearman(rarity, how common its "
                f"taxon is in the pool) = {rho:+.3f} (negative = rare-looking colonies are rarer taxa)")
        else:
            log("colony rarity: not available (needs colony_metadata) — rarity strategies skipped")
    if USE_CNN:
        log("\n" + "=" * 78)
        log("CNN on colony crops (out-of-fold by donor, same folds as the classifier)")
        log("=" * 78)
        crops, crop_ok = build_cnn_crops(df, cache_dir)
        P_cnn, F_cnn, cnn_metrics = oof_cnn(df, crops, crop_ok, classes, cache_dir, outdir)
        del crops
        if P_cnn is not None:
            probs["cnn"] = P_cnn
            probs["rf+cnn"] = 0.5 * (P + P_cnn)
            store["cnn"] = F_cnn
            reps.append("cnn_feat")
            y_chk = df["label_clf"].fillna("__other__").astype(str).map(
                lambda v: v if v in set(classes) else "__other__").values
            ens = np.array(classes)[probs["rf+cnn"].argmax(1)]
            clf_metrics["cnn_top1"] = cnn_metrics["top1_accuracy"]
            clf_metrics["cnn_top5"] = cnn_metrics["top5_accuracy"]
            clf_metrics["rf+cnn_top1"] = float((ens == y_chk).mean())
            log(f"top-1: RF {clf_metrics['top1_accuracy']:.3f} | CNN {cnn_metrics['top1_accuracy']:.3f} | "
                f"RF+CNN {clf_metrics['rf+cnn_top1']:.3f}")
    if CLASSIFIER_LEVEL != DIVERSITY_LEVEL:
        log(f"NOTE: classifier predicts {CLASSIFIER_LEVEL}, diversity is counted at {DIVERSITY_LEVEL}.")

    log("\n" + "=" * 78)
    log("REPRESENTATION EVALUATION")
    log("=" * 78)
    auc = pair_auc(df, store, reps, probs, outdir)

    log("\n" + "=" * 78)
    log("PICKING SIMULATION")
    log("=" * 78)
    curves = simulate(df, store, reps, probs, classes, outdir)
    if curves.empty:
        log(f"No group has a random pool >= {MIN_POOL}. Lower MIN_POOL, change GROUP_COLS "
            f"or check RANDOM_STRATEGY_PATTERN.")
        return
    summ, tests = summarise(curves, outdir)
    plots(summ, auc, outdir)

    df[["isolate_id", "donor", "group", "is_random", "label_div", "label_clf"]].to_csv(
        outdir / "dataset_used.csv", index=False)
    with open(outdir / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"isolates: {len(df)}, donors: {df.donor.nunique()}, groups: {df.group.nunique()}, "
                 f"random-picked: {int(df.is_random.sum())}\n")
        fh.write(f"diversity level: {DIVERSITY_LEVEL}, classifier level: {CLASSIFIER_LEVEL}\n")
        fh.write(f"classifier: {clf_metrics}\n\n")
        fh.write("Representation pair AUC:\n" + auc.to_string(index=False) + "\n\n")
        fh.write("Picking summary:\n" + summ.to_string(index=False) + "\n")
    log(f"\nDone in {time.time() - t_start:.0f}s. Results in: {outdir}")
    return {"summ": summ, "auc": auc, "clf": clf_metrics, "by_group": by_source_group(curves, outdir)}


def by_source_group(curves, outdir):
    """Mean unique taxa per ORIGINAL group (before pool subsampling) — shows whether a gain is
    consistent across donors or driven by one of them."""
    c = curves.copy()
    c["source_group"] = c["group"].astype(str).str.split("#").str[0]
    t = c.pivot_table(index=["source_group", "n_picked"], columns="strategy", values="n_unique", aggfunc="mean")
    t.to_csv(Path(outdir) / "picking_by_source_group.csv")
    n = 50 if 50 in set(c.n_picked) else max(c.n_picked)
    cols = [x for x in ["random", "CAMII (PCA2 + max-min)", "expected novelty [clf]",
                        "expected novelty [cnn]", "expected novelty [rf+cnn]", "max-min [cnn_feat]",
                        "expected novelty [clf_cal]", "hybrid [clf + distance]", "rarity x novelty [clf]",
                        "adaptive rounds [clf + donor]", "adaptive rounds [donor only]",
                        "max-min [embedding]", "facility [embedding+morph]", "oracle"] if x in t.columns]
    log(f"\nPer source group, n={n} (mean unique taxa):")
    log(t.xs(n, level="n_picked")[cols].round(1).to_string())
    return t


def run_experiments():
    base_out = Path(PATHS["output_dir"])
    base_out.mkdir(parents=True, exist_ok=True)
    defaults = {k: globals()[k] for e in EXPERIMENTS for k in e if k != "name"}
    rows, aucs = [], []
    for e in EXPERIMENTS:
        name = e["name"]
        if name in SKIP_EXPERIMENTS:
            log(f"skipping {name}")
            continue
        globals().update(defaults)                       # reset to CONFIG values
        globals().update({k: v for k, v in e.items() if k != "name"})
        log("\n" + "#" * 78)
        log(f"# EXPERIMENT {name}: " + ", ".join(f"{k}={v}" for k, v in e.items() if k != "name"))
        log("#" * 78)
        try:
            res = run(outdir=base_out / name, cache_dir=base_out)
        except Exception as ex:                          # keep going with the other experiments
            import traceback
            traceback.print_exc()
            log(f"!! experiment {name} failed: {ex}")
            continue
        if not res:
            continue
        s = res["summ"]
        for n in [20, 50]:
            ss = s[s.n_picked == n].set_index("strategy")
            if ss.empty:
                continue
            for strat in ss.index:
                rows.append({"experiment": name, "level": DIVERSITY_LEVEL, "n_picked": n, "strategy": strat,
                             "mean_unique": ss.loc[strat, "mean_unique"],
                             "frac_oracle": ss.loc[strat, "mean_frac_oracle"],
                             "gain_vs_random": ss.loc[strat, "mean_unique"] - ss.loc["random", "mean_unique"],
                             "gain_vs_CAMII": ss.loc[strat, "mean_unique"] - ss.loc["CAMII (PCA2 + max-min)", "mean_unique"],
                             "clf_top1": res["clf"]["top1_accuracy"], "clf_top5": res["clf"]["top5_accuracy"]})
        a = res["auc"].copy()
        a.insert(0, "experiment", name)
        aucs.append(a)
    globals().update(defaults)
    if not rows:
        return
    comp = pd.DataFrame(rows)
    # keep results of experiments from earlier runs that were skipped this time
    for fname, new in [("experiments_comparison.csv", comp), ("experiments_pair_auc.csv",
                                                             pd.concat(aucs) if aucs else None)]:
        f = base_out / fname
        if new is None:
            continue
        if f.exists():
            old = pd.read_csv(f)
            new = pd.concat([old[~old.experiment.isin(new.experiment.unique())], new], ignore_index=True)
        new.to_csv(f, index=False)
        if fname == "experiments_comparison.csv":
            comp = new
        else:
            aucs = [new]
    log("\n" + "=" * 78)
    log("ALL EXPERIMENTS — gain over CAMII in unique taxa (n=50); classifier top-1")
    log("=" * 78)
    t = comp[(comp.n_picked == 50) & ~comp.strategy.isin(["oracle", "CAMII (PCA2 + max-min)"])]
    if not t.empty:
        log(t.pivot(index="strategy", columns="experiment", values="gain_vs_CAMII").round(2).to_string())
        log(comp.groupby("experiment")[["clf_top1", "clf_top5"]].first().round(3).to_string())
    if aucs:
        A = pd.concat(aucs)
        log("\nPair AUC per experiment:")
        log(A.pivot(index="representation", columns="experiment", values="pair_AUC").round(3).to_string())
    log(f"\nComparison saved to {base_out / 'experiments_comparison.csv'}")


if __name__ == "__main__":
    if "--inspect" in sys.argv:
        MODE = "inspect"
    elif "--run" in sys.argv:
        MODE = "run"
    if MODE == "inspect":
        inspect()
    elif EXPERIMENTS:
        run_experiments()
    else:
        run()

"""End-to-end: plate images -> detected colonies -> taxon probabilities -> pick list."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .detection import DetectionConfig, draw_colonies, FEATURES
from .features import plate_context
from .model import ColonyModel, OTHER
from .picking import expected_novelty_order, camii_maxmin_select, hybrid_order, _zscore


def pick_group(results: dict, model: ColonyModel, n: int, media: str, exclude_taxa=None,
               compare_camii: bool = False, seed: int = 0, tail: str = "none", switch_gain: float = 0.1):
    """Pick `n` colonies jointly from all plates in `results` (one picking group)."""
    colonies = pd.concat([r.colonies for r in results.values()], ignore_index=True)
    if colonies.empty:
        return colonies, colonies, None
    ctx = plate_context(colonies, colonies)
    P = model.predict_proba(colonies, ctx, media=media)
    classes = model.classes
    top = P.argmax(1)
    colonies["predicted_taxon"] = [model.taxon_label(c) for c in np.array(classes)[top]]
    colonies["predicted_prob"] = P[np.arange(len(P)), top]
    colonies["p_other"] = P[:, classes.index(OTHER)] if OTHER in classes else 0.0
    colonies["n_neighbors"] = ctx["n_neighbors"].to_numpy()

    # morphology for the tail: context-corrected log features (medium columns dropped), z-scored
    X = _zscore(model.builder.transform(colonies, ctx, media=media)[:, :len(FEATURES)])
    order, gains, k = hybrid_order(P, classes, X, n, tail=tail, switch_gain=switch_gain,
                                   exclude_taxa=model.resolve_taxa(exclude_taxa), seed=seed)
    colonies.attrs["phase1_picks"] = k
    picks = colonies.iloc[order].copy()
    picks.insert(0, "pick_rank", np.arange(1, len(order) + 1))
    picks["pick_phase"] = np.where(np.arange(len(order)) < k, "novelty", f"tail:{tail}")
    picks["expected_new_taxa"] = np.round(gains, 4)
    picks["expected_taxa_cumulative"] = np.round(np.nancumsum(gains), 3)

    camii = None
    if compare_camii:
        sel = camii_maxmin_select(colonies[FEATURES].to_numpy(float), n, seed=seed)
        camii = colonies.iloc[sel].copy()
        camii.insert(0, "pick_rank", np.arange(1, len(sel) + 1))
    return colonies, picks, camii


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name)).strip("_")


def make_sample_sheet(image_dir, metadata, out_path, n: int | None = None, log=print) -> pd.DataFrame:
    """Write a CAMII-style sample sheet for all plates in `image_dir`, grouping plates by sample and
    medium taken from the CAMII isolate metadata (sourcePlate, sourceSample, media).

    numGroupPick is `n` for every group, or, if `n` is None, the number of isolates CAMII actually
    picked from that group (useful for a like-for-like comparison). Plates without metadata get
    groupID "unassigned" and numGroupPick 0 (skipped); edit the sheet to include them."""
    from .data import _read, _clean
    from .detection import find_plate_images
    plates = sorted(find_plate_images(image_dir))
    meta = _read(metadata)
    meta["plate"] = _clean(meta["sourcePlate"])
    meta["sample"] = _clean(meta["sourceSample"]).str.replace(r"\s*\(.*\)\s*$", "", regex=True)
    meta["media_"] = _clean(meta["media"]).str.replace(r"\s+", " ", regex=True)
    per_plate = meta.dropna(subset=["plate"]).groupby("plate").agg(
        sample=("sample", lambda x: x.mode().iat[0] if x.notna().any() else "NA"),
        media=("media_", lambda x: x.mode().iat[0] if x.notna().any() else "NA"),
        n_isolates=("plate", "size"))
    rows = []
    for b in plates:
        if b in per_plate.index:
            r = per_plate.loc[b]
            rows.append({"barcode": b, "sampleID": r["sample"], "plateType": r["media"],
                         "groupID": _safe(f"{r['sample']}_{r['media']}"), "n_isolates": int(r["n_isolates"])})
        else:
            rows.append({"barcode": b, "sampleID": "", "plateType": "", "groupID": "unassigned", "n_isolates": 0})
    sheet = pd.DataFrame(rows)
    if sheet.empty:
        raise SystemExit(f"no <barcode>_<timestamp>.bmp image pairs in {image_dir}")
    tot = sheet.groupby("groupID")["n_isolates"].transform("sum")
    sheet["numGroupPick"] = np.where(sheet["groupID"] == "unassigned", 0, n if n else tot)
    sheet = sheet.drop(columns="n_isolates").sort_values(["groupID", "barcode"])
    sheet.to_csv(out_path, index=False)
    n_un = int((sheet["groupID"] == "unassigned").sum())
    log(f"{len(sheet)} plates -> {sheet.loc[sheet.groupID != 'unassigned', 'groupID'].nunique()} groups; "
        f"{n_un} plates without metadata (groupID 'unassigned', skipped). Written to {out_path}")
    return sheet


def _detect_cached(barcode, paths, cfg: DetectionConfig, cache_dir: Path):
    """Detect one plate, or load the result of an earlier run with the same settings.
    Images are not kept in memory; they are reloaded from disk when drawing."""
    import joblib
    from dataclasses import asdict
    from .detection import detect_plate
    cache = cache_dir / f"{_safe(barcode)}.joblib"
    key = {**asdict(cfg), "images": [str(p) for p in paths]}
    if cache.exists():
        try:
            k, res = joblib.load(cache)
            if k == key:
                return res, True
        except Exception:  # noqa: BLE001 - corrupt cache (e.g. interrupted write): recompute
            pass
    res = detect_plate(paths[0], paths[1], cfg, barcode=barcode)
    res.trans = res.epi = None
    joblib.dump((key, res), cache, compress=3)
    return res, False


def run_pick(image_dir, model_path, out_dir, n: int = 50, media: str = "mGAM", cfg: DetectionConfig | None = None,
             sample_sheet=None, batch_size: int | None = None, exclude_taxa=None, compare_camii: bool = False,
             tail: str = "none", switch_gain: float = 0.1, log=print):
    """Detect colonies and write pick lists, processing one batch (picking group) at a time.

    Batches:
      * with a sample sheet (columns barcode, groupID, numGroupPick[, plateType/media]): one batch per
        group, as in CAMII's ``sample_metadata.csv``; groups with numGroupPick 0 are skipped;
      * otherwise: consecutive blocks of `batch_size` plates (sorted by barcode), `n` picks each;
        without `batch_size`, all plates form one group.
    Each batch is detected, picked and written before the next one starts. Detection results are
    cached in ``<out_dir>/cache``, so an interrupted run continues where it stopped."""
    from .detection import find_plate_images
    cfg = cfg or DetectionConfig()
    out = Path(out_dir)
    plate_dir, cache_dir = out / "plates", out / "cache"
    for d in (out, plate_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    model = ColonyModel.load(model_path)
    log(f"model: {model.info.get('level')} classifier, {model.info.get('n_classes')} classes, "
        f"trained on {model.info.get('n_isolates')} isolates")
    pairs = find_plate_images(image_dir)
    log(f"{len(pairs)} plates (image pairs) in {image_dir}")

    if sample_sheet:
        sheet = pd.read_csv(sample_sheet, dtype={"barcode": str})
        # CAMII's sample_metadata.csv calls the medium column "plateTyte"
        mcol = next((c for c in ("media", "medium", "plateType", "plateTyte") if c in sheet), None)
        batches = [(str(g), sub["barcode"].tolist(), int(sub["numGroupPick"].iat[0]),
                    str(sub[mcol].iat[0]) if mcol else media) for g, sub in sheet.groupby("groupID", sort=True)]
        batches = [b for b in batches if b[2] > 0]
    else:
        barcodes = sorted(pairs)
        size = batch_size or len(barcodes)
        batches = [(f"batch{k // size + 1:03d}" if batch_size else "all", barcodes[k:k + size], n, media)
                   for k in range(0, len(barcodes), size)]
        if not batch_size and len(barcodes) > 20:
            log(f"WARNING: {len(barcodes)} plates form ONE picking group of {n} colonies. For many plates use "
                f"--sample-sheet (see 'colonypick sheet') or --batch-size.")
    known = set(model.builder.media)
    for m in sorted({b[3] for b in batches}):
        if known and m.strip().lower() not in known:
            log(f"WARNING: medium '{m}' was not in the training data; predictions ignore the medium")
    log(f"{len(batches)} batches, {sum(len(b[1]) for b in batches)} plates in total\n")

    group_files = []
    for bi, (gname, barcodes, n_g, media_g) in enumerate(batches, 1):
        log(f"[batch {bi}/{len(batches)}] {gname}: {len(barcodes)} plates, {n_g} picks, medium {media_g}")
        results = {}
        for b in barcodes:
            if b not in pairs:
                log(f"    {b}: no image pair - skipped")
                continue
            try:
                res, cached = _detect_cached(b, pairs[b], cfg, cache_dir)
            except Exception as e:  # noqa: BLE001 - one bad plate should not stop the run
                log(f"    {b}: FAILED ({type(e).__name__}: {e})")
                continue
            results[b] = res
            log(f"    {b}: {len(res.colonies)} colonies{' (cached)' if cached else ''}")
        cols, picks, camii = pick_group(results, model, n_g, media_g, exclude_taxa, compare_camii,
                                        tail=tail, switch_gain=switch_gain)
        if cols.empty:
            log("    no colonies - batch skipped")
            continue
        for d in (cols, picks) + ((camii,) if camii is not None else ()):
            d.insert(0, "group", gname)
        g = _safe(gname)
        cols.to_csv(out / f"{g}_colonies.csv", index=False)
        picks.to_csv(out / f"{g}_picks.csv", index=False)
        if camii is not None:
            camii.to_csv(out / f"{g}_camii_picks.csv", index=False)
        group_files.append(g)
        for b, r in results.items():
            pb = picks[picks["plate_barcode"] == b]
            pb[["pick_rank", "plate_barcode", "colony_index", "X_rawImage", "Y_rawImage", "X", "Y",
                "predicted_taxon", "predicted_prob", "expected_new_taxa"]].to_csv(plate_dir / f"{b}_picks.csv",
                                                                                   index=False)
            draw_colonies(r, pb, plate_dir / f"{b}_picks.jpg")
        k = cols.attrs.get("phase1_picks", len(picks))
        tail_msg = f" ({k} by novelty, {len(picks) - k} by {tail} tail)" if k < len(picks) else ""
        log(f"    {len(cols)} colonies -> picked {len(picks)}{tail_msg}; expected distinct taxa "
            f"{picks['expected_taxa_cumulative'].iat[-1]:.1f}\n")
        del results, cols

    if not group_files:
        log("nothing picked")
        return pd.DataFrame()
    picks = pd.concat([pd.read_csv(out / f"{g}_picks.csv") for g in group_files], ignore_index=True)
    picks.to_csv(out / "picks.csv", index=False)
    if compare_camii:
        pd.concat([pd.read_csv(out / f"{g}_camii_picks.csv") for g in group_files
                   if (out / f"{g}_camii_picks.csv").exists()]).to_csv(out / "camii_picks.csv", index=False)
    log(f"done: {len(picks)} picks in {len(group_files)} batches -> {out / 'picks.csv'}\n"
        f"per-group files: <group>_picks.csv, <group>_colonies.csv; per-plate files and images: {plate_dir}")
    return picks

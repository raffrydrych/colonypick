"""Command-line interface: ``colonypick <command> ...`` (or ``python -m colonypick``)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cfg(args):
    from .detection import DetectionConfig
    cfg = DetectionConfig.from_camii_configure(args.config) if args.config else DetectionConfig()
    if args.calibration:
        cfg.calib_parameter_path = args.calibration
    if not cfg.calib_parameter_path or not Path(cfg.calib_parameter_path).exists():
        print("WARNING: no calibration file - images are not flat-field corrected "
              "(features then differ from CAMII's; see README, 'Calibration')")
    return cfg


def _load_training(args):
    from .data import load_camii_isolates, load_colony_table
    from .features import plate_context
    df = load_camii_isolates(args.metadata, args.morphology, args.s16, min_purity=args.min_purity)
    ctx = None
    if args.colonies:
        col = load_colony_table(args.colonies)
        ctx = plate_context(df, col)
        print(f"plate context: {col['plate_barcode'].nunique()} plates, "
              f"{ctx['n_neighbors'].notna().mean():.0%} of isolates matched")
    return df, ctx


def cmd_train(args):
    from .model import ColonyModel
    df, ctx = _load_training(args)
    model = ColonyModel(level=args.level, n_trees=args.trees, min_samples_leaf=args.min_leaf).fit(df, ctx)
    model.save(args.out)
    print(f"saved model to {args.out}")


def cmd_evaluate(args):
    from .evaluate import evaluate
    df, ctx = _load_training(args)
    evaluate(df, ctx, args.out, level=args.level, n_pools=args.pools)


def cmd_benchmark(args):
    from .benchmark import run_benchmark
    df, ctx = _load_training(args)
    run_benchmark(df, ctx, args.out, n_pools_small=args.pools_small, n_pools_large=args.pools_large,
                  quick=args.quick)


def cmd_evaluate_tail(args):
    from .evaluate import evaluate_tail
    df, ctx = _load_training(args)
    evaluate_tail(df, ctx, args.out, level=args.level, n_pools=args.pools, n_max=args.n_max,
                  switch_gains=[float(x) for x in args.switch_gains.split(",")])


def cmd_detect(args):
    from .detection import detect_folder, draw_colonies
    import pandas as pd
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = detect_folder(args.images, _cfg(args))
    for b, r in res.items():
        r.colonies.to_csv(out / f"{b}_colonies.csv", index=False)
        draw_colonies(r, None, out / f"{b}_colonies.jpg")
    if res:
        pd.concat([r.colonies for r in res.values()]).to_csv(out / "all_colonies.csv", index=False)
    print(f"written to {out}")


def cmd_pick(args):
    from .pipeline import run_pick
    exclude = None
    if args.exclude_taxa:
        exclude = [s.strip() for s in Path(args.exclude_taxa).read_text().splitlines() if s.strip()]
    run_pick(args.images, args.model, args.out, n=args.n, media=args.media, cfg=_cfg(args),
             sample_sheet=args.sample_sheet, batch_size=args.batch_size, exclude_taxa=exclude,
             compare_camii=args.compare_camii, tail=args.tail, switch_gain=args.switch_gain)


def cmd_sheet(args):
    from .pipeline import make_sample_sheet
    make_sample_sheet(args.images, args.metadata, args.out, n=args.n)


def cmd_calibrate(args):
    from .detection import DetectionConfig, find_plate_images, make_calibration
    cfg = DetectionConfig.from_camii_configure(args.config) if args.config else DetectionConfig()
    pairs = find_plate_images(args.images)
    if not pairs:
        sys.exit(f"no <barcode>_<timestamp>.bmp image pairs in {args.images}")
    make_calibration([t for t, _ in pairs.values()], [e for _, e in pairs.values()], cfg, args.out)
    print(f"calibration from {len(pairs)} blank plates written to {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="colonypick",
                                description="Morphology-guided colony picking that maximises the expected "
                                            "number of distinct taxa (built on CAMII).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def training_args(sp):
        sp.add_argument("--metadata", required=True, help="CAMII_isolates_metadata.tsv")
        sp.add_argument("--morphology", required=True, help="CAMII_isolates_morphological_features.tsv")
        sp.add_argument("--s16", required=True, help="CAMII_isolates_16SV4_result.tsv")
        sp.add_argument("--colonies", help="folder colony_all.metadata.<date> (optional, enables plate-context correction)")
        sp.add_argument("--level", choices=["asv", "genus"], default="asv", help="taxonomic level (default asv)")
        sp.add_argument("--min-purity", type=float, default=50.0, help="minimum 16S purity in %% (default 50)")

    def detection_args(sp):
        sp.add_argument("--images", required=True, help="folder with <barcode>_<timestamp>.bmp image pairs")
        sp.add_argument("--config", help="CAMII-style configure file (default: published CAMII parameters)")
        sp.add_argument("--calibration", help="calibration .npz (flat-field images of blank plates)")

    sp = sub.add_parser("train", help="train the colony classifier on the public CAMII tables")
    training_args(sp)
    sp.add_argument("--trees", type=int, default=200)
    sp.add_argument("--min-leaf", type=int, default=10, help="min samples per leaf; smaller = larger model (default 10)")
    sp.add_argument("--out", default="colonypick_model.joblib")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="offline comparison with CAMII, random and max-min picking")
    training_args(sp)
    sp.add_argument("--pools", type=int, default=20, help="candidate pools per group (default 20)")
    sp.add_argument("--out", default="evaluation")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("benchmark", help="full comparison of every approach with all metrics -> report.html")
    training_args(sp)
    sp.add_argument("--pools-small", type=int, default=20, help="pools of 200 colonies per group (default 20)")
    sp.add_argument("--pools-large", type=int, default=10, help="whole-group subsamples per group (default 10)")
    sp.add_argument("--quick", action="store_true", help="few pools, for a fast check that everything runs")
    sp.add_argument("--out", default="benchmark")
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("evaluate-tail", help="benchmark for large pick numbers: novelty vs novelty + "
                                              "morphology/anomaly tail (whole groups, up to --n-max picks)")
    training_args(sp)
    sp.add_argument("--pools", type=int, default=10, help="subsampled pools (90%%) per group (default 10)")
    sp.add_argument("--n-max", type=int, default=500, help="largest number of picks evaluated (default 500)")
    sp.add_argument("--switch-gains", default="0.3,0.1",
                    help="comma-separated thresholds: switch to the tail when a pick adds fewer expected new "
                         "taxa than this (default 0.3,0.1)")
    sp.add_argument("--out", default="evaluation_tail")
    sp.set_defaults(func=cmd_evaluate_tail)

    sp = sub.add_parser("detect", help="detect colonies and compute features (no picking)")
    detection_args(sp)
    sp.add_argument("--out", default="detection")
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("pick", help="detect colonies and choose the most diverse ones")
    detection_args(sp)
    sp.add_argument("--model", required=True, help="model file from 'colonypick train'")
    sp.add_argument("--n", type=int, default=50, help="colonies to pick (default 50)")
    sp.add_argument("--media", default="mGAM", help="growth medium of the plates (default mGAM)")
    sp.add_argument("--sample-sheet", help="CAMII sample_metadata.csv-style CSV: barcode, groupID, numGroupPick[, plateType or media]")
    sp.add_argument("--batch-size", type=int, help="without a sample sheet: pick --n colonies from each block "
                                                   "of this many plates (sorted by barcode)")
    sp.add_argument("--exclude-taxa", help="text file of taxa already isolated, one per line: model class (ASV8), genus name or V4 sequence")
    sp.add_argument("--compare-camii", action="store_true", help="also write the original CAMII picks")
    sp.add_argument("--tail", choices=["none", "maxmin", "anomaly", "anomaly_entropy"], default="none",
                    help="for large pick numbers: once expected novelty adds < --switch-gain taxa per pick, fill "
                         "the rest by morphology (maxmin) or Isolation Forest anomaly (default none)")
    sp.add_argument("--switch-gain", type=float, default=0.1)
    sp.add_argument("--out", default="picks")
    sp.set_defaults(func=cmd_pick)

    sp = sub.add_parser("sheet", help="write a sample sheet grouping plates by sample and medium (CAMII metadata)")
    sp.add_argument("--images", required=True, help="folder with <barcode>_<timestamp>.bmp image pairs")
    sp.add_argument("--metadata", required=True, help="CAMII_isolates_metadata.tsv (sourcePlate, sourceSample, media)")
    sp.add_argument("--n", type=int, help="colonies per group (default: as many as CAMII picked in that group)")
    sp.add_argument("--out", default="sample_sheet.csv")
    sp.set_defaults(func=cmd_sheet)

    sp = sub.add_parser("calibrate", help="build a flat-field calibration from images of blank plates")
    sp.add_argument("--images", required=True)
    sp.add_argument("--config")
    sp.add_argument("--out", default="calib_parameter.npz")
    sp.set_defaults(func=cmd_calibrate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

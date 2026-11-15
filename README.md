# colonypick

Picks the most taxonomically diverse bacterial colonies from agar plate images. It builds on
[CAMII](https://github.com/hym0405/CAMII) (Huang et al., *Nat Biotechnol* 2023): colony detection and
the 14 morphological features are a faithful port of CAMII, and the picking strategy is new. A
random forest trained on ~22,000 sequenced CAMII isolates gives, for every colony, the probability
of each taxon. Colonies are then picked greedily to maximise the **expected number of distinct
taxa**, instead of CAMII's maximal spread in a 2-D PCA of the features.

In an offline benchmark on the public CAMII data (50 picks from pools of 200 colonies), this finds
**+20% distinct ASVs, +21% genera and +52% rare ASVs** compared with CAMII. The full evaluation, and
the run that produced these numbers, are described in [TESTS.md](TESTS.md).

## Directory structure

```
.
├── colonypick/              # source code (Python package)
│   ├── detection.py         # CAMII colony detection + 14 features (port)
│   ├── data.py              # loading and labelling the CAMII isolate tables (ASV, genus, donor)
│   ├── features.py          # classifier features, plate-crowding correction
│   ├── model.py             # random forest taxon classifier (train / predict / save)
│   ├── picking.py           # expected-novelty selection, CAMII max-min, experimental tails
│   ├── pipeline.py          # images → picks, in batches with a detection cache
│   ├── evaluate.py          # narrower offline benchmarks (evaluate, evaluate-tail)
│   ├── benchmark.py         # full benchmark of every approach → report.html
│   └── cli.py               # command line (colonypick <command>)
├── configs/
│   └── camii_detection.cfg  # CAMII detection parameters (default)
├── docs/                    # figures used in TESTS.md
├── examples/
│   └── sample_metadata.csv  # example sample sheet (picking groups)
├── experiments/
│   └── camii_picking_eval.py  # full experiment script used during development
├── tests/
│   └── test_smoke.py        # pytest smoke tests on synthetic data
├── TESTS.md                 # tests, evaluation protocol and results of the reference run
├── environment.yml          # conda-forge environment
├── pyproject.toml           # package metadata, `colonypick` command
└── LICENSE
```

Data, models and outputs (`*.bmp`, `*.npz`, `*.joblib`, `picks/`, `benchmark/`, …) are gitignored.

## Setup

With conda, conda-forge only (mixing channels can cause an OpenMP conflict):

```bash
git clone https://github.com/<you>/colonypick.git
cd colonypick
conda env create -f environment.yml
conda activate colonypick
pip install -e . --no-deps
colonypick --help
```

Into an existing environment instead: `conda install -c conda-forge numpy pandas scipy scikit-learn
scikit-image py-opencv joblib matplotlib`, then `pip install -e . --no-deps`. Without installing,
run `python -m colonypick ...` from the repository folder.

Libraries: numpy, pandas, scipy, scikit-learn, scikit-image, OpenCV, joblib, matplotlib (pytest for
the tests). PyTorch is needed only for the image-embedding experiments in `experiments/`.

## Usage

**1. Train the classifier (once).** Uses the public CAMII tables, which are not included here
(they are released with Huang et al. 2023). `--colonies`, the folder of all detected colonies (one
CSV per plate), is optional and enables the crowding correction:

```bash
colonypick train --metadata CAMII_isolates_metadata.tsv \
    --morphology CAMII_isolates_morphological_features.tsv \
    --s16 CAMII_isolates_16SV4_result.tsv \
    --colonies colony_all.metadata.20220714 \
    --out models/model_asv.joblib
```

This takes a few minutes and writes `model_asv.joblib` (~70 MB on disk, ~0.5 GB in memory) and
`model_asv.json` (training summary; for every ASV class its genus and V4 sequence).

**2. Pick colonies from images.** Image pairs `<barcode>_<timestamp>.bmp` in one folder; per plate
the earlier image is trans-illuminated and the later one epi-illuminated.

```bash
# a few plates of one sample: one picking group
colonypick pick --images plates/ --calibration calib_parameter.npz \
    --model models/model_asv.joblib --n 96 --media mGAM --out picks/

# many plates: group them by sample × medium, then pick group by group
colonypick sheet --images plates/ --metadata CAMII_isolates_metadata.tsv --out sample_sheet.csv
colonypick pick  --images plates/ --calibration calib_parameter.npz \
    --model models/model_asv.joblib --sample-sheet sample_sheet.csv --out picks/
```

Outputs in `picks/`:

| file | content |
|---|---|
| `picks.csv` | all picks: group, `pick_rank`, plate, colony, `X_rawImage`/`Y_rawImage` (raw image pixels), 14 features, `predicted_taxon`, `expected_new_taxa` |
| `<group>_picks.csv`, `<group>_colonies.csv` | picks and all detected colonies of one group |
| `plates/<barcode>_picks.jpg` | trans image: colonies outlined in grey, picks in green with their rank |
| `camii_picks.csv` | original CAMII picks (with `--compare-camii`) |
| `cache/` | detection results; an interrupted run resumes from here (safe to delete) |

`pick_rank` is the order of choice: if fewer colonies can be picked, take the top ranks.
`predicted_taxon` is not an identification (see [TESTS.md](TESTS.md), *Classifiers*). Converting
pixels to robot coordinates is specific to the picker and not included.

**Other commands:**

```bash
colonypick detect    --images plates/ --calibration calib_parameter.npz --out detection/  # colonies + features only
colonypick calibrate --images blank_plates/ --out calib_parameter.npz                      # flat-field calibration
colonypick benchmark --metadata ... --morphology ... --s16 ... --colonies ... --out benchmark/  # full evaluation
python -m pytest tests                                                                      # tests
```

## Configs & options

* **Detection parameters:** `--config` takes a CAMII `configure` file. Without it the published CAMII
  values are used (also in `configs/camii_detection.cfg`).
* **Calibration:** `--calibration calib_parameter.npz` corrects uneven illumination. Build it once
  per imaging setup with `colonypick calibrate` from images of blank plates. Without it, the colour
  and brightness features are on a different scale than the training data.
* **Sample sheet** (CAMII `sample_metadata.csv` format, see `examples/sample_metadata.csv`): `barcode`,
  `groupID` (plates picked jointly), `numGroupPick` (colonies per group, 0 = skip), `plateType` or
  `media`. `colonypick sheet` sets `numGroupPick` to the number CAMII picked from that group;
  `--n 96` sets one number for all groups. Without a sheet, `--batch-size 5 --n 30` picks 30 colonies
  from each block of 5 plates.
* **Model:** `train --level asv|genus` (default `asv`, which is better even for genus diversity),
  `--trees` (200) and `--min-leaf` (10; smaller leaves make a much larger model for little gain).
* **Picking:**
  * `--exclude-taxa known.txt`: taxa already in a collection (`ASV8`, `ASV8 (Bacteroides)`, a genus
    name or a V4 sequence), one per line.
  * `--compare-camii`: also writes CAMII's picks.
  * `--tail {none,maxmin,anomaly,anomaly_entropy}` with `--switch-gain`: experimental fill for large
    pick numbers. The default `none` is best (see TESTS.md).

## Swapping / adding picking strategies

Strategies live in `colonypick/picking.py`. Each takes the classifier probabilities `P` (colonies ×
taxa) and/or the feature matrix and returns an ordered list of colony indices:

* `expected_novelty_order(P, classes, n)`: the default;
* `camii_maxmin_select(features, n)`: the CAMII baseline;
* `hybrid_order(P, classes, X, n, tail=...)`: novelty plus a morphology tail.

To compare a new strategy with all the others, add its order to the `nested` dictionary in
`simulate_picking()` in `colonypick/benchmark.py` (or a per-`n` selection next to CAMII and max-min
there). `colonypick benchmark` then scores it on the same pools and adds it to every table and figure
of `report.html`. To use it for real picks, call it in `pick_group()` in `colonypick/pipeline.py`.

## Extending to your own data

* **Own sequenced isolates:** write tables in the CAMII format (isolate metadata, morphological
  features, 16S V4 result) and pass them to `colonypick train`, alone or concatenated with the CAMII
  tables. This is the main way to adapt the model to another imaging setup, other media or other
  sample types.
* **Own imaging setup:** build a calibration from blank plates (`colonypick calibrate`) and, if
  needed, adjust the detection parameters in a `configure` file.
* **Evaluation:** pick part of the colonies at random in each round. Only random picks are an
  unbiased sample, and `colonypick benchmark` uses them as the test set (see TESTS.md, *Protocol*).

## Citation and license

Please cite CAMII, which provides the detection method and the training data: Huang Y, Sheth RU,
Zhao S, et al. High-throughput microbial culturomics using automation and machine learning.
*Nature Biotechnology* 41, 1424–1433 (2023). https://doi.org/10.1038/s41587-023-01674-2

MIT, see `LICENSE`. The colony detection is ported from CAMII (MIT, © 2021 Yiming Huang).

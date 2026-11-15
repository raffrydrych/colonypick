# Tests and evaluation

This file describes how colonypick was tested, in three parts:

1. the automated tests;
2. the reference run on the full public CAMII data set: environment, commands, logs;
3. the results of the full benchmark, which compares every picking approach.

Installation and usage are in [README.md](README.md).

## 1. Automated tests

`tests/test_smoke.py` runs on synthetic data (well-separated fake taxa), so it needs no CAMII
files. Run it with:

```bash
python -m pytest tests
```

| test | checks |
|---|---|
| `test_expected_novelty_prefers_new_taxa` | the greedy picker takes one colony per taxon before a second of the same taxon; gains do not increase |
| `test_exclude_taxa` | taxa passed with `--exclude-taxa` are not picked first |
| `test_camii_maxmin_returns_unique_indices` | the CAMII max-min returns n distinct colonies |
| `test_plate_context_counts_neighbours` | neighbour counts within 50 px and colonies per plate for the crowding correction |
| `test_model_fit_predict_save` (ASV and genus) | train, predict (rows sum to 1), save/load round trip, out-of-fold prediction |
| `test_resolve_taxa` | a taxon given as `ASV1 (Genus0)` or as a genus name maps to the right classes |
| `test_hybrid_tail_fills_to_n` | every tail mode returns n distinct colonies and switches when the gain drops |
| `test_classifier_metrics_and_facility_location` | the benchmark's metric function and the facility-location baseline |

Checked against real data during development:

* **Detection port:** the same colonies as the original CAMII code on its three example plates
  (84, 102 and 102 colonies), with feature differences ≈ 10⁻¹⁴.
* **Faster max-min:** the rewritten CAMII max-min returns exactly the same picks as the original
  algorithm on random test sets, and runs a group of 5,400 colonies with 1,280 picks in ~11 s
  instead of hours.
* **Reproducibility across machines:** `colonypick evaluate` gave identical results on the Windows
  laptop and on Linux (16.29 distinct ASVs at 50 picks without the crowding correction).

## 2. Reference run

### Environment

* **Machine:** a Windows laptop with VS Code, running on the CPU (the GPU is not needed).
* **Python:** a conda environment with all packages from conda-forge (numpy, pandas, scipy,
  scikit-learn, scikit-image, py-opencv, joblib, matplotlib). Mixing the `defaults` and
  `conda-forge` channels had earlier caused an OpenMP conflict ("OMP Error #15").
* **How it was run:** without installing the package, as `python -m colonypick ...` from the
  repository folder in a Command Prompt terminal. In PowerShell, `conda activate` did not take
  effect.

### Data

Public data released with Huang et al. 2023, in one folder:

| file / folder | content |
|---|---|
| `CAMII_isolates_metadata.tsv` | 27,991 isolates: plate, sample, medium, picking strategy |
| `CAMII_isolates_morphological_features.tsv` | the 14 features of each isolate's colony |
| `CAMII_isolates_16SV4_result.tsv` | 16S V4 sequence, taxonomy and purity of each isolate |
| `colony_all.metadata.20220714/` | 706 CSVs, every detected colony per plate |
| `00_raw_image_plate/` | 744 plate image pairs (trans + epi BMP) |
| `calib_parameter.npz` | flat-field calibration from the CAMII repository |

### Step 1: training

```bash
python -m colonypick train --metadata CAMII_isolates_metadata.tsv \
    --morphology CAMII_isolates_morphological_features.tsv --s16 CAMII_isolates_16SV4_result.tsv \
    --colonies colony_all.metadata.20220714 --out colonypick_out/model_asv.joblib
```

```
purity >= 50.0%: kept 22514 of 27991 isolates
ASV: 1097 unique V4 sequences -> 832 after denoising
931 isolates with missing feature values (filled with medians)
isolates: 21937 | donors: 25 | ASVs: 821 | genera: 248 | random-picked: 3037
plate context: 706 plates, 82% of isolates matched
  context correction Area: neighbours -0.518, plate density +0.041, R2 0.065
  context correction Perimeter: neighbours -0.246, plate density +0.019, R2 0.057
  context correction Radius: neighbours -0.235, plate density +0.015, R2 0.067
trained random forest: 21937 isolates, 135 asv classes + other
```

Sequences one mismatch away from one at least twice as abundant were merged as sequencing errors
(1,097 → 832). 82% of isolates were matched to their plate in the colony table, enough to fit the
crowding correction: colonies with more neighbours are smaller.

### Step 2: picking on all plate images

```bash
python -m colonypick sheet --images 00_raw_image_plate --metadata CAMII_isolates_metadata.tsv \
    --out colonypick_out/sample_sheet.csv
python -m colonypick pick --images 00_raw_image_plate --calibration calib_parameter.npz \
    --model colonypick_out/model_asv.joblib --sample-sheet colonypick_out/sample_sheet.csv \
    --compare-camii --out colonypick_out/picks_all
```

```
744 plates -> 125 groups; 230 plates without metadata (groupID 'unassigned', skipped)
...
done: 25415 picks in 125 batches
```

* **Plates and groups:** 514 plates in 125 sample × medium groups were processed, giving 25,415
  picks. The number of picks per group is the number CAMII actually picked from that group, between
  32 and 1,280.
* **First attempt stalled at batch 76** (H1t4, 5,422 colonies, 1,280 picks). The `--compare-camii`
  run of the original max-min was far too slow at that size. After the max-min rewrite the run was
  restarted; the first 76 groups came from the detection cache, and the run completed.
* **Large groups:** in H1t4, H5t1 and H6t1 (1,280 picks each) the model's expected number of distinct
  taxa levels off at ~128, because the classifier knows 135 taxa. Section 3.3 tests whether a
  different fill helps there; it does not.
* **A data quirk:** one plate (TW262M4) has its medium written as `MGAM` and forms its own group.
  This can be fixed in the sample sheet.

### Step 3: evaluation

The full benchmark (section 3) on the same tables:

```bash
python -m colonypick benchmark --metadata CAMII_isolates_metadata.tsv \
    --morphology CAMII_isolates_morphological_features.tsv --s16 CAMII_isolates_16SV4_result.tsv \
    --colonies colony_all.metadata.20220714 --out colonypick_out/benchmark
```

It took 17.8 minutes: three out-of-fold classifiers, then pools of 200 (~50 s per group), whole
groups (90–130 s per group) and learning curves. It writes `report.html`, CSV tables and the figures
in `docs/`.

Two narrower runs before it agree with the benchmark:

* **`colonypick evaluate`** (pools of 200, distinct ASVs at 50 picks): CAMII 13.64, expected
  novelty 16.29 without the crowding correction and 16.65 with it (H6t1: 10.20 → 11.55).
* **`colonypick evaluate-tail`** (whole groups, up to 500 picks): no tail beat plain expected novelty.

## 3. Results of the full benchmark

### 3.1 Protocol

Only randomly picked isolates are an unbiased sample of what grows on a plate. The isolates CAMII
picked by morphology were already selected. Strategies are therefore compared inside candidate
pools drawn from the random picks: four sample × medium groups (H1t1, H1t4, H5t1, H6t1, all on mGAM;
three donors; 599–865 random picks each). Two scales are simulated:

* **pools of 200:** 20 random pools of 200 colonies per group, up to 100 picks;
* **whole groups:** 10 random 90% subsamples of each group (539–778 colonies), up to 500 picks.

Every strategy picks from the same pools. We count distinct 16S V4 ASVs (purity ≥ 50%), distinct
genera and distinct **rare** ASVs (≤ 2 isolates in the group).

Classifier probabilities are **out-of-fold with folds grouped by donor**: every pool is scored by a
model that never saw that donor. Intervals are 95% bootstrap intervals over pools, stratified by
group. With only four groups they understate between-sample uncertainty, so the number of groups won
is reported as well.

### 3.2 Picking: expected novelty is the winner

The score is the area under the picking curve divided by the oracle's, for distinct ASVs. The oracle
knows every taxon; 1.0 means a new taxon with every pick for as long as possible.

| strategy | pools of 200 | Δ vs CAMII [95% CI] | groups > CAMII | whole groups | Δ vs CAMII [95% CI] |
|---|---|---|---|---|---|
| **expected novelty (ASV RF, crowding correction)** | **0.691** | **+0.094 [+0.080, +0.107]** | **4/4** | **0.761** | **+0.083 [+0.075, +0.092]** |
| expected novelty (ASV RF, no crowding correction) | 0.679 | +0.081 [+0.068, +0.093] | 3/4 | 0.748 | +0.071 [+0.062, +0.079] |
| expected novelty (genus RF) | 0.663 | +0.066 [+0.053, +0.078] | 3/4 | 0.738 | +0.061 [+0.051, +0.070] |
| anomaly only (Isolation Forest × distance) | 0.635 | +0.037 [+0.025, +0.049] | 3/4 | 0.718 | +0.040 [+0.031, +0.049] |
| max-min on the 14 features | 0.619 | +0.021 [+0.008, +0.033] | 3/4 | 0.699 | +0.022 [+0.013, +0.030] |
| **CAMII (PCA2 + max-min)** | 0.597 | – | – | 0.678 | – |
| facility location on the 14 features | 0.573 | −0.024 [−0.038, −0.011] | 2/4 | 0.669 | −0.008 [−0.017, +0.001] |
| random | 0.551 | −0.046 [−0.058, −0.036] | 2/4 | 0.644 | −0.034 [−0.041, −0.026] |

Mean distinct taxa at selected pick numbers (pools of 200; ASVs / genera / rare ASVs):

| strategy | n = 20 | n = 50 | n = 100 |
|---|---|---|---|
| random | 7.8 / 5.3 / 0.9 | 12.5 / 8.0 / 2.2 | 17.5 / 11.0 / 4.3 |
| CAMII | 9.1 / 6.2 / 1.3 | 13.9 / 9.1 / 2.8 | 18.3 / 11.7 / 4.8 |
| **expected novelty** | **10.6 / 7.8 / 2.1** | **16.7 / 11.0 / 4.2** | **20.7 / 13.2 / 6.3** |
| oracle | 19.7 / 15.1 / 8.3 | 24.6 / 15.2 / 8.3 | 24.6 / 15.2 / 8.3 |

![Pools of 200, all approaches](docs/benchmark_pool200_all.png)

* **Expected novelty beats CAMII in all six comparisons** (3 metrics × 2 scales), with intervals
  excluding zero. It wins in 4/4 groups for pools of 200 and 3/4 for whole groups. At n = 50 it finds
  +20% ASVs, +21% genera and **+52% rare ASVs**.
* **The ASV-level classifier is best, even for genus diversity** (genus AUC 0.718 vs 0.701 for the
  genus classifier).
* **The crowding correction helps a little and consistently:** +0.012 AUC for ASVs, +0.023 for rare
  ASVs at whole-group scale, and 4/4 vs 3/4 groups won.
* **Facility location is worse than CAMII.** It maximises coverage, so it picks *typical* colonies.
* **Per group** (ASVs at n = 50, pools of 200), expected novelty vs CAMII:

  | group | expected novelty | CAMII |
  |---|---|---|
  | H1t1 | 18.5 | 14.6 |
  | H1t4 | 21.4 | 16.7 |
  | H5t1 | 14.9 | 13.2 |
  | H6t1 | 12.1 | 11.0 |

  At n = 300 (whole groups), H6t1 is the only group where CAMII is ahead: 26.5 vs 28.6.

![Whole groups, all approaches](docs/benchmark_wholegroup_all.png)

### 3.3 Large pick numbers: tails and anomaly detection do not help

With hundreds of picks the expected gain per pick falls towards zero, and the rare taxa (merged into
"other") remain to be found. Two ideas were tested:

* **switching to a morphology tail** once a pick adds < 0.3 or < 0.1 expected new taxa: max-min,
  Isolation Forest anomaly × distance, or anomaly × classifier entropy;
* **anomaly detection alone.**

Results:

* **All six novelty + tail variants are statistically tied with plain expected novelty**
  (|Δ AUC| ≤ 0.003, intervals include zero). Switching early (0.3) costs up to 1.5 taxa between 150
  and 400 picks.
* **Plain novelty keeps finding rare taxa:** at 500 picks it still finds 2.6 more rare ASVs than CAMII.
* **Anomaly detectors as rare-taxon detectors** (target: ASVs with ≤ 2 isolates in the group; base
  rate 4.5%):

  | detector | ROC-AUC | PR-AUC | precision@10% | recall@10% | F1@10% |
  |---|---|---|---|---|---|
  | Isolation Forest | 0.641 | 0.104 | 0.110 | 0.237 | 0.149 |
  | classifier entropy | 0.627 | 0.078 | 0.078 | 0.163 | 0.105 |
  | LOF (k = 20) | 0.599 | 0.072 | 0.077 | 0.159 | 0.103 |
  | kNN distance (k = 10) | 0.594 | 0.084 | 0.074 | 0.153 | 0.099 |
  | classifier p(other) | 0.585 | 0.108 | 0.103 | 0.223 | 0.140 |

  Rare taxa look somewhat unusual: Isolation Forest roughly doubles their share among flagged
  colonies, which is why "anomaly only" beats CAMII. The classifier already captures this, so as a
  tail it adds nothing. In H6t1 every detector is at chance (ROC-AUC 0.45–0.56).

![Whole groups, novelty + tails](docs/benchmark_wholegroup_tails.png)

### 3.4 Classifiers

Out-of-fold on held-out donors. Classes: 135 ASVs (or 81 genera) with ≥ 10 isolates, plus "other".

| classifier | top-1 acc. | top-5 acc. | balanced acc. | F1 macro | F1 weighted | log-loss | Brier | ROC-AUC macro / weighted | PR-AUC macro | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| ASV RF, crowding corr. | 0.065 | 0.227 | 0.031 | 0.023 | 0.070 | 6.40 | 0.992 | 0.596 / 0.681 | 0.022 | 0.086 |
| ASV RF, no correction | 0.064 | 0.228 | 0.031 | 0.023 | 0.069 | 6.35 | 0.992 | 0.584 / 0.679 | 0.022 | 0.086 |
| genus RF | 0.187 | 0.441 | 0.050 | 0.039 | 0.238 | 4.33 | 0.920 | 0.615 / 0.691 | 0.033 | 0.024 |

As **taxon identifiers** the classifiers are weak on new donors:

* **Accuracy:** top-1 accuracy is below the majority-class rate (0.115 for ASVs, 0.290 for genera),
  because training uses balanced class weights.
* **Calibration:** the ASV log-loss (6.40) is worse than a uniform guess over 136 classes
  (ln 136 = 4.91), so the probabilities are overconfident. Recalibration did not improve picking.

Their **ranking** is informative (weighted ROC-AUC 0.68), and that is all expected novelty needs:
which colonies are probably the *same* taxon as ones already picked.

**Comparison with the paper.** Huang et al. report ~70% precision for genera with more than 100
isolates, using a random 70/30 split of isolates. That split puts the same donors in training and
test. We repeated that test on the same tables (24 such genera, 5 repeats; without the paper's
neighbour-count feature):

| split | accuracy | median precision | genera with precision ≥ 0.6 | median recall |
|---|---|---|---|---|
| random 70/30, as in the paper | 0.59 | 0.59 | 10 / 24 | 0.35 |
| 70/30 by donor (test donors unseen) | 0.36 | 0.11 | 0 / 24 | 0.06 |

The paper's setting is reproduced approximately (~0.6 vs ~0.7). On unseen donors, precision drops to
~0.1, and accuracy is barely above always predicting *Bacteroides* (0.32): the model largely learns
how a taxon looks *in a given donor*. The paper notes the same, that morphology of one ASV is
consistent within a person but varies between people. Our 6.5% is lower still because it is at ASV
level, over all isolates, on unseen donors, with balanced class weights.

### 3.5 Representations: are two colonies the same taxon?

ROC-AUC over random colony pairs within a group (0.5 = no information):

| representation | same ASV | same genus | H1t1 | H1t4 | H5t1 | H6t1 |
|---|---|---|---|---|---|---|
| CAMII: PCA2 of z-scored features | 0.607 | 0.581 | 0.610 | 0.677 | 0.584 | 0.505 |
| 14 features (log, z-scored) | 0.606 | 0.587 | 0.622 | 0.676 | 0.585 | 0.504 |
| 14 features + crowding correction | 0.607 | 0.588 | 0.622 | 0.678 | 0.586 | 0.504 |
| classifier probabilities (ASV RF) | 0.619 | 0.598 | 0.605 | 0.738 | 0.579 | 0.511 |
| classifier probabilities (genus RF) | 0.613 | 0.597 | 0.595 | 0.734 | 0.583 | 0.509 |

Pairwise similarity alone separates taxa only modestly. **In H6t1 no representation carries any
taxonomic signal** (all ≈ 0.50): there, colony morphology does not distinguish taxa, which is why every
method is close to random. The gain of expected novelty comes less from a better distance than from
the coverage objective, which uses the classifier's knowledge of how *common* each look-alike taxon is.

### 3.6 Learning curves

The ASV classifier was trained on 5–100% of each fold's training isolates (877–17,549):

| training isolates | 877 | 1,755 | 4,387 | 8,775 | 17,549 |
|---|---|---|---|---|---|
| top-1 accuracy, training / held-out | 0.42 / 0.054 | 0.42 / 0.056 | 0.45 / 0.058 | 0.47 / 0.060 | 0.50 / 0.065 |
| F1 macro, training / held-out | 0.35 / 0.017 | 0.39 / 0.015 | 0.44 / 0.020 | 0.46 / 0.020 | 0.48 / 0.023 |
| log-loss, training / held-out | 2.58 / 7.10 | 2.59 / 6.89 | 2.45 / 6.54 | 2.31 / 6.74 | 2.18 / 6.40 |
| ROC-AUC macro, held-out | 0.49 | 0.46 | 0.53 | 0.55 | 0.60 |
| picking: distinct ASVs at n = 50 | 16.4 | 16.2 | 16.4 | 17.5 | 17.0 |

![Learning curves](docs/benchmark_learning_curves.png)

* **Large gap between training and held-out donors** (top-1 0.50 vs 0.065): the same taxon looks
  different between samples.
* **Held-out ROC-AUC is still rising** (0.49 → 0.60). More labelled isolates, especially from more
  donors, would still improve the classifier.
* **Picking quality is nearly flat** (16.2–17.5; this sub-benchmark uses 5 pools per group). Even ~900
  training isolates give most of the gain over CAMII (13.9).

### 3.7 Tried, without gain

All of these are in `experiments/camii_picking_eval.py`. None beat the random forest on the 14 CAMII
features within the noise of the benchmark:

* DINOv2 image embeddings of colony crops, with or without per-plate contrast normalisation;
* a CNN trained on colony crops (+0.4 genera at n = 50, within noise). Colonies are only ~20–50 px
  on these images;
* other representations for distance-based picking: all 14 features (raw or log-scaled), facility
  location, classifier-probability similarity;
* changes to the objective: adaptive rounds (re-weighting after "sequencing" earlier picks), rarity
  bonuses, probability calibration, hybrid novelty + distance scores, hierarchical genus → ASV
  objectives, a positive weight for the "other" class;
* for large pick numbers: an Isolation Forest / LOF anomaly or max-min tail after expected novelty.

### 3.8 Limitations

* **Few test samples:** only three donors (four groups) have random picks. In H6t1 colony
  morphology carries no taxonomic signal.
* **Simulation only:** results come from historical isolates, not yet from fresh plates picked by a
  robot.
* **Trained on CAMII:** the model learned CAMII's imaging hardware, media (mostly mGAM) and human gut
  samples. Other setups need calibration and, ideally, retraining on their own sequenced isolates.
* **Predictions are not identifications:** `predicted_taxon` / `predicted_prob` are overconfident on
  new donors; only 16S (or MALDI) identifies an isolate.
* **Rare taxa:** classes with fewer than 10 isolates are merged into "other".

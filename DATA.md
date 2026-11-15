# Downloading the dataset

colonypick is trained and evaluated on the public CAMII data (Huang et al., *Nat Biotechnol* 2023).
The data are **not included in this repository**. Download them from the authors' sources:

| source | what | needed for |
|---|---|---|
| **CAMII data portal**, http://microbial-culturomics.com | isolate tables, colony tables, plate images | training, benchmark, picking on CAMII images |
| **CAMII GitHub repository**, https://github.com/hym0405/CAMII | calibration file, 3 example plates | picking from images (calibration), quick test |
| NCBI BioProject [PRJNA745993](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA745993) | raw sequencing reads | not needed: the processed 16S results are on the portal |

The portal is the "open-access web-based database" named in the paper's data-availability
statement. Its layout may change; if a download below is no longer offered, see the paper's *Data
availability* section or contact the authors.

## 1. Files from the data portal

The portal's download section has one button per file. The table lists each button, the file name
used in the reference run (see [TESTS.md](TESTS.md)) and whether colonypick needs it:

| button on the portal | file / folder | content | needed |
|---|---|---|---|
| **Isolates metadata (TSV)** | `CAMII_isolates_metadata.tsv` | one row per isolate (31,424): source plate, sample, medium, isolation strategy | **yes** |
| **Isolates morphology (TSV)** | `CAMII_isolates_morphological_features.tsv` | the 14 morphological features of each isolate's colony | **yes** |
| **Isolates 16S-V4 rRNA (TSV)** | `CAMII_isolates_16SV4_result.tsv` | 16S V4 sequence, taxonomy at several levels, purity | **yes** |
| **Colony metadata (CSV)** | `colony_all.metadata.20220714/` (706 CSVs, one per plate) | every detected colony: plate, coordinates, 14 features | recommended (crowding correction) |
| **Original images (BMP)** | `00_raw_image_plate/` (744 plate image pairs) | trans- and epi-illuminated plate images | only for picking on CAMII images, image experiments |
| Isolates WGS (TSV) | – | genome sequencing summary | no |
| Plate metadata (TSV) | – | plate barcodes, media, samples | no (taken from the isolate metadata) |
| Processed images (numpy object) | – | light-corrected images | no (colonypick processes the raw BMPs) |
| Colony contours (numpy object) | – | colony outlines | no (colonypick detects them itself) |

* **Minimum:** the three isolate tables are enough for `train` and `benchmark`.
* **Recommended:** add the colony metadata for the crowding correction. It gave the best results
  (TESTS.md, section 3.2).
* **Images:** the original images are large. Download them only to run `pick` or `detect` on the
  CAMII plates.

If a download arrives as an archive (`.zip`, `.tar.gz`), extract it. The colony metadata must end up
as a folder of per-plate CSV files, and the images as a folder of `<barcode>_<timestamp>.bmp` files
(not in subfolders).

## 2. Files from the CAMII GitHub repository

Clone the repository, or use **Code → Download ZIP** on GitHub:

```bash
git clone https://github.com/hym0405/CAMII.git
```

* `CAMII/parameters/calib_parameter.npz`: flat-field calibration for CAMII's imaging station. Pass
  it to `pick` / `detect` with `--calibration` when using CAMII images. For your own imaging setup,
  build your own with `colonypick calibrate` from images of blank plates.
* `CAMII/example/raw_plate_images/`: three example plates (H2M1–H2M3), useful for a first test of
  `pick` without downloading all images.

## 3. Where to put the files

Any folder works; paths are passed on the command line. Inside the repository, use `data/`, which
is in `.gitignore`, so the data are never committed:

```
colonypick/
└── data/
    ├── CAMII_isolates_metadata.tsv
    ├── CAMII_isolates_morphological_features.tsv
    ├── CAMII_isolates_16SV4_result.tsv
    ├── colony_all.metadata.20220714/      # 706 × <barcode>.metadata.csv
    ├── 00_raw_image_plate/                # <barcode>_<timestamp>.bmp, two per plate
    └── calib_parameter.npz                # from the CAMII GitHub repository
```

## 4. Check the download

Count the rows and files. The expected values come from the reference run:

```bash
python -c "import pandas as pd, glob; d='data/'; \
print('isolates   ', len(pd.read_csv(d+'CAMII_isolates_metadata.tsv', sep='\t')), '(expect 31424)'); \
print('morphology ', len(pd.read_csv(d+'CAMII_isolates_morphological_features.tsv', sep='\t')), '(expect 31424)'); \
print('16S rows   ', len(pd.read_csv(d+'CAMII_isolates_16SV4_result.tsv', sep='\t')), '(expect 28328)'); \
print('colony CSVs', len(glob.glob(d+'colony_all.metadata.20220714/*.csv')), '(expect 706)'); \
print('images     ', len(glob.glob(d+'00_raw_image_plate/*.bmp')), '(two per plate; colonypick found 744 plates)')"
```

Then train. The first lines of the log should read:

```bash
colonypick train --metadata data/CAMII_isolates_metadata.tsv \
    --morphology data/CAMII_isolates_morphological_features.tsv \
    --s16 data/CAMII_isolates_16SV4_result.tsv \
    --colonies data/colony_all.metadata.20220714 --out models/model_asv.joblib
```

```
purity >= 50.0%: kept 22514 of 27991 isolates
ASV: 1097 unique V4 sequences -> 832 after denoising
isolates: 21937 | donors: 25 | ASVs: 821 | genera: 248 | random-picked: 3037
plate context: 706 plates, 82% of isolates matched
```

Different counts mean a different version of the data. colonypick still works, but the numbers will
not match TESTS.md exactly.

## 5. Use and citation

The data belong to the authors of CAMII. Do not redistribute them in this repository (`data/` is
gitignored). Cite the paper when you use them:

> Huang Y, Sheth RU, Zhao S, et al. High-throughput microbial culturomics using automation and
> machine learning. *Nature Biotechnology* 41, 1424–1433 (2023). https://doi.org/10.1038/s41587-023-01674-2

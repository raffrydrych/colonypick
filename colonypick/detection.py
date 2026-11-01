"""Colony detection, segmentation and morphological feature extraction.

This module is a Python 3 port of the image-analysis part of CAMII
(Huang et al., Nat Biotechnol 2023; https://github.com/hym0405/CAMII, MIT licence).
The algorithm and its parameters are kept as close to the original as possible,
because the colony classifier is trained on features computed by the original
pipeline: features computed here must mean the same thing.

Pipeline for one plate (trans- and epi-illuminated image pair):
    1. crop to the plate area
    2. flat-field correction with a calibration image (blank plates)
    3. sharpening + adaptive thresholding -> foreground mask
    4. Gaussian blur + Canny edges + dilation/erosion -> external contours
    5. quality filtering (size, circularity, convexity, inertia, distance to neighbours)
    6. random-walker splitting of merged colonies
    7. 14 morphological features per colony

Known quirk kept on purpose: the "Inertia" feature uses the original CAMII
expression ``sqrt((2*mu11)**2) + (mu20 - mu02)**2`` (a misplaced bracket compared with
OpenCV's blob detector). Changing it would make the features incompatible with the
public CAMII data the classifier is trained on.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

FEATURES = ["Area", "Perimeter", "Radius", "Circularity", "Convexity", "Inertia",
            "Graymean", "Graystd", "Repimean", "Repistd", "Gepimean", "Gepistd",
            "Bepimean", "Bepistd"]


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class DetectionConfig:
    """Parameters of the CAMII detection pipeline (defaults = the published CAMII ``configure``)."""
    calib_parameter_path: str | None = None   # .npz with image_trans_calib, image_epi_calib_{B,G,R}
    calib_contrast_trans_alpha: float = 5.0
    calib_contrast_trans_beta: float = -70.0  # the CAMII file lists -100 and then -70; the last one wins
    crop_y_min: int = 150
    crop_y_max: int = 1150
    crop_x_min: int = 150
    crop_x_max: int = 1750
    bg_threshold_block_size: int = 501
    bg_threshold_offset: int = 18
    canny_upper_percentile: float = 80.0
    size_subsample: int = 10000
    min_size: float = 30
    max_size: float = 500000
    min_circularity: float = 0.2
    max_circularity: float = 1.0
    small_size_area: float = 80
    small_size_circularity: float = 0.85
    min_convexity: float = 0.2
    max_convexity: float = 1.0
    min_inertia: float = 0.2
    max_inertia: float = 1.0
    min_dist: float = 3
    min_dist_pin: float = 3
    circularity_threshold: float = 0.85
    circularity_threshold_very_bad: float = 0.5
    area_segment_min: float = 300
    area_segment_max: float = 50000
    segment_bias: int = 10
    filter_bias: int = 20
    post_min_size: float = 100
    post_min_circularity: float = 0.5
    post_min_convexity: float = 0.5
    post_min_inertia: float = 0.5
    random_walker_beta: float = 2000
    random_walker_mode: str = "bf"
    random_walker_maxi_size: int = 17
    seed: int = 0                               # CAMII sampled the Canny threshold without a seed

    # mapping from CAMII ``configure`` keys to field names
    _CAMII_KEYS = {
        "calib_parameter_PATH": "calib_parameter_path",
        "calib_contrast_trans_alpha": "calib_contrast_trans_alpha",
        "calib_contrast_trans_beta": "calib_contrast_trans_beta",
        "cropYMin": "crop_y_min", "cropYMax": "crop_y_max", "cropXMin": "crop_x_min", "cropXMax": "crop_x_max",
        "bg_threshold_blockSize": "bg_threshold_block_size", "bg_threshold_offset": "bg_threshold_offset",
        "canny_upper_percentile": "canny_upper_percentile", "size_subSample": "size_subsample",
        "minSize": "min_size", "maxSize": "max_size", "minCircularity": "min_circularity",
        "maxCircularity": "max_circularity", "smallSizeArea": "small_size_area",
        "smallSizeCircularity": "small_size_circularity", "minConvexity": "min_convexity",
        "maxConvexity": "max_convexity", "minInertia": "min_inertia", "maxInertia": "max_inertia",
        "minDist": "min_dist", "minDist_pin": "min_dist_pin", "circularity_threshold": "circularity_threshold",
        "circularity_threshold_veryBad": "circularity_threshold_very_bad", "area_segment_min": "area_segment_min",
        "area_segment_max": "area_segment_max", "segment_bias": "segment_bias", "filter_bias": "filter_bias",
        "post_minSize": "post_min_size", "post_minCircularity": "post_min_circularity",
        "post_minConvexity": "post_min_convexity", "post_minInertia": "post_min_inertia",
        "randomWalker_beta": "random_walker_beta", "randomWalker_method": "random_walker_mode",
        "randowWalker_maxiSize": "random_walker_maxi_size",
    }

    @classmethod
    def from_camii_configure(cls, path: str | os.PathLike, **overrides) -> "DetectionConfig":
        """Read a CAMII-style ``key=value`` configure file. Unknown keys are ignored."""
        cfg = cls()
        base = Path(path).resolve().parent
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = (s.strip() for s in line.split("=", 1))
            name = cls._CAMII_KEYS.get(key)
            if name is None:
                continue
            if name == "calib_parameter_path":
                p = Path(value)
                setattr(cfg, name, str(p if p.is_absolute() else (base / p)))
                continue
            current = getattr(cfg, name)
            setattr(cfg, name, type(current)(value) if not isinstance(current, str) else value)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


# --------------------------------------------------------------------------------------
# Image pairs
# --------------------------------------------------------------------------------------
def find_plate_images(folder: str | os.PathLike) -> dict[str, tuple[Path, Path]]:
    """Map plate barcode -> (trans image, epi image).

    CAMII naming: ``<barcode>_<timestamp>.bmp``; the trans-illuminated image is taken first,
    so the earlier timestamp is trans and the later one epi. With more than two images of
    a plate, the two most recent ones are used.
    """
    groups: dict[str, list[tuple[int, Path]]] = {}
    for p in Path(folder).iterdir():
        if p.suffix.lower() != ".bmp":
            continue
        m = re.match(r"^(.+?)_(\d+)$", p.stem)
        if not m:
            continue
        groups.setdefault(m.group(1), []).append((int(m.group(2)), p))
    pairs = {}
    for barcode, items in groups.items():
        items.sort()
        if len(items) >= 2:
            pairs[barcode] = (items[-2][1], items[-1][1])
    return pairs


def make_calibration(trans_paths, epi_paths, cfg: DetectionConfig, out_path: str | os.PathLike,
                     kernel=(27, 27), iterations: int = 20) -> None:
    """Build a flat-field calibration file from images of blank plates (CAMII ``calculate_calib_image``)."""
    def crop(im):
        return im[cfg.crop_y_min:cfg.crop_y_max, cfg.crop_x_min:cfg.crop_x_max]

    def smooth_norm(stack):
        m = np.mean(stack, axis=0).astype(np.float32)
        for _ in range(iterations):
            m = cv2.GaussianBlur(m, kernel, 0)
        return m / np.mean(m)

    trans = [crop(cv2.imread(str(p), 0)).astype(np.float32) for p in trans_paths]
    epi = [crop(cv2.imread(str(p))).astype(np.float32) for p in epi_paths]
    np.savez(out_path,
             image_trans_calib=smooth_norm(trans),
             image_epi_calib_B=smooth_norm([e[:, :, 0] for e in epi]),
             image_epi_calib_G=smooth_norm([e[:, :, 1] for e in epi]),
             image_epi_calib_R=smooth_norm([e[:, :, 2] for e in epi]))


_CALIB_CACHE: dict = {}


def _load_calibration(cfg: DetectionConfig, shape):
    if cfg.calib_parameter_path and Path(cfg.calib_parameter_path).exists():
        key = str(Path(cfg.calib_parameter_path).resolve())
        if key not in _CALIB_CACHE:
            z = np.load(cfg.calib_parameter_path)
            _CALIB_CACHE.clear()
            _CALIB_CACHE[key] = (z["image_trans_calib"], z["image_epi_calib_B"], z["image_epi_calib_G"],
                                 z["image_epi_calib_R"])
        cal = _CALIB_CACHE[key]
        if cal[0].shape != shape:
            raise ValueError(f"calibration shape {cal[0].shape} does not match cropped image {shape}; "
                             f"check crop settings or rebuild the calibration (make_calibration)")
        return cal
    ones = np.ones(shape, np.float32)
    return ones, ones, ones, ones


# --------------------------------------------------------------------------------------
# Contour geometry (CAMII definitions)
# --------------------------------------------------------------------------------------
def _centroid(m):
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def _radius(contour, x, y):
    d = np.sort(np.linalg.norm(contour[:, 0, :] - np.array([x, y]), axis=1))
    n = len(d)
    return (d[(n - 1) // 2] + d[n // 2]) / 2


def _inertia(m):
    # original CAMII expression (see module docstring)
    denom = np.sqrt((2 * m["mu11"]) ** 2) + (m["mu20"] - m["mu02"]) ** 2
    if denom <= 0.01:
        return 1.0
    cosmin = (m["mu20"] - m["mu02"]) / denom
    sinmin = 2 * m["mu11"] / denom
    imin = 0.5 * (m["mu20"] + m["mu02"]) - 0.5 * (m["mu20"] - m["mu02"]) * cosmin - m["mu11"] * sinmin
    imax = 0.5 * (m["mu20"] + m["mu02"]) - 0.5 * (m["mu20"] - m["mu02"]) * (-cosmin) - m["mu11"] * (-sinmin)
    return imin / imax


def _shape(contour):
    m = cv2.moments(contour)
    area = m["m00"]
    perim = cv2.arcLength(contour, True)
    hull = cv2.contourArea(cv2.convexHull(contour))
    return {
        "m": m, "area": area, "perim": perim,
        "circ": (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0,
        "conv": area / hull if hull > 0 else 0.0,
        "inertia": _inertia(m) if area > 0 else 1.0,
    }


def _distance_filter(contours, centers, min_dist, min_dist_pin):
    """Keep contours whose centroid is >= min_dist from other centroids and >= min_dist_pin
    from any point of another contour (so the picking pin hits one colony only)."""
    keep = []
    centers = np.asarray(centers, float)
    for i, (cx, cy) in enumerate(centers):
        others = [j for j in range(len(contours)) if j != i]
        if not others:
            keep.append(i)
            continue
        d = np.linalg.norm(centers[others] - centers[i], axis=1)
        valid = d > 0.001
        if not valid.any():
            keep.append(i)
            continue
        pin = min(np.linalg.norm(contours[j][:, 0, :] - centers[i], axis=1).min()
                  for j, v in zip(others, valid) if v)
        if d[valid].min() < min_dist or pin < min_dist_pin:
            continue
        keep.append(i)
    return keep


def _first_filter(contours, cfg):
    shapes, kept = [], []
    for c in contours:
        s = _shape(c)
        if s["area"] == 0:
            continue
        if not (cfg.min_size <= s["area"] <= cfg.max_size):
            continue
        if not (cfg.min_circularity <= s["circ"] <= cfg.max_circularity):
            continue
        if s["circ"] < cfg.small_size_circularity and s["area"] < cfg.small_size_area:
            continue
        if not (cfg.min_convexity <= s["conv"] <= cfg.max_convexity):
            continue
        if not (cfg.min_inertia <= s["inertia"] <= cfg.max_inertia):
            continue
        kept.append(c)
        shapes.append(s)
    # distance test against ALL non-degenerate contours (as in CAMII)
    all_valid = [c for c in contours if cv2.moments(c)["m00"] != 0]
    centers_all = [_centroid(cv2.moments(c)) for c in all_valid]
    out, out_shapes = [], []
    for c, s in zip(kept, shapes):
        cx, cy = _centroid(s["m"])
        d = np.linalg.norm(np.asarray(centers_all, float) - (cx, cy), axis=1)
        mask = d > 0.001
        if mask.any():
            pin = min(np.linalg.norm(oc[:, 0, :] - (cx, cy), axis=1).min()
                      for oc, v in zip(all_valid, mask) if v)
            if d[mask].min() < cfg.min_dist or pin < cfg.min_dist_pin:
                continue
        out.append(c)
        out_shapes.append(s)
    return out, out_shapes


def _bbox_shift(contour, bias):
    xs, ys = contour[:, 0, 0], contour[:, 0, 1]
    x0, x1, y0, y1 = xs.min() - bias, xs.max() + bias, ys.min() - bias, ys.max() + bias
    shifted = contour.copy()
    shifted[:, 0, 0] -= x0
    shifted[:, 0, 1] -= y0
    return shifted, int(x0), int(x1), int(y0), int(y1)


def _random_walker_split(sub_image, sub_binary, cfg):
    from skimage.feature import peak_local_max
    from skimage.measure import label
    from skimage.segmentation import random_walker

    distance = ndimage.distance_transform_edt(sub_image)
    fp = np.ones((cfg.random_walker_maxi_size, cfg.random_walker_maxi_size))
    coords = peak_local_max(distance, footprint=fp, labels=sub_binary)
    local_maxi = np.zeros(distance.shape, dtype=bool)
    if len(coords):
        local_maxi[tuple(coords.T)] = True
    markers = label(local_maxi)
    markers[sub_binary == 0] = -1
    if markers.max() < 1:
        return np.zeros_like(markers)
    return random_walker(sub_image, markers, beta=cfg.random_walker_beta, mode=cfg.random_walker_mode,
                         channel_axis=None)


def _split_merged(contours, shapes, trans_corrected, cfg):
    h, w = trans_corrected.shape
    out = []
    for c, s in zip(contours, shapes):
        _, x0, x1, y0, y1 = _bbox_shift(c, cfg.segment_bias + cfg.filter_bias)
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            continue                       # too close to the edge of the crop (dropped, as in CAMII)
        circ, area = s["circ"], s["area"]
        needs_split = ((cfg.circularity_threshold_very_bad <= circ < cfg.circularity_threshold
                        and cfg.area_segment_min < area < cfg.area_segment_max)
                       or circ < cfg.circularity_threshold_very_bad)
        if not needs_split:
            if circ > cfg.post_min_circularity and s["conv"] > cfg.post_min_convexity:
                out.append(c)
            continue
        shifted, x0, x1, y0, y1 = _bbox_shift(c, cfg.segment_bias)
        binary = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.drawContours(binary, [shifted], 0, 255, -1)
        cv2.drawContours(binary, [shifted], 0, 255, 1)
        patch = trans_corrected[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
        sub = cv2.bitwise_and(patch, patch, mask=binary)
        labels = _random_walker_split(sub.copy(), binary.copy(), cfg)
        for i in range(1, int(labels.max()) + 1):
            tmp = np.where(labels == i, 255, 0).astype(np.uint8)
            tmp = cv2.GaussianBlur(tmp, (3, 3), 0)
            _, b = cv2.threshold(tmp, 100, 255, cv2.THRESH_BINARY)
            found = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if not found:
                continue
            piece = found[0].copy()
            piece[:, :, 0] += x0
            piece[:, :, 1] += y0
            ps = _shape(piece)
            if ps["area"] == 0 or not (cfg.post_min_size <= ps["area"] <= cfg.max_size):
                continue
            if ps["circ"] < cfg.post_min_circularity or ps["conv"] < cfg.post_min_convexity:
                continue
            if not (cfg.post_min_inertia <= ps["inertia"] <= cfg.max_inertia):
                continue
            out.append(piece)
    return out


def _features(contours, trans_corrected, epi_corrected, cfg):
    h, w = trans_corrected.shape
    rows = []
    for c in contours:
        s = _shape(c)
        x, y = _centroid(s["m"])
        black = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(black, c, 255)
        bx, by, bw, bh = cv2.boundingRect(c)
        msk = black[by:by + bh + 1, bx:bx + bw + 1]
        tpatch = trans_corrected[by:by + bh + 1, bx:bx + bw + 1]
        epatch = epi_corrected[by:by + bh + 1, bx:bx + bw + 1]
        t = cv2.bitwise_and(tpatch, tpatch, mask=msk)
        e = cv2.bitwise_and(epatch, epatch, mask=msk)

        def stats(a):
            v = a.ravel()
            v = v[v != 0]
            if v.size == 0:
                return np.nan, np.nan
            mu = float(v.mean())
            return mu, float(v.std() / mu) if mu else np.nan

        gm, gs = stats(t)
        bm, bs = stats(e[:, :, 0])
        gm_e, gs_e = stats(e[:, :, 1])
        rm, rs = stats(e[:, :, 2])
        rows.append({"X": x, "Y": y, "Area": s["area"], "Perimeter": s["perim"], "Radius": _radius(c, x, y),
                     "Circularity": s["circ"], "Convexity": s["conv"], "Inertia": s["inertia"],
                     "Graymean": gm, "Graystd": gs, "Repimean": rm, "Repistd": rs,
                     "Gepimean": gm_e, "Gepistd": gs_e, "Bepimean": bm, "Bepistd": bs})
    df = pd.DataFrame(rows, columns=["X", "Y"] + FEATURES)
    df["X_rawImage"] = df["X"] + cfg.crop_x_min
    df["Y_rawImage"] = df["Y"] + cfg.crop_y_min
    return df


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
@dataclass
class PlateResult:
    barcode: str
    colonies: pd.DataFrame                 # one row per colony: X, Y (cropped coords), 14 features, raw coords
    contours: list = field(repr=False)     # OpenCV contours in cropped coordinates
    trans: np.ndarray | None = field(default=None, repr=False)  # corrected trans image (cropped, float32)
    epi: np.ndarray | None = field(default=None, repr=False)    # corrected epi image (cropped, float32, BGR)
    trans_path: str | None = None          # source images, used to reload the trans image when
    epi_path: str | None = None            # images were not kept in memory (large runs)
    cfg: "DetectionConfig | None" = field(default=None, repr=False)

    def trans_image(self) -> np.ndarray:
        """Corrected trans image; reloaded from disk if it was not kept in memory."""
        if self.trans is None:
            cfg = self.cfg or DetectionConfig()
            raw = cv2.imread(str(self.trans_path), 0)
            crop = raw[cfg.crop_y_min:cfg.crop_y_max, cfg.crop_x_min:cfg.crop_x_max]
            cal_t = _load_calibration(cfg, crop.shape)[0]
            return (crop.astype(np.float32) / cal_t.astype(np.float32)) * cfg.calib_contrast_trans_alpha \
                + cfg.calib_contrast_trans_beta
        return self.trans


def detect_plate(trans_path, epi_path, cfg: DetectionConfig | None = None, barcode: str | None = None) -> PlateResult:
    """Detect and measure colonies on one plate."""
    cfg = cfg or DetectionConfig()
    rng = np.random.default_rng(cfg.seed)
    trans_raw = cv2.imread(str(trans_path), 0)
    epi_raw = cv2.imread(str(epi_path))
    if trans_raw is None or epi_raw is None:
        raise FileNotFoundError(f"cannot read {trans_path} / {epi_path}")
    sl = (slice(cfg.crop_y_min, cfg.crop_y_max), slice(cfg.crop_x_min, cfg.crop_x_max))
    trans_crop, epi_crop = trans_raw[sl], epi_raw[sl]
    cal_t, cal_b, cal_g, cal_r = _load_calibration(cfg, trans_crop.shape)

    trans = (trans_crop.astype(np.float32) / cal_t.astype(np.float32)) * cfg.calib_contrast_trans_alpha \
        + cfg.calib_contrast_trans_beta
    epi = epi_crop.astype(np.float32)
    epi[:, :, 0] /= cal_b.astype(np.float32)
    epi[:, :, 1] /= cal_g.astype(np.float32)
    epi[:, :, 2] /= cal_r.astype(np.float32)

    # foreground mask: sharpen + adaptive Gaussian threshold (inverted: colonies are dark in trans)
    kernel = 1.0 / (18 - 8) * np.array([[-1, -1, -1], [-1, 18, -1], [-1, -1, -1]], np.float32)
    sharp = cv2.filter2D(trans, -1, kernel)
    mask_bg = cv2.adaptiveThreshold(np.clip(sharp, 0, 255).astype(np.uint8), 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                    cfg.bg_threshold_block_size, cfg.bg_threshold_offset)
    fg = cv2.bitwise_and(trans, trans, mask=mask_bg)

    # edges -> contours
    blur = cv2.GaussianBlur(fg, (5, 5), 0)
    sample = rng.choice(blur.ravel(), cfg.size_subsample, replace=True)
    sample = sample[sample != 0]
    upper = float(np.percentile(sample, cfg.canny_upper_percentile)) if sample.size else 255.0
    edges = cv2.Canny(np.clip(blur, 0, 255).astype(np.uint8), 1, upper)
    edges = cv2.dilate(edges, None, iterations=3)
    edges = cv2.erode(edges, None, iterations=1)
    contours = list(cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2])

    kept, shapes = _first_filter(contours, cfg)
    split = _split_merged(kept, shapes, trans, cfg)
    centers = [_centroid(cv2.moments(c)) for c in split]
    final = [split[i] for i in _distance_filter(split, centers, cfg.min_dist, cfg.min_dist_pin)]
    colonies = _features(final, trans, epi, cfg)
    colonies.insert(0, "plate_barcode", barcode or Path(trans_path).stem.split("_")[0])
    colonies.insert(1, "colony_index", np.arange(len(colonies)))
    return PlateResult(barcode=colonies["plate_barcode"].iat[0] if len(colonies) else (barcode or ""),
                       colonies=colonies, contours=final, trans=trans, epi=epi,
                       trans_path=str(trans_path), epi_path=str(epi_path), cfg=cfg)


def detect_folder(folder, cfg: DetectionConfig | None = None, barcodes=None, progress=print,
                  keep_images: bool | None = None) -> dict[str, PlateResult]:
    """Detect colonies on every plate (image pair) in a folder.

    Images are kept in memory only for small runs (<= 20 plates by default); otherwise they are
    dropped after detection and reloaded when an annotated image is drawn. A plate that fails
    (unreadable image, etc.) is reported and skipped."""
    pairs = find_plate_images(folder)
    if barcodes is not None:
        wanted = set(barcodes)
        pairs = {b: p for b, p in pairs.items() if b in wanted}
    if keep_images is None:
        keep_images = len(pairs) <= 20
    out = {}
    for i, (b, (tp, ep)) in enumerate(sorted(pairs.items())):
        try:
            res = detect_plate(tp, ep, cfg, barcode=b)
        except Exception as e:  # noqa: BLE001 - one bad plate should not stop a large run
            if progress:
                progress(f"  [{i + 1}/{len(pairs)}] {b}: FAILED ({type(e).__name__}: {e})")
            continue
        if not keep_images:
            res.trans = res.epi = None
        out[b] = res
        if progress:
            progress(f"  [{i + 1}/{len(pairs)}] {b}: {len(res.colonies)} colonies")
    return out


def draw_colonies(result: PlateResult, picks: pd.DataFrame | None = None, path=None) -> np.ndarray:
    """Trans image with all colony contours (grey) and picked colonies (green ring + pick order)."""
    img = np.clip(result.trans_image(), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c in result.contours:
        cv2.drawContours(img, [c], 0, (90, 90, 90), 1)
    if picks is not None and len(picks):
        for _, r in picks.iterrows():
            c = result.contours[int(r["colony_index"])]
            cv2.drawContours(img, [c], 0, (0, 200, 0), 2)
            org = (int(r["X"]) + 10, int(r["Y"]) - 10)
            label = str(int(r["pick_rank"]))
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 0), 2, cv2.LINE_AA)
    if path:
        cv2.imwrite(str(path), img)
    return img

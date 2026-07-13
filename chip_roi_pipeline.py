#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified OpenCV chip calibration and batch-cropping pipeline.

Functions
---------
1. init-config
   Write a YAML/JSON configuration containing both SRS and HX parameters.
2. calibrate
   Calibrate SRS thresholds or HX background samples on one representative image,
   preview the located chip, and save parameters back to the configuration file.
3. crop
   Batch-locate chips using the saved configuration and crop images/masks while
   preserving train/val/test directory structure. A CSV manifest and failure log
   are generated.

Examples
--------
python chip_roi_pipeline.py init-config --config chip_roi_config.yaml

python chip_roi_pipeline.py calibrate \
  --config chip_roi_config.yaml \
  --mode srs \
  --image sample.bmp \
  --interactive

python chip_roi_pipeline.py calibrate \
  --config chip_roi_config.yaml \
  --mode hx \
  --image sample.bmp \
  --interactive

python chip_roi_pipeline.py crop \
  --config chip_roi_config.yaml \
  --src /path/to/dataset \
  --out /path/to/chip_crops \
  --rebuild --save-preview
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "active_mode": "srs",
    "srs": {
        "saturation_threshold": 75,
        "value_threshold": 35,
        "max_side": 1024,
        "pre_open_size": 3,
        "close_w_fraction": 0.004,
        "close_h_fraction": 0.004,
        "close_iterations": 1,
        "post_open_size": 0,
        "component_min_area_fraction": 0.01,
        "projection_source": "largest_component",
        "column_min_fraction": 0.05,
        "row_min_fraction": 0.05,
    },
    "hx": {
        "background_samples": [],
        "h_margin": 10,
        "s_margin": 45,
        "v_margin": 45,
        "min_value": 8,
        "max_side": 1024,
        "pre_open_size": 3,
        "close_w_fraction": 0.004,
        "close_h_fraction": 0.004,
        "close_iterations": 1,
        "post_open_size": 0,
        "component_min_area_fraction": 0.01,
        "projection_source": "largest_component",
        "column_min_fraction": 0.05,
        "row_min_fraction": 0.05,
    },
    "crop": {
        # Pixel margins are applied first, then ratio margins.
        # Order: left, top, right, bottom.
        "margin_px": [20, 10, 20, 10],
        "margin_ratio": [0.01, 0.005, 0.01, 0.005],
        "failure_policy": "skip",  # skip | full_image
        "preserve_extension": True,
        "mask_extension": ".png",
    },
    "validation": {
        "min_area_fraction": 0.08,
        "max_area_fraction": 0.995,
        "min_width_fraction": 0.20,
        "min_height_fraction": 0.20,
        "max_aspect_ratio": 12.0,
    },
    # Optional chip-relative regions. Example:
    # {"name": "left", "ratio_xyxy": [0.0, 0.0, 0.5, 1.0]}
    "rois": [],
    "calibration": {
        "reference_image": "",
        "reference_width": 0,
        "reference_height": 0,
    },
}


@dataclass
class LocateResult:
    bbox: tuple[int, int, int, int] | None
    raw_bbox: tuple[int, int, int, int] | None
    mode: str
    valid: bool
    reason: str = ""
    fallback: str = "none"
    score: float = 0.0
    debug: dict[str, np.ndarray] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid configuration: {path}")
    return deep_merge(DEFAULT_CONFIG, data)


def save_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )


# -----------------------------------------------------------------------------
# Robust image I/O
# -----------------------------------------------------------------------------


def robust_imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    path = Path(path)

    # Standard OpenCV path first.
    image = cv2.imread(str(path), flags)
    if image is not None:
        return image

    # Handles occasional path/decoder issues.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size:
            image = cv2.imdecode(data, flags)
            if image is not None:
                return image
    except Exception:
        pass

    # Pillow fallback for unusual BMP/TIFF variants.
    try:
        with Image.open(path) as pil_image:
            pil_image.load()
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.asarray(pil_image.convert("L"))
            if flags == cv2.IMREAD_UNCHANGED:
                return np.asarray(pil_image)
            rgb = np.asarray(pil_image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def robust_imwrite(path: Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


# -----------------------------------------------------------------------------
# Dataset traversal
# -----------------------------------------------------------------------------


def iter_images(src: Path) -> list[tuple[Path, str]]:
    image_root = src / "images"
    if image_root.exists():
        rows: list[tuple[Path, str]] = []
        split_dirs = sorted(p for p in image_root.iterdir() if p.is_dir())
        if split_dirs:
            for split_dir in split_dirs:
                for image_path in sorted(split_dir.rglob("*")):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                        rows.append((image_path, split_dir.name))
            if rows:
                return rows
        for image_path in sorted(image_root.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                rows.append((image_path, "all"))
        if rows:
            return rows

    return [
        (p, "all")
        for p in sorted(src.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and {"labels", "label", "masks", "mask"}.isdisjoint({part.lower() for part in p.parts})
    ]


def find_mask(src: Path, image_path: Path, split: str) -> Path | None:
    roots = [src / "labels" / split, src / "masks" / split, src / "labels", src / "masks"]
    for root in roots:
        if not root.exists():
            continue
        for ext in MASK_EXTS:
            candidate = root / f"{image_path.stem}{ext}"
            if candidate.exists():
                return candidate
        matches = [p for p in root.rglob(f"{image_path.stem}.*") if p.suffix.lower() in MASK_EXTS]
        if matches:
            return sorted(matches)[0]
    return None


# -----------------------------------------------------------------------------
# Geometry and masks
# -----------------------------------------------------------------------------


def odd_kernel(value: int, maximum: int | None = None) -> int:
    value = max(0, int(value))
    if maximum is not None:
        value = min(value, maximum)
    if value > 0 and value % 2 == 0:
        value += 1
    return value


def clamp_bbox(
    bbox: tuple[int | float, int | float, int | float, int | float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_px: Iterable[float],
    margin_ratio: Iterable[float],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    chip_w = max(1, x2 - x1)
    chip_h = max(1, y2 - y1)
    px = list(margin_px)
    ratio = list(margin_ratio)
    if len(px) != 4 or len(ratio) != 4:
        raise ValueError("crop.margin_px and crop.margin_ratio must contain 4 values")
    left = px[0] + chip_w * ratio[0]
    top = px[1] + chip_h * ratio[1]
    right = px[2] + chip_w * ratio[2]
    bottom = px[3] + chip_h * ratio[3]
    return clamp_bbox((x1 - left, y1 - top, x2 + right, y2 + bottom), width, height)


def best_projection_run(indices: np.ndarray, total_length: int) -> tuple[int, int] | None:
    if indices.size == 0:
        return None
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    last = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == last + 1:
            last = value
        else:
            runs.append((start, last + 1))
            start = last = value
    runs.append((start, last + 1))

    min_len = max(8, int(total_length * 0.03))
    runs = [run for run in runs if run[1] - run[0] >= min_len]
    if not runs:
        return None
    return max(runs, key=lambda run: run[1] - run[0])


def projection_bbox(mask: np.ndarray, col_fraction: float, row_fraction: float) -> tuple[int, int, int, int] | None:
    h, w = mask.shape[:2]
    col_indices = np.where(mask.sum(axis=0) > 255 * h * float(col_fraction))[0]
    row_indices = np.where(mask.sum(axis=1) > 255 * w * float(row_fraction))[0]
    x_run = best_projection_run(col_indices, w)
    y_run = best_projection_run(row_indices, h)
    if x_run is None or y_run is None:
        return None
    return x_run[0], y_run[0], x_run[1], y_run[1]


def largest_component(mask: np.ndarray, min_area_fraction: float) -> tuple[np.ndarray, tuple[int, int, int, int] | None, int]:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask), None, 0

    min_area = mask.shape[0] * mask.shape[1] * float(min_area_fraction)
    best_label = 0
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_label = label
            best_area = area

    if best_label == 0:
        return np.zeros_like(mask), None, 0

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    component_mask = ((labels == best_label).astype(np.uint8) * 255)
    return component_mask, (x, y, x + w, y + h), best_area


def morphology_cleanup(mask: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    opened = mask
    pre_open = odd_kernel(int(params.get("pre_open_size", 0)), 101)
    if pre_open > 1:
        kernel = np.ones((pre_open, pre_open), np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    h, w = mask.shape[:2]
    close_w = odd_kernel(round(w * float(params.get("close_w_fraction", 0.0))), 301)
    close_h = odd_kernel(round(h * float(params.get("close_h_fraction", 0.0))), 301)
    closed = opened
    if close_w > 1 or close_h > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, close_w), max(1, close_h)))
        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=max(1, int(params.get("close_iterations", 1))),
        )

    post_open = odd_kernel(int(params.get("post_open_size", 0)), 101)
    if post_open > 1:
        kernel = np.ones((post_open, post_open), np.uint8)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened, closed


def validate_bbox(
    bbox: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    validation: dict[str, Any],
) -> tuple[bool, str, float]:
    if bbox is None:
        return False, "bbox_not_found", 0.0
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    bw = x2 - x1
    bh = y2 - y1
    image_area = max(1, width * height)
    area_fraction = (bw * bh) / image_area
    width_fraction = bw / max(1, width)
    height_fraction = bh / max(1, height)
    aspect = max(bw / max(1, bh), bh / max(1, bw))

    if area_fraction < float(validation.get("min_area_fraction", 0.0)):
        return False, f"area_too_small:{area_fraction:.4f}", area_fraction
    if area_fraction > float(validation.get("max_area_fraction", 1.0)):
        return False, f"area_too_large:{area_fraction:.4f}", area_fraction
    if width_fraction < float(validation.get("min_width_fraction", 0.0)):
        return False, f"width_too_small:{width_fraction:.4f}", area_fraction
    if height_fraction < float(validation.get("min_height_fraction", 0.0)):
        return False, f"height_too_small:{height_fraction:.4f}", area_fraction
    if aspect > float(validation.get("max_aspect_ratio", math.inf)):
        return False, f"aspect_too_large:{aspect:.3f}", area_fraction
    return True, "ok", area_fraction


def scaled_background_points(samples: list[dict[str, Any]], width: int, height: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        ratio = sample.get("xy_ratio")
        xy = sample.get("xy")
        if isinstance(ratio, (list, tuple)) and len(ratio) == 2:
            x = int(round(float(ratio[0]) * max(1, width - 1)))
            y = int(round(float(ratio[1]) * max(1, height - 1)))
        elif isinstance(xy, (list, tuple)) and len(xy) == 2:
            x, y = int(xy[0]), int(xy[1])
        else:
            continue
        points.append((max(0, min(width - 1, x)), max(0, min(height - 1, y))))
    return points


def hsv_background_mask(
    hsv: np.ndarray,
    points: list[tuple[int, int]],
    h_margin: int,
    s_margin: int,
    v_margin: int,
) -> np.ndarray:
    background = np.zeros(hsv.shape[:2], dtype=np.uint8)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    for x, y in points:
        h0, s0, v0 = [int(v) for v in hsv[y, x]]
        if h_margin >= 90:
            hue_mask = np.full(hsv.shape[:2], 255, dtype=np.uint8)
        else:
            low_h = h0 - h_margin
            high_h = h0 + h_margin
            if low_h < 0:
                hue_mask = cv2.inRange(hue, 0, high_h) | cv2.inRange(hue, 180 + low_h, 179)
            elif high_h > 179:
                hue_mask = cv2.inRange(hue, low_h, 179) | cv2.inRange(hue, 0, high_h - 180)
            else:
                hue_mask = cv2.inRange(hue, low_h, high_h)
        sat_mask = cv2.inRange(sat, max(0, s0 - s_margin), min(255, s0 + s_margin))
        val_mask = cv2.inRange(val, max(0, v0 - v_margin), min(255, v0 + v_margin))
        background |= hue_mask & sat_mask & val_mask
    return background


# -----------------------------------------------------------------------------
# Chip location
# -----------------------------------------------------------------------------


def locate_on_scaled_image(
    image: np.ndarray,
    mode: str,
    params: dict[str, Any],
    validation: dict[str, Any],
) -> LocateResult:
    full_h, full_w = image.shape[:2]
    max_side = max(64, int(params.get("max_side", 1024)))
    scale = min(1.0, max_side / max(full_h, full_w))
    if scale < 1.0:
        small = cv2.resize(
            image,
            (max(1, int(round(full_w * scale))), max(1, int(round(full_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    if mode == "srs":
        raw_mask = (
            (hsv[:, :, 1] > int(params.get("saturation_threshold", 75)))
            & (hsv[:, :, 2] > int(params.get("value_threshold", 35)))
        ).astype(np.uint8) * 255
        background = np.zeros_like(raw_mask)
    elif mode == "hx":
        points = scaled_background_points(params.get("background_samples", []), small.shape[1], small.shape[0])
        if not points:
            return LocateResult(None, None, mode, False, "hx_background_samples_empty")
        background = hsv_background_mask(
            hsv,
            points,
            int(params.get("h_margin", 10)),
            int(params.get("s_margin", 45)),
            int(params.get("v_margin", 45)),
        )
        raw_mask = cv2.bitwise_not(background)
        min_value = int(params.get("min_value", 0))
        if min_value > 0:
            raw_mask[hsv[:, :, 2] < min_value] = 0
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    opened, closed = morphology_cleanup(raw_mask, params)
    largest, component_bbox, component_area = largest_component(
        closed,
        float(params.get("component_min_area_fraction", 0.01)),
    )

    masks = {
        "raw": raw_mask,
        "opened": opened,
        "closed": closed,
        "largest_component": largest,
    }
    projection_source = str(params.get("projection_source", "largest_component"))
    projection_mask = masks.get(projection_source, largest)

    small_bbox = projection_bbox(
        projection_mask,
        float(params.get("column_min_fraction", 0.05)),
        float(params.get("row_min_fraction", 0.05)),
    )
    fallback = "none"
    if small_bbox is None and component_bbox is not None:
        small_bbox = component_bbox
        fallback = "largest_component_bbox"
    if small_bbox is None:
        small_bbox = projection_bbox(
            closed,
            max(0.005, float(params.get("column_min_fraction", 0.05)) * 0.5),
            max(0.005, float(params.get("row_min_fraction", 0.05)) * 0.5),
        )
        if small_bbox is not None:
            fallback = "relaxed_projection"

    raw_bbox: tuple[int, int, int, int] | None = None
    if small_bbox is not None:
        inv = 1.0 / scale
        raw_bbox = clamp_bbox(
            (
                math.floor(small_bbox[0] * inv),
                math.floor(small_bbox[1] * inv),
                math.ceil(small_bbox[2] * inv),
                math.ceil(small_bbox[3] * inv),
            ),
            full_w,
            full_h,
        )

    valid, reason, score = validate_bbox(raw_bbox, full_w, full_h, validation)

    debug = {
        "background": background,
        "raw_mask": raw_mask,
        "opened": opened,
        "closed": closed,
        "largest_component": largest,
        "projection_source": projection_mask,
    }
    return LocateResult(
        bbox=raw_bbox if valid else None,
        raw_bbox=raw_bbox,
        mode=mode,
        valid=valid,
        reason=reason,
        fallback=fallback,
        score=score,
        debug=debug,
    )


def locate_chip(image: np.ndarray, config: dict[str, Any], mode: str | None = None) -> LocateResult:
    mode = str(mode or config.get("active_mode", "srs")).lower()
    if mode not in {"srs", "hx"}:
        raise ValueError(f"Unsupported mode: {mode}")
    params = config[mode]
    result = locate_on_scaled_image(image, mode, params, config.get("validation", {}))

    if result.valid and result.bbox is not None:
        h, w = image.shape[:2]
        crop_cfg = config.get("crop", {})
        result.bbox = expand_bbox(
            result.bbox,
            w,
            h,
            crop_cfg.get("margin_px", [0, 0, 0, 0]),
            crop_cfg.get("margin_ratio", [0, 0, 0, 0]),
        )
    return result


# -----------------------------------------------------------------------------
# Visualization and interactive calibration
# -----------------------------------------------------------------------------


def draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    label: str,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> np.ndarray:
    out = image.copy()
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        out,
        label,
        (x1, max(28, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def fit_for_display(image: np.ndarray, max_display: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_display / max(h, w))
    if scale < 1.0:
        return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA), scale
    return image.copy(), 1.0


def choose_image_dialog(initial_dir: Path) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Select representative chip image",
        initialdir=str(initial_dir),
        filetypes=[("Images", "*.bmp *.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


def collect_background_points(image: np.ndarray, max_display: int) -> list[dict[str, Any]]:
    display, scale = fit_for_display(image, max_display)
    h, w = image.shape[:2]
    points: list[tuple[int, int]] = []
    window = "HX background samples: left click add | right click undo | Enter finish | Esc cancel"

    def redraw() -> None:
        canvas = display.copy()
        for index, (x, y) in enumerate(points, start=1):
            dx, dy = int(round(x * scale)), int(round(y * scale))
            cv2.circle(canvas, (dx, dy), 6, (0, 0, 255), -1)
            cv2.putText(canvas, str(index), (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.imshow(window, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            ox = max(0, min(w - 1, int(round(x / scale))))
            oy = max(0, min(h - 1, int(round(y / scale))))
            points.append((ox, oy))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):
            break
        if key == 27:
            points = []
            break
    cv2.destroyWindow(window)

    samples = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for x, y in points:
        h0, s0, v0 = [int(v) for v in hsv[y, x]]
        b0, g0, r0 = [int(v) for v in image[y, x]]
        samples.append(
            {
                "xy": [x, y],
                "xy_ratio": [round(x / max(1, w - 1), 8), round(y / max(1, h - 1), 8)],
                "hsv": [h0, s0, v0],
                "bgr": [b0, g0, r0],
            }
        )
    return samples


def interactive_calibrate(
    image: np.ndarray,
    config: dict[str, Any],
    mode: str,
    max_display: int,
) -> bool:
    mode_cfg = config[mode]

    if mode == "hx" and not mode_cfg.get("background_samples"):
        samples = collect_background_points(image, max_display)
        if not samples:
            print("[CANCEL] No HX background sample selected.")
            return False
        mode_cfg["background_samples"] = samples

    window = f"Calibrate {mode.upper()}: Enter/S save | Esc/Q cancel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    if mode == "srs":
        cv2.createTrackbar("S threshold", window, int(mode_cfg.get("saturation_threshold", 75)), 255, lambda _v: None)
        cv2.createTrackbar("V threshold", window, int(mode_cfg.get("value_threshold", 35)), 255, lambda _v: None)
    else:
        cv2.createTrackbar("H margin", window, int(mode_cfg.get("h_margin", 10)), 90, lambda _v: None)
        cv2.createTrackbar("S margin", window, int(mode_cfg.get("s_margin", 45)), 255, lambda _v: None)
        cv2.createTrackbar("V margin", window, int(mode_cfg.get("v_margin", 45)), 255, lambda _v: None)
        cv2.createTrackbar("Min V", window, int(mode_cfg.get("min_value", 8)), 255, lambda _v: None)

    cv2.createTrackbar(
        "Column fraction %",
        window,
        int(round(float(mode_cfg.get("column_min_fraction", 0.05)) * 100)),
        50,
        lambda _v: None,
    )
    cv2.createTrackbar(
        "Row fraction %",
        window,
        int(round(float(mode_cfg.get("row_min_fraction", 0.05)) * 100)),
        50,
        lambda _v: None,
    )

    saved = False
    while True:
        if mode == "srs":
            mode_cfg["saturation_threshold"] = cv2.getTrackbarPos("S threshold", window)
            mode_cfg["value_threshold"] = cv2.getTrackbarPos("V threshold", window)
        else:
            mode_cfg["h_margin"] = cv2.getTrackbarPos("H margin", window)
            mode_cfg["s_margin"] = cv2.getTrackbarPos("S margin", window)
            mode_cfg["v_margin"] = cv2.getTrackbarPos("V margin", window)
            mode_cfg["min_value"] = cv2.getTrackbarPos("Min V", window)

        mode_cfg["column_min_fraction"] = max(0.005, cv2.getTrackbarPos("Column fraction %", window) / 100.0)
        mode_cfg["row_min_fraction"] = max(0.005, cv2.getTrackbarPos("Row fraction %", window) / 100.0)

        result = locate_chip(image, config, mode)
        preview = image.copy()
        if result.raw_bbox is not None:
            preview = draw_bbox(preview, result.raw_bbox, f"raw {result.reason}", (0, 255, 0), 2)
        if result.bbox is not None:
            preview = draw_bbox(preview, result.bbox, f"final {mode}", (255, 255, 0), 3)
        if mode == "hx":
            h, w = image.shape[:2]
            for sample in mode_cfg.get("background_samples", []):
                pts = scaled_background_points([sample], w, h)
                if pts:
                    cv2.circle(preview, pts[0], 10, (0, 0, 255), -1)

        status = f"valid={result.valid} score={result.score:.3f} fallback={result.fallback}"
        cv2.putText(preview, status, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
        display, _ = fit_for_display(preview, max_display)
        cv2.imshow(window, display)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, ord("s"), ord("S")):
            saved = result.valid
            if not saved:
                print(f"[WARN] Current bbox is invalid: {result.reason}")
            else:
                break
        elif key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyWindow(window)
    return saved


def parse_points(text: str, image: np.ndarray) -> list[dict[str, Any]]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    samples: list[dict[str, Any]] = []
    if not text.strip():
        return samples
    for item in text.replace("|", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        x_text, y_text = item.split(",", 1)
        x = max(0, min(w - 1, int(float(x_text))))
        y = max(0, min(h - 1, int(float(y_text))))
        h0, s0, v0 = [int(v) for v in hsv[y, x]]
        b0, g0, r0 = [int(v) for v in image[y, x]]
        samples.append(
            {
                "xy": [x, y],
                "xy_ratio": [round(x / max(1, w - 1), 8), round(y / max(1, h - 1), 8)],
                "hsv": [h0, s0, v0],
                "bgr": [b0, g0, r0],
            }
        )
    return samples


def save_calibration_debug(
    out_dir: Path,
    image: np.ndarray,
    result: LocateResult,
    config: dict[str, Any],
    mode: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    robust_imwrite(out_dir / "01_original.jpg", image)
    for index, (name, mask) in enumerate(result.debug.items(), start=2):
        robust_imwrite(out_dir / f"{index:02d}_{name}.png", mask)
    preview = image.copy()
    preview = draw_bbox(preview, result.raw_bbox, "raw", (0, 255, 0), 2)
    preview = draw_bbox(preview, result.bbox, f"final_{mode}", (255, 255, 0), 3)
    robust_imwrite(out_dir / "99_bbox_preview.jpg", preview)
    save_config(out_dir / "config_snapshot.yaml", config)


# -----------------------------------------------------------------------------
# Batch crop
# -----------------------------------------------------------------------------


def region_bbox(
    chip_bbox: tuple[int, int, int, int],
    ratio_xyxy: Iterable[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = chip_bbox
    cw = max(1, x2 - x1)
    ch = max(1, y2 - y1)
    rx1, ry1, rx2, ry2 = [float(v) for v in ratio_xyxy]
    return clamp_bbox((x1 + cw * rx1, y1 + ch * ry1, x1 + cw * rx2, y1 + ch * ry2), width, height)


def output_extension(image_path: Path, config: dict[str, Any]) -> str:
    if bool(config.get("crop", {}).get("preserve_extension", True)):
        return image_path.suffix
    return ".png"


def batch_crop(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    mode = args.mode or config.get("active_mode", "srs")
    config["active_mode"] = mode

    if args.rebuild and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = iter_images(args.src)
    if args.limit > 0:
        rows = rows[: args.limit]

    manifest_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    preview_count = 0
    processed = 0

    rois = config.get("rois", []) or []
    selected = None if args.regions == "all" else {x.strip() for x in args.regions.split(",") if x.strip()}

    for index, (image_path, split) in enumerate(rows, start=1):
        image = robust_imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            failed_rows.append({"image": str(image_path), "split": split, "reason": "read_failed"})
            print(f"[WARN] read failed: {image_path}")
            continue

        h, w = image.shape[:2]
        result = locate_chip(image, config, mode)
        final_bbox = result.bbox

        if final_bbox is None:
            policy = str(config.get("crop", {}).get("failure_policy", "skip"))
            if policy == "full_image":
                final_bbox = (0, 0, w, h)
                result.fallback = "full_image"
            else:
                failed_rows.append(
                    {
                        "image": str(image_path),
                        "split": split,
                        "reason": result.reason,
                        "raw_bbox": str(result.raw_bbox or ""),
                    }
                )
                print(f"[WARN] locate failed: {image_path} ({result.reason})")
                continue

        mask_path = None if args.no_labels else find_mask(args.src, image_path, split)
        mask = robust_imread(mask_path, cv2.IMREAD_UNCHANGED) if mask_path else None
        if mask is not None and mask.shape[:2] != image.shape[:2]:
            failed_rows.append(
                {
                    "image": str(image_path),
                    "split": split,
                    "reason": f"mask_shape_mismatch:{mask.shape[:2]}!={image.shape[:2]}",
                }
            )
            mask = None

        regions: list[tuple[str, tuple[int, int, int, int]]] = [("chip", final_bbox)]
        for roi in rois:
            if not isinstance(roi, dict) or "ratio_xyxy" not in roi:
                continue
            name = str(roi.get("name", "roi"))
            if selected is not None and name not in selected:
                continue
            regions.append((name, region_bbox(final_bbox, roi["ratio_xyxy"], w, h)))

        ext = output_extension(image_path, config)
        mask_ext = str(config.get("crop", {}).get("mask_extension", ".png"))
        if not mask_ext.startswith("."):
            mask_ext = "." + mask_ext

        for region_name, bbox in regions:
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            stem = f"{image_path.stem}__{region_name}"
            image_out = args.out / region_name / "images" / split / f"{stem}{ext}"
            robust_imwrite(image_out, crop)

            mask_out = ""
            if mask is not None:
                mask_crop = mask[y1:y2, x1:x2]
                out_path = args.out / region_name / "labels" / split / f"{stem}{mask_ext}"
                robust_imwrite(out_path, mask_crop)
                mask_out = str(out_path)

            manifest_rows.append(
                {
                    "split": split,
                    "source_image": str(image_path),
                    "source_mask": str(mask_path) if mask_path else "",
                    "mode": mode,
                    "region": region_name,
                    "valid": int(result.valid),
                    "reason": result.reason,
                    "fallback": result.fallback,
                    "score": f"{result.score:.6f}",
                    "raw_x1": result.raw_bbox[0] if result.raw_bbox else "",
                    "raw_y1": result.raw_bbox[1] if result.raw_bbox else "",
                    "raw_x2": result.raw_bbox[2] if result.raw_bbox else "",
                    "raw_y2": result.raw_bbox[3] if result.raw_bbox else "",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "crop_w": x2 - x1,
                    "crop_h": y2 - y1,
                    "out_image": str(image_out),
                    "out_mask": mask_out,
                }
            )

        if args.save_preview and preview_count < args.preview_limit:
            preview = image.copy()
            preview = draw_bbox(preview, result.raw_bbox, "raw", (0, 255, 0), 2)
            for region_name, bbox in regions:
                color = (255, 255, 0) if region_name == "chip" else (0, 0, 255)
                preview = draw_bbox(preview, bbox, region_name, color, 2)
            robust_imwrite(args.out / "_preview" / split / f"{image_path.stem}.jpg", preview)
            preview_count += 1

        processed += 1
        if index % 50 == 0 or index == len(rows):
            print(f"[INFO] {index}/{len(rows)} scanned, {processed} cropped, {len(failed_rows)} failed")

    manifest_path = args.out / "crop_manifest.csv"
    manifest_fields = [
        "split", "source_image", "source_mask", "mode", "region", "valid", "reason", "fallback", "score",
        "raw_x1", "raw_y1", "raw_x2", "raw_y2", "x1", "y1", "x2", "y2", "crop_w", "crop_h",
        "out_image", "out_mask",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    failed_path = args.out / "failed_images.csv"
    failed_fields = sorted({key for row in failed_rows for key in row}) or ["image", "split", "reason"]
    with failed_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=failed_fields)
        writer.writeheader()
        writer.writerows(failed_rows)

    print("\n========== Batch crop completed ==========")
    print(f"mode: {mode}")
    print(f"scanned: {len(rows)}")
    print(f"cropped images: {processed}")
    print(f"failed: {len(failed_rows)}")
    print(f"manifest: {manifest_path}")
    print(f"failed list: {failed_path}")
    print(f"output: {args.out}")


# -----------------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------------


def calibrate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    mode = args.mode
    config["active_mode"] = mode

    image_path = args.image
    if image_path is None and args.dialog:
        image_path = choose_image_dialog(args.src)
    if image_path is None:
        candidates = iter_images(args.src)
        if not candidates:
            raise FileNotFoundError(f"No image found under {args.src}")
        image_path = candidates[0][0]

    image_path = image_path.expanduser().resolve()
    image = robust_imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Failed to read image: {image_path}")

    mode_cfg = config[mode]

    # Manual command-line overrides.
    if mode == "srs":
        if args.s_threshold is not None:
            mode_cfg["saturation_threshold"] = args.s_threshold
        if args.v_threshold is not None:
            mode_cfg["value_threshold"] = args.v_threshold
    else:
        if args.points:
            mode_cfg["background_samples"] = parse_points(args.points, image)
        if args.h_margin is not None:
            mode_cfg["h_margin"] = args.h_margin
        if args.s_margin is not None:
            mode_cfg["s_margin"] = args.s_margin
        if args.v_margin is not None:
            mode_cfg["v_margin"] = args.v_margin
        if args.min_value is not None:
            mode_cfg["min_value"] = args.min_value

    if args.column_fraction is not None:
        mode_cfg["column_min_fraction"] = args.column_fraction
    if args.row_fraction is not None:
        mode_cfg["row_min_fraction"] = args.row_fraction

    if args.interactive:
        ok = interactive_calibrate(image, config, mode, args.max_display)
        if not ok:
            raise SystemExit("Calibration cancelled or bbox invalid; configuration was not saved.")
    else:
        if mode == "hx" and not mode_cfg.get("background_samples"):
            raise ValueError("HX non-interactive calibration requires --points or existing background_samples in config")
        result = locate_chip(image, config, mode)
        if not result.valid:
            raise RuntimeError(f"Calibration bbox invalid: {result.reason}")

    result = locate_chip(image, config, mode)
    h, w = image.shape[:2]
    config["calibration"] = {
        "reference_image": str(image_path),
        "reference_width": w,
        "reference_height": h,
        "last_mode": mode,
        "last_bbox_xyxy": list(result.bbox) if result.bbox else None,
        "last_raw_bbox_xyxy": list(result.raw_bbox) if result.raw_bbox else None,
        "last_score": round(result.score, 8),
    }
    save_config(args.config, config)

    debug_dir = args.out / image_path.stem
    save_calibration_debug(debug_dir, image, result, config, mode)

    print("\n========== Calibration saved ==========")
    print(f"mode: {mode}")
    print(f"image: {image_path}")
    print(f"raw bbox: {result.raw_bbox}")
    print(f"final bbox: {result.bbox}")
    print(f"score: {result.score:.4f}")
    print(f"config: {args.config}")
    print(f"debug: {debug_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init-config", help="Write a default YAML/JSON configuration")
    p_init.add_argument("--config", type=Path, required=True)

    p_cal = subparsers.add_parser("calibrate", help="Calibrate one representative image and save parameters")
    p_cal.add_argument("--config", type=Path, required=True)
    p_cal.add_argument("--mode", choices=("srs", "hx"), required=True)
    p_cal.add_argument("--image", type=Path, default=None)
    p_cal.add_argument("--src", type=Path, default=Path("."))
    p_cal.add_argument("--out", type=Path, default=Path("./chip_roi_calibration"))
    p_cal.add_argument("--dialog", action=argparse.BooleanOptionalAction, default=True)
    p_cal.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True)
    p_cal.add_argument("--max-display", type=int, default=1600)
    p_cal.add_argument("--s-threshold", type=int, default=None)
    p_cal.add_argument("--v-threshold", type=int, default=None)
    p_cal.add_argument("--points", default="", help='HX points, e.g. "100,100;300,120"')
    p_cal.add_argument("--h-margin", type=int, default=None)
    p_cal.add_argument("--s-margin", type=int, default=None)
    p_cal.add_argument("--v-margin", type=int, default=None)
    p_cal.add_argument("--min-value", type=int, default=None)
    p_cal.add_argument("--column-fraction", type=float, default=None)
    p_cal.add_argument("--row-fraction", type=float, default=None)

    p_crop = subparsers.add_parser("crop", help="Batch locate and crop images/masks")
    p_crop.add_argument("--config", type=Path, required=True)
    p_crop.add_argument("--src", type=Path, required=True)
    p_crop.add_argument("--out", type=Path, required=True)
    p_crop.add_argument("--mode", choices=("srs", "hx"), default=None)
    p_crop.add_argument("--regions", default="all", help="all or comma-separated ROI names")
    p_crop.add_argument("--rebuild", action="store_true")
    p_crop.add_argument("--save-preview", action="store_true")
    p_crop.add_argument("--preview-limit", type=int, default=50)
    p_crop.add_argument("--limit", type=int, default=0)
    p_crop.add_argument("--no-labels", action="store_true")

    return parser


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("config", "src", "out", "image"):
        value = getattr(args, name, None)
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def main() -> None:
    parser = build_parser()
    args = normalize_paths(parser.parse_args())

    if args.command == "init-config":
        save_config(args.config, DEFAULT_CONFIG)
        print(f"Default configuration written: {args.config}")
    elif args.command == "calibrate":
        calibrate(args)
    elif args.command == "crop":
        batch_crop(args)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

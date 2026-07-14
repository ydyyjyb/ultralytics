#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed SRS chip-body crop, ROI crop, and YOLO label conversion workflow."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")

DEFAULT_IMAGE_ROOT = Path(r"D:\liulanqi\code\Data\SRS+100+cropped\images")
DEFAULT_DATASET_ROOT = DEFAULT_IMAGE_ROOT.parent
DEFAULT_OUT_ROOT = Path(r"D:\liulanqi\code\Data\SRS+100+cropped+merge")
DEFAULT_CHIP_ROOT = DEFAULT_OUT_ROOT / "_chip_body"
DEFAULT_ROI_CONFIG = DEFAULT_OUT_ROOT / "roi_config.json"
DEFAULT_CLASS_NAMES = DEFAULT_DATASET_ROOT / "class_names.txt"


@dataclass(frozen=True)
class ImageItem:
    path: Path
    split: str
    rel: Path


def robust_imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    image = cv2.imread(str(path), flags)
    if image is not None:
        return image
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size:
            image = cv2.imdecode(data, flags)
            if image is not None:
                return image
    except Exception:
        pass
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
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def read_mask(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            image.load()
            return np.array(image)
    except Exception:
        return None


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def split_items(src: Path) -> list[ImageItem]:
    src = src.expanduser().resolve()
    if src.is_file():
        return [ImageItem(src, "all", Path(src.name))]

    image_root = src / "images" if (src / "images").is_dir() else src
    rows: list[ImageItem] = []
    for split in SPLITS:
        split_dir = image_root / split
        if not split_dir.is_dir():
            continue
        for path in image_files(split_dir):
            rows.append(ImageItem(path, split, path.relative_to(split_dir)))
    if rows:
        return rows

    for path in image_files(image_root):
        if {"labels", "label", "lable", "masks", "mask"}.isdisjoint({part.lower() for part in path.parts}):
            rows.append(ImageItem(path, "all", path.relative_to(image_root)))
    return rows


def dataset_root_from_image_src(src: Path) -> Path:
    src = src.expanduser().resolve()
    if src.is_file():
        return src.parent
    if src.name.lower() == "images":
        return src.parent
    if (src / "images").is_dir():
        return src
    return src


def find_mask(dataset_root: Path, item: ImageItem) -> Path | None:
    for root_name in ("labels", "label", "lable", "masks", "mask"):
        roots = [dataset_root / root_name / item.split, dataset_root / root_name]
        for root in roots:
            if not root.exists():
                continue
            stem_rel = item.rel.with_suffix("")
            for ext in MASK_EXTS:
                candidate = root / stem_rel.with_suffix(ext)
                if candidate.exists():
                    return candidate
            matches = sorted((root / stem_rel.parent).glob(stem_rel.name + ".*")) if (root / stem_rel.parent).exists() else []
            for match in matches:
                if match.suffix.lower() in MASK_EXTS:
                    return match
    return None


def best_projection_run(indices: np.ndarray, total_length: int) -> tuple[int, int] | None:
    if indices.size == 0:
        return None
    runs: list[tuple[int, int]] = []
    start = last = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == last + 1:
            last = value
            continue
        runs.append((start, last + 1))
        start = last = value
    runs.append((start, last + 1))
    min_len = max(8, int(total_length * 0.03))
    runs = [run for run in runs if run[1] - run[0] >= min_len]
    return max(runs, key=lambda run: run[1] - run[0]) if runs else None


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


def locate_srs_chip(
    image: np.ndarray,
    saturation_threshold: int = 75,
    value_threshold: int = 35,
    max_side: int = 1024,
    column_min_fraction: float = 0.05,
    row_min_fraction: float = 0.05,
    chip_margin: int = 0,
) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    small = (
        cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > saturation_threshold) & (hsv[:, :, 2] > value_threshold)).astype(np.uint8) * 255
    small_h, small_w = mask.shape[:2]
    cols = np.where(mask.sum(axis=0) > 255 * small_h * column_min_fraction)[0]
    rows = np.where(mask.sum(axis=1) > 255 * small_w * row_min_fraction)[0]
    x_run = best_projection_run(cols, small_w)
    y_run = best_projection_run(rows, small_h)
    if x_run is None or y_run is None:
        return 0, 0, width, height
    inv = 1.0 / scale
    return clamp_bbox(
        (
            int(np.floor(x_run[0] * inv)) - chip_margin,
            int(np.floor(y_run[0] * inv)) - chip_margin,
            int(np.ceil(x_run[1] * inv)) + chip_margin,
            int(np.ceil(y_run[1] * inv)) + chip_margin,
        ),
        width,
        height,
    )


def draw_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return out


def crop_pair(
    image_path: Path,
    mask_path: Path | None,
    image_out: Path,
    mask_out: Path | None,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, bool]:
    image = robust_imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"failed to read image: {image_path}")
    x1, y1, x2, y2 = bbox
    robust_imwrite(image_out, image[y1:y2, x1:x2])

    wrote_mask = False
    if mask_path is not None and mask_out is not None:
        mask = read_mask(mask_path)
        if mask is not None and mask.shape[:2] == image.shape[:2]:
            write_mask(mask_out, mask[y1:y2, x1:x2])
            wrote_mask = True
    return x2 - x1, y2 - y1, wrote_mask


def crop_chip_stage(args: argparse.Namespace) -> None:
    src = args.src.expanduser().resolve()
    out_root = args.out_chip_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else dataset_root_from_image_src(src)
    if args.rebuild:
        clean_dir(out_root)
    else:
        out_root.mkdir(parents=True, exist_ok=True)

    rows = split_items(src)
    if args.limit > 0:
        rows = rows[: args.limit]
    manifest_rows = []
    skipped = processed = failed = 0

    for index, item in enumerate(rows, start=1):
        mask_path = None if args.no_masks else find_mask(dataset_root, item)
        image_out = out_root / "images" / item.split / item.rel
        mask_out = out_root / "masks" / item.split / item.rel.with_suffix(args.mask_ext)
        if image_out.exists() and (mask_path is None or mask_out.exists()) and not args.force:
            skipped += 1
            continue

        image = robust_imread(item.path, cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            manifest_rows.append({"split": item.split, "image": str(item.path), "status": "read_failed"})
            continue
        bbox = locate_srs_chip(
            image,
            args.saturation_threshold,
            args.value_threshold,
            args.max_side,
            args.column_min_fraction,
            args.row_min_fraction,
            args.chip_margin,
        )
        try:
            crop_w, crop_h, wrote_mask = crop_pair(item.path, mask_path, image_out, mask_out if mask_path else None, bbox)
        except OSError as exc:
            failed += 1
            manifest_rows.append({"split": item.split, "image": str(item.path), "status": str(exc)})
            continue
        processed += 1
        x1, y1, x2, y2 = bbox
        manifest_rows.append(
            {
                "split": item.split,
                "image": str(item.path),
                "mask": str(mask_path) if mask_path else "",
                "status": "ok",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "out_image": str(image_out),
                "out_mask": str(mask_out) if wrote_mask else "",
            }
        )
        if args.save_preview and index <= args.preview_limit:
            preview = draw_bbox(image, bbox, "chip", (255, 255, 0))
            robust_imwrite(out_root / "previews" / item.split / item.rel.with_suffix(".jpg"), preview)
        if index % 50 == 0 or index == len(rows):
            print(f"[chip] {index}/{len(rows)} scanned, {processed} cropped, {skipped} skipped, {failed} failed")

    write_csv(out_root / "chip_manifest.csv", manifest_rows)
    print(f"chip root: {out_root}")
    print(f"processed={processed}, skipped={skipped}, failed={failed}")


def fit_display(image: np.ndarray, max_display: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_display / max(h, w))
    if scale < 1.0:
        return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA), scale
    return image.copy(), 1.0


def choose_first_image(root: Path) -> Path:
    candidates = image_files(root)
    if not candidates:
        raise FileNotFoundError(f"no image found under {root}")
    return candidates[0]


def interactive_roi(image: np.ndarray, max_display: int) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    base, scale = fit_display(image, max_display)
    window = "OMG ROI: drag ROI, use trackbars to fine tune, S/Enter save, Esc cancel"
    box = [w // 4, h // 4, w * 3 // 4, h * 3 // 4]
    drawing = False
    start = (0, 0)

    def sync_trackbars() -> None:
        cv2.setTrackbarPos("x1", window, box[0])
        cv2.setTrackbarPos("y1", window, box[1])
        cv2.setTrackbarPos("x2", window, box[2])
        cv2.setTrackbarPos("y2", window, box[3])

    def redraw() -> None:
        box[0], box[1], box[2], box[3] = clamp_bbox(tuple(box), w, h)
        canvas = base.copy()
        sx1, sy1, sx2, sy2 = [int(round(v * scale)) for v in box]
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"ROI {box[0]},{box[1]},{box[2]},{box[3]}",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)

    def on_trackbar(_value: int) -> None:
        box[0] = cv2.getTrackbarPos("x1", window)
        box[1] = cv2.getTrackbarPos("y1", window)
        box[2] = cv2.getTrackbarPos("x2", window)
        box[3] = cv2.getTrackbarPos("y2", window)
        redraw()

    def to_orig(x: int, y: int) -> tuple[int, int]:
        return max(0, min(w - 1, int(round(x / scale)))), max(0, min(h - 1, int(round(y / scale))))

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal drawing, start
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = to_orig(x, y)
            box[:] = [start[0], start[1], start[0] + 1, start[1] + 1]
            sync_trackbars()
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            ox, oy = to_orig(x, y)
            box[:] = list(clamp_bbox((start[0], start[1], ox, oy), w, h))
            sync_trackbars()
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            ox, oy = to_orig(x, y)
            box[:] = list(clamp_bbox((start[0], start[1], ox, oy), w, h))
            sync_trackbars()
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("x1", window, box[0], max(1, w - 1), on_trackbar)
    cv2.createTrackbar("y1", window, box[1], max(1, h - 1), on_trackbar)
    cv2.createTrackbar("x2", window, box[2], w, on_trackbar)
    cv2.createTrackbar("y2", window, box[3], h, on_trackbar)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, ord("s"), ord("S")):
            roi = clamp_bbox(tuple(box), w, h)
            cv2.destroyWindow(window)
            return roi
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window)
            raise SystemExit("ROI selection cancelled")


def save_roi_config(path: Path, image_path: Path, image_shape: tuple[int, int], roi: tuple[int, int, int, int]) -> None:
    h, w = image_shape
    x1, y1, x2, y2 = roi
    data = {
        "version": 1,
        "reference_image": str(image_path),
        "reference_width": w,
        "reference_height": h,
        "roi_xyxy": [x1, y1, x2, y2],
        "roi_ratio_xyxy": [x1 / w, y1 / h, x2 / w, y2 / h],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_roi_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "roi_ratio_xyxy" not in data:
        raise ValueError(f"invalid ROI config: {path}")
    return data


def roi_stage(args: argparse.Namespace) -> None:
    image_path = args.image.expanduser().resolve() if args.image else choose_first_image(args.chip_root / "images")
    image = robust_imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"failed to read image: {image_path}")
    roi = interactive_roi(image, args.max_display)
    save_roi_config(args.roi_config, image_path, image.shape[:2], roi)
    preview = draw_bbox(image, roi, "roi", (0, 0, 255))
    robust_imwrite(args.roi_config.parent / "roi_preview.jpg", preview)
    print(f"roi xyxy: {roi}")
    print(f"roi config: {args.roi_config}")


def bbox_from_ratio(ratio: Iterable[float], width: int, height: int) -> tuple[int, int, int, int]:
    rx1, ry1, rx2, ry2 = [float(v) for v in ratio]
    return clamp_bbox((rx1 * width, ry1 * height, rx2 * width, ry2 * height), width, height)


def merge_stage(args: argparse.Namespace) -> None:
    chip_root = args.chip_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    roi_config = load_roi_config(args.roi_config.expanduser().resolve())
    if args.rebuild:
        for name in ("images", "masks", "labels", "previews"):
            path = out_root / name
            if path.exists():
                shutil.rmtree(path)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = split_items(chip_root / "images")
    if args.limit > 0:
        rows = rows[: args.limit]
    manifest_rows = []
    processed = failed = 0
    for index, item in enumerate(rows, start=1):
        image = robust_imread(item.path, cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            manifest_rows.append({"split": item.split, "image": str(item.path), "status": "read_failed"})
            continue
        h, w = image.shape[:2]
        roi = bbox_from_ratio(roi_config["roi_ratio_xyxy"], w, h)
        mask_path = None if args.no_masks else find_mask(chip_root, item)

        image_out = out_root / "images" / item.split / item.rel
        mask_out = out_root / "masks" / item.split / item.rel.with_suffix(args.mask_ext)
        try:
            crop_w, crop_h, wrote_mask = crop_pair(item.path, mask_path, image_out, mask_out if mask_path else None, roi)
        except OSError as exc:
            failed += 1
            manifest_rows.append({"split": item.split, "image": str(item.path), "status": str(exc)})
            continue
        processed += 1
        x1, y1, x2, y2 = roi
        manifest_rows.append(
            {
                "split": item.split,
                "image": str(item.path),
                "mask": str(mask_path) if mask_path else "",
                "status": "ok",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "out_image": str(image_out),
                "out_mask": str(mask_out) if wrote_mask else "",
            }
        )
        if args.save_preview and index <= args.preview_limit:
            preview = draw_bbox(image, roi, "roi", (0, 0, 255))
            robust_imwrite(out_root / "previews" / item.split / item.rel.with_suffix(".jpg"), preview)
        if index % 50 == 0 or index == len(rows):
            print(f"[merge] {index}/{len(rows)} scanned, {processed} cropped, {failed} failed")

    write_csv(out_root / "crop_manifest.csv", manifest_rows)
    if not args.no_yolo:
        convert_yolo_stage(args)
    print(f"merge output: {out_root}")
    print(f"processed={processed}, failed={failed}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_class_names(class_file: Path | None, class_mode: str) -> tuple[list[str], dict[int, int]]:
    if class_mode == "single":
        return ["anomaly"], {}
    if class_file is None or not class_file.exists():
        raise FileNotFoundError("multi-class YOLO conversion requires --class-names")
    raw_names = [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if raw_names and raw_names[0] == "_background_":
        names = raw_names[1:]
        value_to_class = {idx: idx - 1 for idx in range(1, len(raw_names))}
    else:
        names = raw_names
        value_to_class = {idx + 1: idx for idx in range(len(names))}
    if not names:
        raise ValueError(f"no foreground class names in {class_file}")
    return names, value_to_class


def load_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    return mask[:, :, 0] if mask.ndim == 3 else mask


def labels_from_mask(mask: np.ndarray, min_area: int, class_mode: str, value_to_class: dict[int, int]) -> list[str]:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    h, w = mask.shape[:2]
    rows = []
    if class_mode == "single":
        class_masks = [(0, (mask > 0).astype(np.uint8))]
    else:
        class_masks = []
        for value in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
            class_id = value_to_class.get(value)
            if class_id is not None:
                class_masks.append((class_id, (mask == value).astype(np.uint8)))

    for class_id, binary in class_masks:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            rows.append(f"{class_id} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return rows


def write_data_yaml(out_root: Path, names: list[str], splits: list[str]) -> None:
    lines = [f"path: {out_root.as_posix()}"]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    lines.extend(["", f"nc: {len(names)}", "names:"])
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_yolo_stage(args: argparse.Namespace) -> None:
    out_root = args.out_root.expanduser().resolve()
    class_file = args.class_names.expanduser().resolve() if args.class_names else DEFAULT_CLASS_NAMES
    names, value_to_class = read_class_names(class_file, args.class_mode)
    mask_root = out_root / "masks"
    labels_done = boxes_done = 0
    used_splits = []
    for split in SPLITS:
        split_root = mask_root / split
        if not split_root.exists():
            continue
        used_splits.append(split)
        for mask_path in sorted(p for p in split_root.rglob("*") if p.is_file() and p.suffix.lower() in MASK_EXTS):
            rows = labels_from_mask(load_mask(mask_path), args.min_area, args.class_mode, value_to_class)
            txt_path = out_root / "labels" / split / mask_path.relative_to(split_root).with_suffix(".txt")
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            labels_done += 1
            boxes_done += len(rows)
    (out_root / "class_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    write_data_yaml(out_root, names, used_splits or list(SPLITS))
    print(f"YOLO labels: {out_root / 'labels'} ({labels_done} masks, {boxes_done} boxes)")
    print(f"YOLO yaml: {out_root / 'data.yaml'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_chip_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--src", type=Path, default=DEFAULT_IMAGE_ROOT, help="image root, dataset root, image dir, or file")
        p.add_argument("--dataset-root", type=Path, default=None, help="root that contains labels/lable/masks")
        p.add_argument("--out-chip-root", type=Path, default=DEFAULT_CHIP_ROOT)
        p.add_argument("--saturation-threshold", type=int, default=75)
        p.add_argument("--value-threshold", type=int, default=35)
        p.add_argument("--max-side", type=int, default=1024)
        p.add_argument("--column-min-fraction", type=float, default=0.05)
        p.add_argument("--row-min-fraction", type=float, default=0.05)
        p.add_argument("--chip-margin", type=int, default=0)
        p.add_argument("--mask-ext", default=".png")
        p.add_argument("--no-masks", action="store_true")
        p.add_argument("--save-preview", action="store_true")
        p.add_argument("--preview-limit", type=int, default=50)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--force", action="store_true", help="recrop files even if chip outputs exist")
        p.add_argument("--rebuild", action="store_true", help="delete chip output folder first")

    p_chip = sub.add_parser("chip", help="crop SRS chip body; skips existing outputs by default")
    add_chip_args(p_chip)

    p_roi = sub.add_parser("roi", help="draw ROI on one cropped chip image and save roi_config.json")
    p_roi.add_argument("--chip-root", type=Path, default=DEFAULT_CHIP_ROOT)
    p_roi.add_argument("--image", type=Path, default=None)
    p_roi.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    p_roi.add_argument("--max-display", type=int, default=1600)

    p_merge = sub.add_parser("merge", help="apply ROI to cropped chip body images and convert masks to YOLO labels")
    p_merge.add_argument("--chip-root", type=Path, default=DEFAULT_CHIP_ROOT)
    p_merge.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p_merge.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    p_merge.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES)
    p_merge.add_argument("--class-mode", choices=("single", "multi"), default="multi")
    p_merge.add_argument("--min-area", type=int, default=1)
    p_merge.add_argument("--mask-ext", default=".png")
    p_merge.add_argument("--no-masks", action="store_true")
    p_merge.add_argument("--no-yolo", action="store_true")
    p_merge.add_argument("--save-preview", action="store_true")
    p_merge.add_argument("--preview-limit", type=int, default=50)
    p_merge.add_argument("--limit", type=int, default=0)
    p_merge.add_argument("--rebuild", action="store_true")

    p_all = sub.add_parser("all", help="default workflow: crop chip, select ROI if needed, apply ROI, write YOLO dataset")
    add_chip_args(p_all)
    p_all.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p_all.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    p_all.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES)
    p_all.add_argument("--class-mode", choices=("single", "multi"), default="multi")
    p_all.add_argument("--min-area", type=int, default=1)
    p_all.add_argument("--max-display", type=int, default=1600)
    p_all.add_argument("--no-yolo", action="store_true")
    p_all.add_argument("--rebuild-final", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "mask_ext") and not args.mask_ext.startswith("."):
        args.mask_ext = "." + args.mask_ext

    if args.command == "chip":
        crop_chip_stage(args)
    elif args.command == "roi":
        roi_stage(args)
    elif args.command == "merge":
        merge_stage(args)
    elif args.command == "all":
        crop_chip_stage(args)
        if not args.roi_config.exists():
            roi_args = argparse.Namespace(
                chip_root=args.out_chip_root,
                image=None,
                roi_config=args.roi_config,
                max_display=args.max_display,
            )
            roi_stage(roi_args)
        merge_args = argparse.Namespace(
            chip_root=args.out_chip_root,
            out_root=args.out_root,
            roi_config=args.roi_config,
            class_names=args.class_names,
            class_mode=args.class_mode,
            min_area=args.min_area,
            mask_ext=args.mask_ext,
            no_masks=args.no_masks,
            no_yolo=args.no_yolo,
            save_preview=args.save_preview,
            preview_limit=args.preview_limit,
            limit=args.limit,
            rebuild=args.rebuild_final,
        )
        merge_stage(merge_args)
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()

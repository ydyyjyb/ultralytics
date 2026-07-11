# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Convert semantic mask labels to YOLO detection label txt files.

This script only handles mask-to-YOLO conversion. It does not copy images or tile datasets.
It supports dataset roots with labels/train, labels/val, labels/test and custom mask files or directories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MASK_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_SPLITS = ("train", "val", "test")


def mask_files(root: Path) -> list[Path]:
    """Return supported mask files below a root."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in MASK_EXTS)


def read_class_names(class_file: Path | None, class_mode: str) -> tuple[list[str], dict[int, int]]:
    """Read class names and map mask pixel values to YOLO class IDs."""
    if class_mode == "single":
        return ["anomaly"], {}
    if class_file is None or not class_file.exists():
        raise FileNotFoundError("--class-mode multi requires --class-names or class_names.txt in --src-root")

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
    """Load one mask image as a 2D array."""
    mask = np.array(Image.open(mask_path))
    return mask[:, :, 0] if mask.ndim == 3 else mask


def labels_from_mask(mask: np.ndarray, min_area: int, class_mode: str, value_to_class: dict[int, int]) -> list[str]:
    """Convert connected components in a mask to YOLO detection labels."""
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


def write_dataset_yaml(out_root: Path, yaml_path: Path, names: list[str], splits: list[str]) -> None:
    """Write a simple Ultralytics dataset YAML that points at images/<split>."""
    lines = [f"path: {out_root.as_posix()}"]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    lines.extend(["", f"nc: {len(names)}", "names:"])
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_split(
    src_root: Path,
    out_root: Path,
    split: str,
    min_area: int,
    class_mode: str,
    value_to_class: dict[int, int],
) -> tuple[int, int]:
    """Convert one split from mask images to YOLO txt labels."""
    src_label_dir = src_root / "labels" / split
    out_label_dir = out_root / "labels" / split
    if not src_label_dir.exists():
        print(f"Skip missing split: {src_label_dir}")
        return 0, 0

    masks_done = boxes_done = 0
    for mask_path in mask_files(src_label_dir):
        label_rows = labels_from_mask(load_mask(mask_path), min_area, class_mode, value_to_class)
        dst_label = out_label_dir / mask_path.relative_to(src_label_dir).with_suffix(".txt")
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
        masks_done += 1
        boxes_done += len(label_rows)
    return masks_done, boxes_done


def convert_mask_source(
    mask_source: Path,
    out_root: Path,
    min_area: int,
    class_mode: str,
    value_to_class: dict[int, int],
    out_label: Path | None,
) -> tuple[int, int]:
    """Convert a custom mask file or mask directory to YOLO txt labels."""
    if mask_source.is_file():
        masks = [mask_source]
    elif mask_source.is_dir():
        masks = mask_files(mask_source)
    else:
        raise FileNotFoundError(f"mask source not found: {mask_source}")

    masks_done = boxes_done = 0
    for mask_path in masks:
        label_rows = labels_from_mask(load_mask(mask_path), min_area, class_mode, value_to_class)
        if out_label is not None:
            if len(masks) != 1:
                raise ValueError("--out-label can only be used when --mask-source is a single file")
            dst_label = out_label
        elif mask_source.is_file():
            dst_label = out_root / f"{mask_path.stem}.txt"
        else:
            dst_label = out_root / mask_path.relative_to(mask_source).with_suffix(".txt")
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
        masks_done += 1
        boxes_done += len(label_rows)
    return masks_done, boxes_done


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=None, help="Dataset root containing labels/<split> masks.")
    parser.add_argument("--mask-source", type=Path, default=None, help="Custom mask file or mask directory to convert.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root. Defaults to --src-root in dataset mode or --mask-source location in custom mode.",
    )
    parser.add_argument("--out-label", type=Path, default=None, help="Output txt path for a single --mask-source file.")
    parser.add_argument(
        "--class-names", type=Path, default=None, help="Class names txt. Defaults to src-root/class_names.txt."
    )
    parser.add_argument("--class-mode", choices=("single", "multi"), default="multi")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Dataset label splits to convert.")
    parser.add_argument("--min-area", type=int, default=4, help="Minimum connected-component area in mask pixels.")
    parser.add_argument("--yaml-name", default="data.yaml", help="Dataset YAML name written under out-root.")
    parser.add_argument("--no-yaml", action="store_true", help="Do not write dataset YAML.")
    return parser.parse_args()


def main() -> None:
    """Convert masks to YOLO labels."""
    args = parse_args()
    if args.src_root is None and args.mask_source is None:
        raise ValueError("provide either --src-root or --mask-source")
    if args.src_root is not None and args.mask_source is not None:
        raise ValueError("--src-root and --mask-source are mutually exclusive")

    src_root = args.src_root.expanduser().resolve() if args.src_root else None
    mask_source = args.mask_source.expanduser().resolve() if args.mask_source else None
    if mask_source is not None and not (mask_source.is_file() or mask_source.is_dir()):
        raise FileNotFoundError(f"mask source not found: {mask_source}")
    if args.out_root is not None:
        out_root = args.out_root.expanduser().resolve()
    elif src_root is not None:
        out_root = src_root
    elif mask_source and mask_source.is_file():
        out_root = mask_source.parent
    else:
        out_root = mask_source

    class_file = args.class_names.expanduser().resolve() if args.class_names else None
    if class_file is None and src_root is not None:
        class_file = src_root / "class_names.txt"
    names, value_to_class = read_class_names(class_file, args.class_mode)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "class_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    if mask_source is not None:
        out_label = args.out_label.expanduser().resolve() if args.out_label else None
        masks, boxes = convert_mask_source(
            mask_source, out_root, args.min_area, args.class_mode, value_to_class, out_label
        )
        print(f"custom: {masks} masks, {boxes} boxes")
        print(f"labels: {out_label.parent if out_label else out_root}")
        return

    for split in args.splits:
        masks, boxes = convert_split(src_root, out_root, split, args.min_area, args.class_mode, value_to_class)
        print(f"{split}: {masks} masks, {boxes} boxes")

    if not args.no_yaml:
        yaml_path = out_root / args.yaml_name
        write_dataset_yaml(out_root, yaml_path, names, args.splits)
        print(f"yaml: {yaml_path}")
    print(f"labels: {out_root / 'labels'}")


if __name__ == "__main__":
    main()

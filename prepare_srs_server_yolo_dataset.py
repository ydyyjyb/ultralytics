# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Build a 2x2 original-image SRS YOLO detection dataset.

Defaults match the server layout:

    python prepare_srs_server_yolo_dataset.py --out /home/vina04/srs_yolo_2x2 --rebuild

The train root must contain masks. The test root is image-only and is written as an unlabeled test split.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
DEFAULT_TRAIN_ROOT = Path("/home/vina04/SRS+100+cropped")
DEFAULT_TEST_ROOT = Path("/home/vina04/SRS0708")


def resolve_path(root: Path, value: str) -> Path:
    """Resolve a dataset manifest entry relative to the dataset root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def image_files(root: Path) -> list[Path]:
    """Return supported image files below a root."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_mask(dataset: Path, split: str, image_path: Path) -> Path:
    """Find a mask for one source image in common SRS layouts."""
    candidates = []
    for ext in MASK_EXTS:
        candidates.extend(
            [
                dataset / "labels" / split / f"{image_path.stem}{ext}",
                dataset / "labels" / split / image_path.name,
                dataset / "masks" / split / f"{image_path.stem}{ext}",
                dataset / "masks" / split / image_path.name,
                dataset / "labels" / f"{image_path.stem}{ext}",
                dataset / "labels" / image_path.name,
                dataset / "masks" / f"{image_path.stem}{ext}",
                dataset / "masks" / image_path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot find mask for {split}/{image_path.name}")


def iter_labeled_items(dataset: Path, split: str) -> list[tuple[Path, Path]]:
    """Read labeled items from split txt files or images/<split> plus labels/<split>."""
    split_file = dataset / f"{split}.txt"
    items: list[tuple[Path, Path]] = []
    if split_file.exists():
        for raw in split_file.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            parts = raw.split()
            image_path = resolve_path(dataset, parts[0])
            mask_path = resolve_path(dataset, parts[1]) if len(parts) > 1 else find_mask(dataset, split, image_path)
            items.append((image_path, mask_path))
        return items

    image_dir = dataset / "images" / split
    if not image_dir.exists():
        return items
    return [(image_path, find_mask(dataset, split, image_path)) for image_path in image_files(image_dir)]


def iter_test_images(dataset: Path) -> list[Path]:
    """Read image-only test files from test.txt, images/test, images, or the root."""
    split_file = dataset / "test.txt"
    if split_file.exists():
        return [
            resolve_path(dataset, raw.split()[0])
            for raw in split_file.read_text(encoding="utf-8").splitlines()
            if raw.strip()
        ]

    for candidate in (dataset / "images" / "test", dataset / "images", dataset):
        if candidate.exists():
            images = image_files(candidate)
            if images:
                return [
                    p
                    for p in images
                    if {"label", "labels", "mask", "masks"}.isdisjoint({part.lower() for part in p.parts})
                ]
    return []


def read_class_names(dataset: Path, class_mode: str) -> tuple[list[str], dict[int, int]]:
    """Read class names and map mask pixel values to YOLO class IDs."""
    if class_mode == "single":
        return ["anomaly"], {}

    class_file = dataset / "class_names.txt"
    if not class_file.exists():
        raise FileNotFoundError(f"multi-class mode requires {class_file}")

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


def load_mask(mask_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    """Load a mask and resize it to match the image if needed."""
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    h, w = image_shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def labels_from_mask(mask: np.ndarray, min_area: int, class_mode: str, value_to_class: dict[int, int]) -> list[str]:
    """Convert connected components in one mask tile to YOLO detection labels."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    h, w = mask.shape[:2]
    rows = []
    if class_mode == "single":
        masks = [(0, (mask > 0).astype(np.uint8))]
    else:
        masks = []
        for value in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
            class_id = value_to_class.get(value)
            if class_id is not None:
                masks.append((class_id, (mask == value).astype(np.uint8)))

    for class_id, binary in masks:
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


def tile_ranges(length: int, tiles: int) -> list[tuple[int, int]]:
    """Split one dimension into fixed non-overlapping tiles."""
    ranges = []
    for idx in range(tiles):
        start = round(idx * length / tiles)
        end = round((idx + 1) * length / tiles)
        ranges.append((int(start), int(end)))
    return ranges


def convert_image_mode(image: np.ndarray, mode: str) -> np.ndarray:
    """Apply optional image preprocessing before writing tiles."""
    if mode == "rgb":
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "gray":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unknown image mode: {mode}")


def output_stem(image_path: Path, source_root: Path) -> str:
    """Build a collision-resistant stem from a source image path."""
    try:
        return "__".join(image_path.relative_to(source_root).with_suffix("").parts)
    except ValueError:
        return image_path.stem


def write_yaml(out: Path, name: str, names: list[str]) -> Path:
    """Write an Ultralytics dataset YAML."""
    yaml_path = out / f"{name}.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out.as_posix()}",
                f"train: {(out / 'train_images.txt').as_posix()}",
                f"val: {(out / 'val_images.txt').as_posix()}",
                f"test: {(out / 'test_images.txt').as_posix()}",
                "",
                f"nc: {len(names)}",
                "names:",
                *[f"  {idx}: {class_name}" for idx, class_name in enumerate(names)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def process_labeled_split(
    args: argparse.Namespace,
    split: str,
    value_to_class: dict[int, int],
    manifest_rows: list[dict],
) -> tuple[list[str], dict[str, int]]:
    """Write one labeled split after original-image 2x2 tiling."""
    image_list = []
    summary = {"images": 0, "boxes": 0}
    x_ranges: list[tuple[int, int]]
    y_ranges: list[tuple[int, int]]

    for image_path, mask_path in iter_labeled_items(args.train_root, split):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] failed to read image: {image_path}")
            continue
        mask = load_mask(mask_path, image.shape[:2])
        h, w = image.shape[:2]
        x_ranges = tile_ranges(w, args.tiles_x)
        y_ranges = tile_ranges(h, args.tiles_y)
        stem = output_stem(image_path, args.train_root)

        for tile_y, (y1, y2) in enumerate(y_ranges):
            for tile_x, (x1, x2) in enumerate(x_ranges):
                tile_idx = tile_y * args.tiles_x + tile_x
                tile_name = f"{stem}__tile{tile_idx}{image_path.suffix}"
                tile_stem = Path(tile_name).stem
                dst_image = args.out / "images" / split / tile_name
                dst_label = args.out / "labels" / split / f"{tile_stem}.txt"
                dst_image.parent.mkdir(parents=True, exist_ok=True)
                dst_label.parent.mkdir(parents=True, exist_ok=True)

                tile_image = image[y1:y2, x1:x2]
                tile_mask = mask[y1:y2, x1:x2]
                if not cv2.imwrite(str(dst_image), convert_image_mode(tile_image, args.image_mode)):
                    raise OSError(f"failed to write image: {dst_image}")

                label_rows = labels_from_mask(tile_mask, args.min_area, args.class_mode, value_to_class)
                dst_label.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
                image_list.append(dst_image.absolute().as_posix())
                summary["images"] += 1
                summary["boxes"] += len(label_rows)
                manifest_rows.append(
                    {
                        "split": split,
                        "image": dst_image.absolute().as_posix(),
                        "source_image": image_path.absolute().as_posix(),
                        "source_mask": mask_path.absolute().as_posix(),
                        "x": x1,
                        "y": y1,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "source_w": w,
                        "source_h": h,
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "boxes": len(label_rows),
                        "has_labels": 1,
                    }
                )
    return image_list, summary


def process_test_split(args: argparse.Namespace, manifest_rows: list[dict]) -> tuple[list[str], dict[str, int]]:
    """Write image-only test split after original-image 2x2 tiling."""
    image_list = []
    summary = {"images": 0, "boxes": 0}

    for image_path in iter_test_images(args.test_root):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] failed to read test image: {image_path}")
            continue
        h, w = image.shape[:2]
        x_ranges = tile_ranges(w, args.tiles_x)
        y_ranges = tile_ranges(h, args.tiles_y)
        stem = output_stem(image_path, args.test_root)

        for tile_y, (y1, y2) in enumerate(y_ranges):
            for tile_x, (x1, x2) in enumerate(x_ranges):
                tile_idx = tile_y * args.tiles_x + tile_x
                tile_name = f"{stem}__tile{tile_idx}{image_path.suffix}"
                tile_stem = Path(tile_name).stem
                dst_image = args.out / "images" / "test" / tile_name
                dst_label = args.out / "labels" / "test" / f"{tile_stem}.txt"
                dst_image.parent.mkdir(parents=True, exist_ok=True)

                tile_image = image[y1:y2, x1:x2]
                if not cv2.imwrite(str(dst_image), convert_image_mode(tile_image, args.image_mode)):
                    raise OSError(f"failed to write image: {dst_image}")
                if args.write_empty_test_labels:
                    dst_label.parent.mkdir(parents=True, exist_ok=True)
                    dst_label.write_text("", encoding="utf-8")

                image_list.append(dst_image.absolute().as_posix())
                summary["images"] += 1
                manifest_rows.append(
                    {
                        "split": "test",
                        "image": dst_image.absolute().as_posix(),
                        "source_image": image_path.absolute().as_posix(),
                        "source_mask": "",
                        "x": x1,
                        "y": y1,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "source_w": w,
                        "source_h": h,
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "boxes": 0,
                        "has_labels": 0,
                    }
                )
    return image_list, summary


def write_image_list(path: Path, image_list: list[str]) -> None:
    """Write a YOLO image-list txt file."""
    path.write_text("\n".join(image_list) + ("\n" if image_list else ""), encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> None:
    """Build the 2x2 YOLO dataset."""
    if args.rebuild and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    names, value_to_class = read_class_names(args.train_root, args.class_mode)
    (args.out / "class_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    manifest_rows: list[dict] = []
    summaries: dict[str, dict[str, int]] = {}
    split_lists = {"train": [], "val": [], "test": []}

    for split in ("train", "val"):
        split_lists[split], summaries[split] = process_labeled_split(args, split, value_to_class, manifest_rows)
        write_image_list(args.out / f"{split}_images.txt", split_lists[split])

    split_lists["test"], summaries["test"] = process_test_split(args, manifest_rows)
    write_image_list(args.out / "test_images.txt", split_lists["test"])

    yaml_path = write_yaml(args.out, args.name, names)
    manifest_path = args.out / "tile_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "image",
            "source_image",
            "source_mask",
            "x",
            "y",
            "w",
            "h",
            "source_w",
            "source_h",
            "tile_x",
            "tile_y",
            "boxes",
            "has_labels",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (args.out / ".complete").write_text("ok\n", encoding="utf-8")

    print(f"train root: {args.train_root}")
    print(f"test root: {args.test_root}")
    print(f"out: {args.out}")
    print(f"class_mode={args.class_mode} image_mode={args.image_mode} tiles={args.tiles_x}x{args.tiles_y}")
    for split in ("train", "val", "test"):
        print(f"{split}: {summaries[split]['images']} images, {summaries[split]['boxes']} boxes")
    if not args.write_empty_test_labels:
        print("test labels: skipped because /home/vina04/SRS0708 has images only")
    print(f"yaml: {yaml_path}")
    print(f"manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root", type=Path, default=DEFAULT_TRAIN_ROOT, help="Mask-labeled SRS train dataset root."
    )
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT, help="Image-only SRS test dataset root.")
    parser.add_argument("--out", type=Path, required=True, help="Output YOLO dataset root.")
    parser.add_argument("--name", default="srs_original_2x2", help="Output YAML filename stem.")
    parser.add_argument("--class-mode", choices=("single", "multi"), default="multi")
    parser.add_argument("--image-mode", choices=("rgb", "gray", "clahe"), default="rgb")
    parser.add_argument("--min-area", type=int, default=4, help="Minimum connected-component area in mask pixels.")
    parser.add_argument("--tiles-x", type=int, default=2)
    parser.add_argument("--tiles-y", type=int, default=2)
    parser.add_argument(
        "--write-empty-test-labels",
        action="store_true",
        help="Write empty labels/test/*.txt files for image-only test samples.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Delete output before rebuilding.")
    args = parser.parse_args()

    args.train_root = args.train_root.expanduser().resolve()
    args.test_root = args.test_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.tiles_x = max(1, args.tiles_x)
    args.tiles_y = max(1, args.tiles_y)
    return args


def main() -> None:
    """CLI entrypoint."""
    build_dataset(parse_args())


if __name__ == "__main__":
    main()

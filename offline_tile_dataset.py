# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import yaml

IMG_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def progress_iter(items: list[Path], desc: str, enabled: bool = True):
    """Iterate with tqdm when available, otherwise show a lightweight text progress bar."""
    if not enabled:
        yield from items
        return
    try:
        from tqdm import tqdm

        yield from tqdm(items, desc=desc, unit="img")
        return
    except Exception:
        total = len(items)
        width = 28
        for i, item in enumerate(items, 1):
            filled = int(width * i / max(total, 1))
            bar = "#" * filled + "-" * (width - filled)
            print(f"\r{desc} [{bar}] {i}/{total}", end="", flush=True)
            yield item
        print()


def progress_futures(futures: list, desc: str, enabled: bool = True):
    """Iterate completed futures with tqdm when available, otherwise show a text progress bar."""
    completed = as_completed(futures)
    if not enabled:
        yield from completed
        return
    try:
        from tqdm import tqdm

        yield from tqdm(completed, total=len(futures), desc=desc, unit="img")
        return
    except Exception:
        total = len(futures)
        width = 28
        for i, future in enumerate(completed, 1):
            filled = int(width * i / max(total, 1))
            bar = "#" * filled + "-" * (width - filled)
            print(f"\r{desc} [{bar}] {i}/{total}", end="", flush=True)
            yield future
        print()


def parse_tile_size(value: str) -> tuple[int, int]:
    """Parse tile grid as N, NxM, N,M, or [N,M]."""
    text = value.strip().lower().replace("[", "").replace("]", "").replace(" ", "")
    parts = text.split("x") if "x" in text else text.split(",")
    if len(parts) == 1:
        rows = cols = int(parts[0])
    elif len(parts) == 2:
        rows, cols = (int(x) for x in parts)
    else:
        raise argparse.ArgumentTypeError(f"invalid tile size: {value}")
    if rows < 1 or cols < 1:
        raise argparse.ArgumentTypeError("tile row and column counts must be >= 1")
    return rows, cols


def axis_windows(length: int, count: int, overlap: float) -> list[tuple[int, int]]:
    """Return exactly count windows along one image axis."""
    if count == 1:
        return [(0, length)]
    window = min(length, max(round(length / (count - (count - 1) * overlap)), 1))
    step = max(round(window * (1.0 - overlap)), 1)
    starts = [min(step * i, max(length - window, 0)) for i in range(count)]
    starts[-1] = max(length - window, 0)
    return [(start, min(start + window, length)) for start in starts]


def make_windows(h: int, w: int, tile_size: tuple[int, int], overlap: float) -> list[tuple[int, int, int, int]]:
    """Create row-major sliding windows for an image."""
    rows, cols = tile_size
    y_windows = axis_windows(h, rows, overlap)
    x_windows = axis_windows(w, cols, overlap)
    return [(x1, y1, x2, y2) for y1, y2 in y_windows for x1, x2 in x_windows]


def load_yolo_labels(label_path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO detect labels and convert normalized xywh boxes to absolute xyxy."""
    labels = []
    if not label_path.exists():
        return labels

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        xc, yc, bw, bh = (float(x) for x in parts[1:5])
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        labels.append((cls, x1, y1, x2, y2))
    return labels


def remap_labels_to_tile(
    labels: list[tuple[int, float, float, float, float]], window: tuple[int, int, int, int], tile_min_area: float
) -> tuple[list[str], list[tuple[int, float, float, float, float]]]:
    """Return YOLO tile labels and clipped boxes in original-image coordinates for visualization."""
    x1w, y1w, x2w, y2w = window
    tw, th = x2w - x1w, y2w - y1w
    tile_labels = []
    clipped_boxes = []

    for cls, x1, y1, x2, y2 in labels:
        orig_area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        if orig_area <= 0:
            continue

        cx1, cy1 = max(x1, x1w), max(y1, y1w)
        cx2, cy2 = min(x2, x2w), min(y2, y2w)
        new_area = max(cx2 - cx1, 0.0) * max(cy2 - cy1, 0.0)
        if new_area <= 0 or new_area / orig_area < tile_min_area:
            continue

        tx1, ty1, tx2, ty2 = cx1 - x1w, cy1 - y1w, cx2 - x1w, cy2 - y1w
        xc = ((tx1 + tx2) / 2) / tw
        yc = ((ty1 + ty2) / 2) / th
        bw = (tx2 - tx1) / tw
        bh = (ty2 - ty1) / th
        tile_labels.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        clipped_boxes.append((cls, cx1, cy1, cx2, cy2))

    return tile_labels, clipped_boxes


def draw_tile_visualization(
    image,
    windows: list[tuple[int, int, int, int]],
    boxes_by_tile: list[list[tuple[int, float, float, float, float]]],
    output_path: Path,
    max_side: int = 0,
) -> None:
    """Draw tile windows and clipped tile labels on the source image."""
    canvas = image.copy()
    palette = [
        (255, 80, 80),
        (80, 180, 255),
        (80, 220, 120),
        (240, 180, 70),
        (180, 100, 255),
        (80, 240, 240),
    ]

    for idx, (window, boxes) in enumerate(zip(windows, boxes_by_tile)):
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = window
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        cv2.putText(canvas, str(idx), (x1 + 6, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        for cls, bx1, by1, bx2, by2 in boxes:
            p1 = int(round(bx1)), int(round(by1))
            p2 = int(round(bx2)), int(round(by2))
            cv2.rectangle(canvas, p1, p2, color, 2)
            cv2.putText(canvas, str(cls), (p1[0], max(p1[1] - 4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if max_side and max(canvas.shape[:2]) > max_side:
        scale = max_side / max(canvas.shape[:2])
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def output_stem(image_path: Path, image_dir: Path) -> str:
    """Build a collision-resistant output stem from a path relative to image_dir."""
    return "__".join(image_path.relative_to(image_dir).with_suffix("").parts)


def process_image(
    image_path: Path,
    image_dir: Path,
    label_dir: Path,
    out_image_dir: Path,
    out_label_dir: Path,
    vis_dir: Path,
    tile_size: tuple[int, int],
    overlap: float,
    tile_min_area: float,
    save_empty: bool,
    save_vis: bool,
    vis_max_side: int,
) -> tuple[int, int, int, int]:
    """Process one image and return image, tile, label, visualization counts."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Skip unreadable image: {image_path}")
        return 0, 0, 0, 0

    h, w = img.shape[:2]
    rel_label = image_path.relative_to(image_dir).with_suffix(".txt")
    labels = load_yolo_labels(label_dir / rel_label, w, h)
    windows = make_windows(h, w, tile_size, overlap)
    boxes_by_tile = []
    tiles_saved = labels_saved = 0

    stem = output_stem(image_path, image_dir)
    for idx, window in enumerate(windows):
        x1, y1, x2, y2 = window
        tile_img = img[y1:y2, x1:x2]
        tile_labels, clipped_boxes = remap_labels_to_tile(labels, window, tile_min_area)
        boxes_by_tile.append(clipped_boxes)
        if not tile_labels and not save_empty:
            continue

        tile_name = f"{stem}__tile_{idx:04d}__x{x1}_y{y1}_w{x2 - x1}_h{y2 - y1}"
        cv2.imwrite(str(out_image_dir / f"{tile_name}{image_path.suffix}"), tile_img)
        label_text = "\n".join(tile_labels) + ("\n" if tile_labels else "")
        (out_label_dir / f"{tile_name}.txt").write_text(label_text, encoding="utf-8")
        tiles_saved += 1
        labels_saved += len(tile_labels)

    visualizations_saved = 0
    if save_vis:
        draw_tile_visualization(img, windows, boxes_by_tile, vis_dir / f"{stem}_tiles.jpg", vis_max_side)
        visualizations_saved = 1

    return 1, tiles_saved, labels_saved, visualizations_saved


def tile_split(
    image_dir: Path,
    label_dir: Path,
    out_image_dir: Path,
    out_label_dir: Path,
    vis_dir: Path,
    tile_size: tuple[int, int],
    overlap: float,
    tile_min_area: float,
    save_empty: bool = True,
    save_vis: bool = True,
    vis_max_side: int = 0,
    show_progress: bool = True,
    workers: int = 1,
) -> None:
    """Split one dataset split and optionally save original-image visualizations."""
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    image_files = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    images_done = tiles_saved = labels_saved = visualizations_saved = 0
    desc = f"tiling {image_dir.name}"
    workers = max(int(workers), 1)

    def add_counts(counts: tuple[int, int, int, int]) -> None:
        nonlocal images_done, tiles_saved, labels_saved, visualizations_saved
        images_done += counts[0]
        tiles_saved += counts[1]
        labels_saved += counts[2]
        visualizations_saved += counts[3]

    if workers == 1:
        for image_path in progress_iter(image_files, desc=desc, enabled=show_progress):
            add_counts(
                process_image(
                    image_path,
                    image_dir,
                    label_dir,
                    out_image_dir,
                    out_label_dir,
                    vis_dir,
                    tile_size,
                    overlap,
                    tile_min_area,
                    save_empty,
                    save_vis,
                    vis_max_side,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    process_image,
                    image_path,
                    image_dir,
                    label_dir,
                    out_image_dir,
                    out_label_dir,
                    vis_dir,
                    tile_size,
                    overlap,
                    tile_min_area,
                    save_empty,
                    save_vis,
                    vis_max_side,
                )
                for image_path in image_files
            ]
            for future in progress_futures(futures, desc=f"{desc} ({workers} threads)", enabled=show_progress):
                add_counts(future.result())

    print(
        f"{image_dir.name}: {images_done}/{len(image_files)} images, {tiles_saved} tiles, "
        f"{labels_saved} labels, {visualizations_saved} visualizations"
    )


def build_offline_tiled_dataset(
    src_root: Path,
    dst_root: Path,
    tile_size: tuple[int, int],
    overlap: float,
    tile_min_area: float,
    save_empty: bool,
    save_vis: bool,
    vis_max_side: int,
    show_progress: bool,
    workers: int,
) -> None:
    """Build a tiled YOLO dataset from images/train|val and labels/train|val."""
    for split in ("train", "val"):
        image_dir = src_root / "images" / split
        label_dir = src_root / "labels" / split
        if not image_dir.exists():
            print(f"Skip missing split: {image_dir}")
            continue
        tile_split(
            image_dir=image_dir,
            label_dir=label_dir,
            out_image_dir=dst_root / "images" / split,
            out_label_dir=dst_root / "labels" / split,
            vis_dir=dst_root / "vis_tiles" / split,
            tile_size=tile_size,
            overlap=overlap,
            tile_min_area=tile_min_area,
            save_empty=save_empty,
            save_vis=save_vis,
            vis_max_side=vis_max_side,
            show_progress=show_progress,
            workers=workers,
        )


def write_dataset_yaml(src_yaml: Path, dst_yaml: Path, dst_root: Path) -> None:
    """Copy dataset YAML metadata while pointing path to the tiled dataset root."""
    with open(src_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = str(dst_root.resolve())
    with open(dst_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create an offline tiled YOLO dataset with source-image visualizations."
    )
    parser.add_argument("--src-root", required=True, type=Path, help="Source dataset root with images/ and labels/.")
    parser.add_argument("--src-yaml", required=True, type=Path, help="Source dataset YAML.")
    parser.add_argument("--dst-root", required=True, type=Path, help="Output tiled dataset root.")
    parser.add_argument("--dst-yaml", required=True, type=Path, help="Output tiled dataset YAML.")
    parser.add_argument("--tile-size", default="2,3", type=parse_tile_size, help="Grid count as N, NxM, or N,M.")
    parser.add_argument("--tile-overlap", default=0.2, type=float, help="Fractional overlap between neighboring tiles.")
    parser.add_argument("--tile-min-area", default=0.1, type=float, help="Minimum retained object area fraction.")
    parser.add_argument("--no-empty", action="store_true", help="Do not save empty/background tiles.")
    parser.add_argument("--no-vis", action="store_true", help="Do not save source-image tile visualizations.")
    parser.add_argument("--no-progress", action="store_true", help="Disable processing progress bars.")
    parser.add_argument("--vis-max-side", default=0, type=int, help="Resize visualization max side; 0 keeps original.")
    parser.add_argument("--workers", default=min(os.cpu_count() or 1, 8), type=int, help="Image processing threads.")
    return parser.parse_args()


def main() -> None:
    """Create the tiled dataset and visualization images."""
    args = parse_args()
    if not 0.0 <= args.tile_overlap < 1.0:
        raise ValueError("--tile-overlap must be in [0.0, 1.0).")
    if not 0.0 <= args.tile_min_area <= 1.0:
        raise ValueError("--tile-min-area must be in [0.0, 1.0].")

    build_offline_tiled_dataset(
        src_root=args.src_root,
        dst_root=args.dst_root,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        tile_min_area=args.tile_min_area,
        save_empty=not args.no_empty,
        save_vis=not args.no_vis,
        vis_max_side=args.vis_max_side,
        show_progress=not args.no_progress,
        workers=args.workers,
    )
    write_dataset_yaml(args.src_yaml, args.dst_yaml, args.dst_root)
    print(f"Done. Tiled dataset YAML: {args.dst_yaml}")
    if not args.no_vis:
        print(f"Visualization images: {args.dst_root / 'vis_tiles'}")


if __name__ == "__main__":
    main()

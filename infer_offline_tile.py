# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


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


def find_images(source: Path) -> list[Path]:
    """Return image files from a single image path or a directory."""
    if source.is_file():
        return [source] if source.suffix.lower() in IMG_EXTS else []
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)


def output_stem(image_path: Path, source: Path) -> str:
    """Build a collision-resistant output stem."""
    if source.is_file():
        return image_path.stem
    return "__".join(image_path.relative_to(source).with_suffix("").parts)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calculate IoU between one xyxy box and many xyxy boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    area1 = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    area2 = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    return inter / np.maximum(area1 + area2 - inter, 1e-12)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float, agnostic: bool) -> np.ndarray:
    """Apply score-sorted NMS after tile predictions are mapped back to the original image."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    keep = []
    groups = [None] if agnostic else np.unique(classes)
    for cls in groups:
        inds = np.arange(len(boxes)) if cls is None else np.where(classes == cls)[0]
        order = inds[np.argsort(scores[inds])[::-1]]
        while len(order):
            current = order[0]
            keep.append(current)
            if len(order) == 1:
                break
            remaining = order[1:]
            order = remaining[box_iou(boxes[current], boxes[remaining]) <= iou_thr]

    keep = np.array(keep, dtype=np.int64)
    return keep[np.argsort(scores[keep])[::-1]]


def write_yolo_txt(
    label_path: Path,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    image_shape: tuple[int, int],
    save_conf: bool,
) -> None:
    """Save merged predictions in YOLO normalized xywh format."""
    h, w = image_shape
    lines = []
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        if save_conf:
            lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {float(score):.6f}")
        else:
            lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_predictions(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    names: dict[int, str] | list[str],
    output_path: Path,
    line_width: int,
) -> None:
    """Draw merged full-image predictions."""
    canvas = image.copy()
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(round(x)) for x in box]
        color = ((37 * int(cls) + 60) % 255, (17 * int(cls) + 180) % 255, (29 * int(cls) + 100) % 255)
        name = names[int(cls)] if isinstance(names, list) else names.get(int(cls), str(int(cls)))
        label = f"{name} {score:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, max(line_width - 1, 1))
        y_text = max(y1 - th - 6, 0)
        cv2.rectangle(canvas, (x1, y_text), (x1 + tw + 4, y_text + th + 6), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, y_text + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            max(line_width - 1, 1),
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def save_tile_debug(
    tiles: list[np.ndarray],
    windows: list[tuple[int, int, int, int]],
    image_path: Path,
    source: Path,
    out_dir: Path,
) -> None:
    """Save tile crops for debugging."""
    stem = output_stem(image_path, source)
    tile_dir = out_dir / stem
    tile_dir.mkdir(parents=True, exist_ok=True)
    for i, (tile, (x1, y1, x2, y2)) in enumerate(zip(tiles, windows)):
        name = f"{stem}__tile_{i:04d}__x{x1}_y{y1}_w{x2 - x1}_h{y2 - y1}{image_path.suffix}"
        cv2.imwrite(str(tile_dir / name), tile)


def predict_one_image(model, image_path: Path, source: Path, args: argparse.Namespace) -> tuple[int, int]:
    """Run tiled inference on one image and save merged results."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skip unreadable image: {image_path}")
        return 0, 0

    h, w = image.shape[:2]
    windows = make_windows(h, w, args.tile_size, args.tile_overlap)
    tiles = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in windows]
    if args.save_tiles:
        save_tile_debug(tiles, windows, image_path, source, args.out_dir / "tiles")

    boxes_all, scores_all, classes_all = [], [], []
    for start in range(0, len(tiles), args.batch):
        batch_tiles = tiles[start : start + args.batch]
        batch_windows = windows[start : start + args.batch]
        results = model.predict(
            batch_tiles,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.tile_iou,
            device=args.device,
            half=args.half,
            max_det=args.max_det,
            classes=args.classes,
            verbose=False,
        )
        for result, (x1, y1, x2, y2) in zip(results, batch_windows):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.cpu().numpy().astype(np.float32)
            classes = result.boxes.cls.cpu().numpy().astype(np.int64)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] + x1, 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] + y1, 0, h)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes_all.append(boxes[valid])
            scores_all.append(scores[valid])
            classes_all.append(classes[valid])

    if boxes_all:
        boxes = np.concatenate(boxes_all, axis=0)
        scores = np.concatenate(scores_all, axis=0)
        classes = np.concatenate(classes_all, axis=0)
        keep = nms(boxes, scores, classes, args.merge_iou, args.agnostic_nms)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        classes = np.zeros(0, dtype=np.int64)

    stem = output_stem(image_path, source)
    if args.save_txt:
        write_yolo_txt(args.out_dir / "labels" / f"{stem}.txt", boxes, scores, classes, (h, w), args.save_conf)
    if args.save_vis:
        draw_predictions(
            image, boxes, scores, classes, model.names, args.out_dir / "images" / f"{stem}.jpg", args.line_width
        )
    return 1, len(boxes)


def progress_iter(items: list[Path], enabled: bool = True):
    """Iterate with tqdm when available, otherwise show a lightweight text progress bar."""
    if not enabled:
        yield from items
        return
    try:
        from tqdm import tqdm

        yield from tqdm(items, desc="tile predict", unit="img")
        return
    except Exception:
        total = len(items)
        width = 28
        for i, item in enumerate(items, 1):
            filled = int(width * i / max(total, 1))
            bar = "#" * filled + "-" * (width - filled)
            print(f"\rtile predict [{bar}] {i}/{total}", end="", flush=True)
            yield item
        print()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Offline tiled YOLO detection inference with full-image NMS merge.")
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to YOLO weights, e.g. runs/detect/train/weights/best.pt"
    )
    parser.add_argument("--source", type=Path, required=True, help="Image file or image directory.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/tile_predict"), help="Directory to save merged outputs."
    )
    parser.add_argument(
        "--tile-size", type=parse_tile_size, default=(2, 2), help="Tile grid, e.g. 2, 2,3, 2x3, or [2,3]."
    )
    parser.add_argument("--tile-overlap", type=float, default=0.2, help="Overlap ratio for adjacent tile windows.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size for each tile.")
    parser.add_argument("--conf", type=float, default=0.25, help="Tile-level confidence threshold.")
    parser.add_argument("--tile-iou", type=float, default=0.7, help="YOLO NMS IoU threshold inside each tile.")
    parser.add_argument(
        "--merge-iou", type=float, default=0.5, help="Final full-image NMS IoU threshold after merging tiles."
    )
    parser.add_argument("--batch", type=int, default=16, help="Number of tiles predicted per YOLO call.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0, cpu, cuda:0.")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Optional class IDs to keep.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per tile before full-image merge.")
    parser.add_argument("--line-width", type=int, default=2, help="Line width for visualization.")
    parser.add_argument("--save-txt", action="store_true", help="Save merged YOLO txt predictions.")
    parser.add_argument("--save-conf", action="store_true", help="Append confidence to saved txt predictions.")
    parser.add_argument("--save-vis", action="store_true", help="Save merged visualization images.")
    parser.add_argument("--save-tiles", action="store_true", help="Save tile crops for debugging.")
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="Use class-agnostic NMS when merging tile predictions."
    )
    parser.add_argument("--half", action="store_true", help="Use FP16 inference when supported by the selected device.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display.")
    args = parser.parse_args()
    if not 0.0 <= args.tile_overlap < 1.0:
        raise ValueError("--tile-overlap must be in [0.0, 1.0).")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1.")
    if not args.save_txt and not args.save_vis and not args.save_tiles:
        args.save_txt = True
        args.save_vis = True
    return args


def main() -> None:
    """Run offline tiled inference."""
    args = parse_args()
    source = args.source.resolve()
    image_files = find_images(source)
    if not image_files:
        raise FileNotFoundError(f"No supported images found in {source}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(args.weights)
    images_done = detections_saved = 0
    for image_path in progress_iter(image_files, enabled=not args.no_progress):
        done, detections = predict_one_image(model, image_path, source, args)
        images_done += done
        detections_saved += detections
    print(f"Done: {images_done} images, {detections_saved} merged detections saved to {args.out_dir}")


if __name__ == "__main__":
    main()

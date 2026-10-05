"""Live red-bin empty check: RealSense + YOLOE, count items still inside."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bin_overlay import draw
from src.bin_roi import BinView, find_bin
from src.camera import RealSenseCamera, list_devices
from src.detect import PackageDetector
from src.types import CLASS_IDS, Package

warnings.filterwarnings("ignore", message=".*Found GPU.*compute capability.*")

# Drop a detection only when its core is the red bin, not when it fills the opening.
_MAX_RED_FRAC = 0.72
# This many non-red pixels inside the opening is a package, even if the mask also touches plastic.
_MIN_PACKAGE_PX = 800


def parse_conf(value: str) -> float:
    cleaned = value.strip().rstrip(".,")
    try:
        conf = float(cleaned)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from exc
    if not 0.0 < conf <= 1.0:
        raise argparse.ArgumentTypeError("--conf must be between 0 and 1")
    return conf


def power_mode() -> str:
    try:
        import subprocess

        out = subprocess.check_output(["nvpmodel", "-q"], text=True, timeout=1.0)
        for line in out.splitlines():
            if "NV Power Mode" in line:
                return line.split(":", 1)[-1].strip()
    except Exception:
        pass
    return "?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense red-bin empty check")
    parser.add_argument("--model", default="yoloe-11s-seg.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=parse_conf, default=0.12)
    parser.add_argument("--width", type=int, default=848, help="Preferred color/depth width (auto-falls back per camera)")
    parser.add_argument("--height", type=int, default=480, help="Preferred color/depth height")
    parser.add_argument("--fps", type=int, default=30, help="Preferred stream FPS")
    parser.add_argument("--serial", default=None, help="RealSense serial if more than one camera is plugged in")
    parser.add_argument("--list-cameras", action="store_true", help="Print connected RealSense cameras and exit")
    parser.add_argument("--windowed", action="store_true", help="Do not go fullscreen")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 = run forever)")
    return parser.parse_args()


def _core(mask: np.ndarray) -> np.ndarray:
    """Inner part of a mask, so a red halo around a package does not hide it."""
    area = int(mask.sum())
    if area < 400:
        return mask
    k = int(round(float(np.sqrt(area)) * 0.12))
    k = max(3, min(21, k))
    if k % 2 == 0:
        k += 1
    core = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if int(core.sum()) < 100:
        return mask
    return core > 0


def _centroid_inside(mask: np.ndarray, opening: np.ndarray) -> bool:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return False
    cy = int(round(float(ys.mean())))
    cx = int(round(float(xs.mean())))
    h, w = opening.shape
    return 0 <= cy < h and 0 <= cx < w and bool(opening[cy, cx])


def items_in_bin(packages: list[Package], view: BinView) -> list[Package]:
    """Keep box, bag, and jiffy masks that sit inside the opening."""
    if not view.found:
        return []
    kept: list[Package] = []
    for pkg in packages:
        area = int(pkg.mask.sum())
        if area < 1 or pkg.class_id not in CLASS_IDS or not _centroid_inside(pkg.mask, view.opening):
            continue
        inside = pkg.mask & view.opening
        non_red = int((inside & ~view.red).sum())
        core = _core(inside)
        core_n = max(int(core.sum()), 1)
        red_frac = float((core & view.red).sum()) / float(core_n)
        plastic_frac = float((core & view.plastic).sum()) / float(core_n)
        opening_n = max(int(view.opening.sum()), 1)
        # A mask of the bin itself is large and still red plastic.
        if int(inside.sum()) / opening_n > 0.65 and plastic_frac > 0.55:
            continue
        if non_red < _MIN_PACKAGE_PX and red_frac > _MAX_RED_FRAC:
            continue
        kept.append(pkg)
    return kept


def segment_contents(detector: PackageDetector, color: np.ndarray, view: BinView) -> list[Package]:
    """One segmentation pass on the bin opening. Masks are box, bag, or jiffy."""
    if not view.found:
        return []
    ys, xs = np.nonzero(view.opening)
    if xs.size == 0:
        return []
    h, w = color.shape[:2]
    pad = 12
    y1, y2 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    x1, x2 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    crop = color[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return []
    # Hide the red plastic so the one pass looks at the packages, not the bin.
    crop[view.plastic[y1:y2, x1:x2]] = (32, 32, 32)
    mapped: list[Package] = []
    for pkg in detector.infer(crop):
        if pkg.mask.shape[:2] != crop.shape[:2]:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = pkg.mask
        mask &= view.opening
        if int(mask.sum()) < 100:
            continue
        bbox = pkg.bbox.astype(np.float32).copy()
        bbox[0] += x1
        bbox[2] += x1
        bbox[1] += y1
        bbox[3] += y1
        mapped.append(
            Package(
                class_id=pkg.class_id,
                prompt=pkg.prompt,
                conf=pkg.conf,
                bbox=bbox,
                mask=mask,
            )
        )
    return items_in_bin(mapped, view)


def main() -> int:
    args = parse_args()
    if args.list_cameras:
        cams = list_devices()
        if not cams:
            print("No RealSense cameras found.")
            return 1
        for cam in cams:
            usb = f"  USB {cam['usb']}" if cam["usb"] else ""
            print(f"{cam['name']}  serial={cam['serial']}{usb}")
        return 0

    cam = RealSenseCamera(
        width=args.width, height=args.height, fps=args.fps, serial=args.serial
    )
    print("Starting RealSense ...")
    cam.start()
    print(f"{cam.describe()}  depth_scale={cam.depth_scale:.5f}")

    detector = PackageDetector(model_name=args.model, imgsz=args.imgsz, conf=args.conf)
    mode = power_mode()
    gpu = f"{detector.device} {'FP16' if detector.half else 'FP32'}"
    print(f"Compute {gpu}  nvpmodel={mode}")
    if mode != "MAXN_SUPER":
        print("Board is not in MAXN_SUPER (HUD may say throttled). For more FPS:")
        print("  sudo nvpmodel -m 2 && sudo jetson_clocks")

    win = "Bin Empty Check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if not args.windowed:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    paused = False
    fps = 0.0
    ema = 0.9
    frame_i = 0
    items: list[Package] = []
    view = BinView(
        found=False,
        opening=np.zeros((1, 1), dtype=bool),
        interior=np.zeros((1, 1), dtype=bool),
        red=np.zeros((1, 1), dtype=bool),
        content=np.zeros((1, 1), dtype=bool),
        blobs=[],
        outline=np.zeros((0, 2), dtype=np.int32),
        floor_z_m=None,
        height=np.full((1, 1), np.nan, dtype=np.float32),
        plastic=np.zeros((1, 1), dtype=bool),
    )
    t_inf = t_bin = 0.0
    print("Running. q=quit  p=pause")
    try:
        while True:
            t0 = time.perf_counter()
            color, depth = cam.read()
            if color is None or depth is None:
                continue
            if not paused:
                t_bin = time.perf_counter()
                view = find_bin(color, depth, cam.depth_scale)
                t_bin = time.perf_counter() - t_bin
                t_inf = time.perf_counter()
                items = segment_contents(detector, color, view)
                t_inf = time.perf_counter() - t_inf
            extra = f"{gpu}  {mode}  inf {t_inf * 1000:.0f}ms  bin {t_bin * 1000:.0f}ms"
            vis = draw(color, view, items, fps, extra=extra)
            cv2.imshow(win, vis)
            dt = time.perf_counter() - t0
            inst = 1.0 / dt if dt > 1e-6 else 0.0
            fps = inst if fps == 0.0 else (ema * fps + (1.0 - ema) * inst)
            frame_i += 1
            if args.max_frames and frame_i >= args.max_frames:
                cv2.waitKey(1)
                out = ROOT / "models" / "last_bin_frame.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), vis)
                state = "BIN NOT FOUND" if not view.found else ("EMPTY" if not items else f"{len(items)} remaining")
                print(f"Saved {out} after {frame_i} frames ({fps:.1f} FPS)  {state}")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

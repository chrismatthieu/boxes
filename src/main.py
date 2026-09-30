"""Live mixed-package volume demo: D436 + YOLOE-seg + depth measurement."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classify import refine_package
from src.camera import RealSenseCamera
from src.detect import PackageDetector
from src.measure import measure_all
from src.overlay import draw
from src.stack import mark_stacked, split_layers
from src.track import SessionTracker

warnings.filterwarnings("ignore", message=".*Found GPU.*compute capability.*")


def parse_conf(value: str) -> float:
    cleaned = value.strip().rstrip(".,")
    try:
        conf = float(cleaned)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from exc
    if not 0.0 < conf <= 1.0:
        raise argparse.ArgumentTypeError("--conf must be between 0 and 1")
    return conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense mixed-package volume demo")
    parser.add_argument("--model", default="yoloe-11s-seg.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=parse_conf, default=0.12)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--windowed", action="store_true", help="Do not go fullscreen")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 = run forever)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cam = RealSenseCamera(width=args.width, height=args.height, fps=args.fps)
    print("Starting RealSense D436 ...")
    cam.start()
    assert cam.intrinsics is not None
    fx, fy = cam.intrinsics.fx, cam.intrinsics.fy
    ppx, ppy = cam.intrinsics.ppx, cam.intrinsics.ppy
    print(
        f"Stream {cam.width}x{cam.height} @ {cam.fps}  "
        f"depth_scale={cam.depth_scale:.5f}  fx={fx:.1f}"
    )

    detector = PackageDetector(model_name=args.model, imgsz=args.imgsz, conf=args.conf)
    tracker = SessionTracker()

    win = "Package Volume Demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if not args.windowed:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    paused = False
    fps = 0.0
    ema = 0.9
    frame_i = 0
    print("Running. q=quit  r=reset counts  p=pause")
    try:
        while True:
            t0 = time.perf_counter()
            color, depth = cam.read()
            if color is None or depth is None:
                continue
            if not paused:
                packages = detector.infer(color)
                expanded = []
                for pkg in packages:
                    measured = measure_all(
                        [pkg], depth, fx, fy, ppx, ppy, cam.depth_scale
                    )[0]
                    layers = split_layers(measured, depth, cam.depth_scale)
                    if len(layers) == 1:
                        expanded.append(measured)
                    else:
                        expanded.extend(
                            measure_all(layers, depth, fx, fy, ppx, ppy, cam.depth_scale)
                        )
                stacked_warning = mark_stacked(expanded)
                refined = []
                for pkg in expanded:
                    cleaned = refine_package(pkg)
                    if cleaned is not None:
                        refined.append(cleaned)
                packages = tracker.update(refined)
            else:
                packages = [t.last_pkg for t in tracker.tracks if t.last_pkg is not None]
                stacked_warning = any(p.stacked for p in packages)

            vis = draw(color, packages, tracker, fps, stacked_warning)
            cv2.imshow(win, vis)
            dt = time.perf_counter() - t0
            inst = 1.0 / dt if dt > 1e-6 else 0.0
            fps = inst if fps == 0.0 else (ema * fps + (1.0 - ema) * inst)
            frame_i += 1
            if args.max_frames and frame_i >= args.max_frames:
                cv2.waitKey(1)
                out = ROOT / "models" / "last_frame.jpg"
                cv2.imwrite(str(out), vis)
                print(f"Saved {out} after {frame_i} frames ({fps:.1f} FPS)")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                tracker.reset()
                print("Counts reset.")
            if key == ord("p"):
                paused = not paused
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

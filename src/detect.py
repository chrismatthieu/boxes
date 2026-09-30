"""YOLOE open-vocabulary instance segmentation for packages."""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from .types import CLASS_BOX, CLASS_JIFFY, CLASS_PADDED_BAG, MAILER_PROMPTS, PROMPT_TO_CLASS, PROMPTS, Package


def _bbox_area(bbox: np.ndarray) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment_of_smaller(a: np.ndarray, b: np.ndarray) -> float:
    """How much of the smaller box sits inside the intersection."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min(_bbox_area(a), _bbox_area(b))
    return inter / smaller if smaller > 0 else 0.0


def _similar_size(a: np.ndarray, b: np.ndarray) -> bool:
    aa, ab = _bbox_area(a), _bbox_area(b)
    larger = max(aa, ab, 1.0)
    return min(aa, ab) / larger >= 0.60


def _prefer_incoming(kept: Package, incoming: Package) -> bool:
    """Same physical item: keep the better class.

    Envelopes may replace a plastic-bag label. They must not replace a box —
    small cartons often score as kraft/jiffy in open-vocab YOLO.
    """
    if incoming.class_id == CLASS_JIFFY and kept.class_id == CLASS_PADDED_BAG:
        return incoming.conf >= kept.conf * 0.80
    if incoming.class_id == CLASS_JIFFY and kept.class_id == CLASS_BOX:
        return False
    if incoming.class_id == CLASS_BOX and kept.class_id == CLASS_JIFFY:
        return incoming.conf >= kept.conf * 0.85
    return incoming.conf > kept.conf + 0.05


def suppress_duplicates(packages: list[Package]) -> list[Package]:
    """Keep the largest object; drop labels and class-flicker doubles on the same item."""
    if len(packages) <= 1:
        return packages
    ordered = sorted(packages, key=lambda p: _bbox_area(p.bbox), reverse=True)
    kept: list[Package] = []
    for pkg in ordered:
        skip = False
        for i, other in enumerate(kept):
            iou = _bbox_iou(pkg.bbox, other.bbox)
            nested = _containment_of_smaller(pkg.bbox, other.bbox)
            if iou < 0.40 and nested < 0.65:
                continue
            if _similar_size(pkg.bbox, other.bbox) and _prefer_incoming(other, pkg):
                kept[i] = pkg
            skip = True
            break
        if not skip:
            kept.append(pkg)
    return kept

warnings.filterwarnings("ignore", message=".*Found GPU.*compute capability.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODELS = ("yoloe-11s-seg.pt", "yoloe-26s-seg.pt", "yoloe-26n-seg.pt")


class PackageDetector:
    def __init__(
        self,
        model_name: str = "yoloe-11s-seg.pt",
        imgsz: int = 640,
        conf: float = 0.12,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLOE
        from ultralytics import settings as yolo_settings

        self.imgsz = imgsz
        self.conf = conf
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.quantize = 16 if self.device.startswith("cuda") else None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            yolo_settings.update({"weights_dir": str(MODELS_DIR)})
        except Exception:
            pass
        last_error: Exception | None = None
        self.model = None
        tried = (model_name,) + tuple(m for m in DEFAULT_MODELS if m != model_name)
        for name in tried:
            try:
                weights = self._resolve_weights(name)
                print(f"Loading {weights.name} on {self.device} ...")
                self.model = YOLOE(str(weights))
                downloaded = Path(name)
                if downloaded.exists() and downloaded.resolve().parent != MODELS_DIR.resolve():
                    dest = MODELS_DIR / downloaded.name
                    if not dest.exists():
                        downloaded.replace(dest)
                break
            except Exception as exc:
                last_error = exc
                print(f"Could not load {name}: {exc}")
        if self.model is None:
            raise RuntimeError(f"Could not load a YOLOE-seg model: {last_error}")
        names = list(PROMPTS)
        emb_path = MODELS_DIR / "package_text_pe.pt"
        embeddings = None
        if emb_path.exists():
            try:
                cached = torch.load(emb_path, map_location="cpu", weights_only=False)
                if isinstance(cached, dict) and cached.get("names") == names:
                    embeddings = cached["embeddings"]
                    print(f"Loaded cached text embeddings ({len(names)} prompts).")
            except Exception as exc:
                print(f"Could not read {emb_path.name}: {exc}")
        if embeddings is None:
            print("Baking text embeddings with MobileCLIP (one-time, about a minute)...")
            embeddings = self.model.get_text_pe(names)
            torch.save({"names": names, "embeddings": embeddings}, emb_path)
            print(f"Saved {emb_path}")
        self.model.set_classes(names, embeddings)
        try:
            self.model.fuse(verbose=False)
        except Exception:
            pass
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self._predict(dummy)
        except Exception as exc:
            if self.quantize == 16:
                print(f"FP16 warmup failed ({exc}); using FP32.")
                self.quantize = None
                self._predict(dummy)
            else:
                raise
        prec = "FP16" if self.quantize == 16 else "FP32"
        print(f"Text prompts set ({len(names)} classes) on {self.device} {prec}.")

    @property
    def half(self) -> bool:
        return self.quantize == 16

    def _predict(self, bgr: np.ndarray):
        return self.model.predict(
            source=bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=0.55,
            max_det=12,
            agnostic_nms=False,
            verbose=False,
            device=self.device,
            quantize=self.quantize,
            retina_masks=False,
        )

    def _resolve_weights(self, model_name: str) -> Path:
        local = MODELS_DIR / model_name
        if local.exists():
            return local
        return Path(model_name)

    def infer(self, bgr: np.ndarray) -> list[Package]:
        results = self._predict(bgr)
        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        h, w = bgr.shape[:2]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names
        masks_data = None
        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()

        packages: list[Package] = []
        for i, bbox in enumerate(boxes):
            prompt = names.get(int(clss[i]), "") if isinstance(names, dict) else str(names[int(clss[i])])
            class_id = PROMPT_TO_CLASS.get(prompt)
            if class_id is None:
                continue
            if masks_data is not None and i < len(masks_data):
                mask = masks_data[i]
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                mask_bool = mask > 0.5
            else:
                mask_bool = np.zeros((h, w), dtype=bool)
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask_bool[y1:y2, x1:x2] = True
            if int(mask_bool.sum()) < 180:
                continue
            conf = float(confs[i])
            if prompt in MAILER_PROMPTS and conf < min(0.95, self.conf + 0.14):
                continue
            packages.append(
                Package(
                    class_id=class_id,
                    prompt=prompt,
                    conf=conf,
                    bbox=bbox.astype(np.float32),
                    mask=mask_bool,
                )
            )
        return suppress_duplicates(packages)

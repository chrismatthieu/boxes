"""RealSense color+depth capture (D415, D436, D585, and similar)."""

from __future__ import annotations

import ctypes
import fcntl
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs

# Same WxH@fps on both sensors when possible. 848x480 is D435/D436 and D585;
# D415 has no 848x480 depth and typically lands on 640x480 or 1280x720.
# 1280x960 is the D585 full mode.
PREFERRED_MODES = (
    (848, 480, 30),
    (1280, 960, 30),
    (640, 480, 30),
    (1280, 720, 30),
    (960, 540, 30),
    (640, 480, 15),
    (1280, 720, 15),
    (640, 360, 30),
    (480, 270, 30),
    (424, 240, 30),
)

USB2_MODES = (
    (640, 480, 15),
    (640, 360, 15),
    (424, 240, 15),
    (640, 480, 6),
)

_COLOR_FORMATS = (
    rs.format.bgr8,
    rs.format.rgb8,
    rs.format.yuyv,
    rs.format.uyvy,
    rs.format.nv12,
)

# D500 RGB calibration table (librealsense d500_rgb_calibration_table).
_GET_HKR_CONFIG_TABLE = 0xA7
_RGB_CALIB_TABLE = 0xB9
_TABLE_HEADER = 16
_SENSOR_COEF = 160
_DISTORTION_BROWN = 1
_DISTORTION_BROWN_AND_FISHEYE = 2


@dataclass
class Pinhole:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


def _info(dev: rs.device, key: rs.camera_info) -> str:
    try:
        return str(dev.get_info(key))
    except Exception:
        return ""


def list_devices() -> list[dict[str, str]]:
    ctx = rs.context()
    found: list[dict[str, str]] = []
    for dev in ctx.query_devices():
        found.append(
            {
                "name": _info(dev, rs.camera_info.name) or "RealSense",
                "serial": _info(dev, rs.camera_info.serial_number),
                "usb": _info(dev, rs.camera_info.usb_type_descriptor),
                "product_line": _info(dev, rs.camera_info.product_line),
            }
        )
    return found


def _collect_modes(dev: rs.device, stream_type: rs.stream, formats: tuple[rs.format, ...]) -> set[tuple[int, int, int]]:
    modes: set[tuple[int, int, int]] = set()
    for sensor in dev.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != stream_type:
                continue
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            if video.format() not in formats:
                continue
            modes.add((int(video.width()), int(video.height()), int(video.fps())))
    return modes


def _collect_color_formats(dev: rs.device) -> dict[tuple[int, int, int], rs.format]:
    """Map each color mode to the preferred pixel format the SDK can stream."""
    rank = {fmt: i for i, fmt in enumerate(_COLOR_FORMATS)}
    best: dict[tuple[int, int, int], rs.format] = {}
    for sensor in dev.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            fmt = video.format()
            if fmt not in rank:
                continue
            mode = (int(video.width()), int(video.height()), int(video.fps()))
            current = best.get(mode)
            if current is None or rank[fmt] < rank[current]:
                best[mode] = fmt
    return best


def pick_stream_modes(
    depth_modes: set[tuple[int, int, int]],
    color_modes: set[tuple[int, int, int]],
    requested: tuple[int, int, int] | None = None,
    usb2: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return (depth_whf, color_whf). Prefer identical modes so align is cheap."""
    if not depth_modes or not color_modes:
        raise RuntimeError("RealSense device has no color+depth video modes")

    candidates: list[tuple[int, int, int]] = []
    if requested is not None:
        candidates.append(requested)
    if usb2:
        candidates.extend(USB2_MODES)
    candidates.extend(PREFERRED_MODES)

    seen: set[tuple[int, int, int]] = set()
    for mode in candidates:
        if mode in seen:
            continue
        seen.add(mode)
        if mode in depth_modes and mode in color_modes:
            return mode, mode

    for fps in (30, 15, 10, 6, 5):
        depths = [m for m in depth_modes if m[2] == fps]
        colors = [m for m in color_modes if m[2] == fps]
        if depths and colors:
            depth = max(depths, key=lambda m: m[0] * m[1])
            color = max(colors, key=lambda m: m[0] * m[1])
            return depth, color

    depth = max(depth_modes, key=lambda m: (m[2], m[0] * m[1]))
    color = max(color_modes, key=lambda m: (m[2], m[0] * m[1]))
    return depth, color


def _select_device(serial: str | None) -> rs.device:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise RuntimeError(
            "No RealSense camera found. Plug in a D415, D436, or D585 on a USB 3 port."
        )
    if serial:
        for dev in devices:
            if _info(dev, rs.camera_info.serial_number) == serial:
                return dev
        known = ", ".join(
            f"{_info(d, rs.camera_info.name)} ({_info(d, rs.camera_info.serial_number)})" for d in devices
        )
        raise RuntimeError(f"No RealSense with serial {serial!r}. Connected: {known}")
    return devices[0]


def _scale_intrinsics(base: Pinhole, width: int, height: int) -> Pinhole:
    """Match librealsense compute_rect_params_from_resolution."""
    if base.width <= 0 or base.height <= 0:
        raise RuntimeError("Calibration image size is zero")
    scale = max(width / base.width, height / base.height)
    crop_x = (base.width * scale - width) * 0.5
    crop_y = (base.height * scale - height) * 0.5
    return Pinhole(
        width=width,
        height=height,
        fx=base.fx * scale,
        fy=base.fy * scale,
        ppx=(base.ppx + 0.5) * scale - crop_x - 0.5,
        ppy=(base.ppy + 0.5) * scale - crop_y - 0.5,
    )


def _apply_fisheye_principal(intr: Pinhole, coeffs: tuple[float, ...]) -> Pinhole:
    """D500 brown+fisheye tables need a small K adjustment after scaling."""
    focal_x, focal_y, shift_x, shift_y = coeffs[5:9]
    return Pinhole(
        width=intr.width,
        height=intr.height,
        fx=intr.fx * (1.0 + focal_x),
        fy=intr.fy * (1.0 + focal_y),
        ppx=intr.fx * shift_x + intr.ppx,
        ppy=intr.fy * shift_y + intr.ppy,
    )


def _read_rgb_calibration(dev: rs.device) -> bytes:
    debug = dev.as_debug_protocol()
    command = debug.build_command(_GET_HKR_CONFIG_TABLE, 1, _RGB_CALIB_TABLE, 0, 0)
    payload = bytes(debug.send_and_receive_raw_data(command))
    if len(payload) < 4 + _TABLE_HEADER + _SENSOR_COEF:
        raise RuntimeError(f"D585 RGB calibration response is too short ({len(payload)} bytes)")
    table = payload[4:]
    table_id = int.from_bytes(table[2:4], "little")
    if table_id != _RGB_CALIB_TABLE:
        raise RuntimeError(f"Expected RGB calibration table 0xB9, got 0x{table_id:04X}")
    return table


def _d500_color_pose(
    table: bytes, width: int, height: int
) -> tuple[Pinhole, np.ndarray, np.ndarray]:
    """Color pinhole plus depth-to-color rotation and translation (meters)."""
    coef = _TABLE_HEADER
    base = _unpack_mini(table, coef)
    model = int.from_bytes(table[coef + 24 : coef + 28], "little")
    coeffs = tuple(np.frombuffer(table[coef + 28 : coef + 80], dtype="<f4").astype(np.float64))
    reserved3 = table[coef + 83]
    use_base = model == _DISTORTION_BROWN and reserved3 == 0
    if use_base:
        intr = _scale_intrinsics(base, width, height)
    else:
        rectified = _unpack_mini(table, coef + _SENSOR_COEF + 12)
        intr = _scale_intrinsics(rectified, width, height)
        if model == _DISTORTION_BROWN_AND_FISHEYE:
            intr = _apply_fisheye_principal(intr, coeffs)
    # float3x3 is stored column-wise. pts @ R applies that matrix to row vectors.
    rotation = np.frombuffer(table[coef + _SENSOR_COEF - 36 : coef + _SENSOR_COEF], dtype="<f4")
    rotation = np.array(rotation, dtype=np.float64).reshape(3, 3)
    translation_mm = np.frombuffer(table[coef + _SENSOR_COEF : coef + _SENSOR_COEF + 12], dtype="<f4")
    # librealsense flips the sign while converting mm to meters.
    translation = -np.array(translation_mm, dtype=np.float64) * 0.001
    return intr, rotation, translation


def _unpack_mini(table: bytes, offset: int) -> Pinhole:
    width, height = np.frombuffer(table[offset : offset + 4], dtype="<u2")
    ppx, ppy, fx, fy = np.frombuffer(table[offset + 4 : offset + 20], dtype="<f4")
    return Pinhole(int(width), int(height), float(fx), float(fy), float(ppx), float(ppy))


def _usb_device_sysfs(physical_port: str) -> str:
    path = physical_port
    marker = "/video4linux/"
    if marker in path:
        path = path.split(marker, 1)[0]
    base = os.path.basename(path)
    if ":" in base:
        path = os.path.join(os.path.dirname(path), base.split(":", 1)[0])
    return path


def _ioc(direction: int, number: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord("V") << 8) | number


class _FmtDesc(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("description", ctypes.c_char * 32),
        ("pixelformat", ctypes.c_uint32),
        ("mbus_code", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class _Discrete(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32)]


class _Stepwise(ctypes.Structure):
    _fields_ = [
        ("min_width", ctypes.c_uint32),
        ("max_width", ctypes.c_uint32),
        ("step_width", ctypes.c_uint32),
        ("min_height", ctypes.c_uint32),
        ("max_height", ctypes.c_uint32),
        ("step_height", ctypes.c_uint32),
    ]


class _SizeUnion(ctypes.Union):
    _fields_ = [("discrete", _Discrete), ("stepwise", _Stepwise)]


class _FrmSize(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("union", _SizeUnion),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class _Fract(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


class _IvalStep(ctypes.Structure):
    _fields_ = [("min", _Fract), ("max", _Fract), ("step", _Fract)]


class _IvalUnion(ctypes.Union):
    _fields_ = [("discrete", _Fract), ("stepwise", _IvalStep)]


class _FrmIval(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("union", _IvalUnion),
        ("reserved", ctypes.c_uint32 * 2),
    ]


_VIDIOC_ENUM_FMT = _ioc(3, 2, ctypes.sizeof(_FmtDesc))
_VIDIOC_ENUM_FRAMESIZES = _ioc(3, 74, ctypes.sizeof(_FrmSize))
_VIDIOC_ENUM_FRAMEINTERVALS = _ioc(3, 75, ctypes.sizeof(_FrmIval))
_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
_YUYV = int.from_bytes(b"YUYV", "little")


def _v4l_formats(node: str) -> set[bytes]:
    formats: set[bytes] = set()
    fd = os.open(node, os.O_RDWR | os.O_NONBLOCK)
    try:
        index = 0
        while index < 32:
            desc = _FmtDesc()
            desc.index = index
            desc.type = _V4L2_BUF_TYPE_VIDEO_CAPTURE
            try:
                fcntl.ioctl(fd, _VIDIOC_ENUM_FMT, desc)
            except OSError:
                break
            formats.add(bytes(desc.pixelformat.to_bytes(4, "little")))
            index += 1
    finally:
        os.close(fd)
    return formats


def _v4l_yuyv_modes(node: str) -> set[tuple[int, int, int]]:
    modes: set[tuple[int, int, int]] = set()
    fd = os.open(node, os.O_RDWR | os.O_NONBLOCK)
    try:
        for size_index in range(32):
            size = _FrmSize()
            size.index = size_index
            size.pixel_format = _YUYV
            try:
                fcntl.ioctl(fd, _VIDIOC_ENUM_FRAMESIZES, size)
            except OSError:
                break
            if int(size.type) != 1:
                continue
            width = int(size.union.discrete.width)
            height = int(size.union.discrete.height)
            for ival_index in range(32):
                ival = _FrmIval()
                ival.index = ival_index
                ival.pixel_format = _YUYV
                ival.width = width
                ival.height = height
                try:
                    fcntl.ioctl(fd, _VIDIOC_ENUM_FRAMEINTERVALS, ival)
                except OSError:
                    break
                if int(ival.type) != 1 or int(ival.union.discrete.numerator) == 0:
                    continue
                fps = int(round(ival.union.discrete.denominator / ival.union.discrete.numerator))
                modes.add((width, height, fps))
    finally:
        os.close(fd)
    return modes


def _dedicated_color_node(dev: rs.device) -> tuple[str, set[tuple[int, int, int]]]:
    """YUYV node for the D585 dedicated RGB sensor.

    pyrealsense2 2.58 exposes that sensor's controls but zero color profiles on
    the D585 prototype (USB PID 0x0C08). The kernel still advertises YUYV.
    """
    prefix = _usb_device_sysfs(_info(dev, rs.camera_info.physical_port))
    found: list[tuple[str, set[tuple[int, int, int]]]] = []
    sysfs = "/sys/class/video4linux"
    if not os.path.isdir(sysfs):
        raise RuntimeError("No V4L2 devices found for the D585 color sensor")
    for entry in sorted(os.listdir(sysfs)):
        device_path = os.path.realpath(os.path.join(sysfs, entry, "device"))
        if not device_path.startswith(prefix):
            continue
        node = f"/dev/{entry}"
        try:
            formats = _v4l_formats(node)
        except OSError:
            continue
        if b"YUYV" not in formats:
            continue
        modes = _v4l_yuyv_modes(node)
        if modes:
            found.append((node, modes))
    if not found:
        raise RuntimeError(
            "D585 color sensor has no SDK profiles and no YUYV video node was found"
        )
    found.sort(key=lambda item: len(item[1]), reverse=True)
    return found[0]


_PIXEL_GRIDS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    key = (height, width)
    cached = _PIXEL_GRIDS.get(key)
    if cached is not None:
        return cached
    us = np.empty((height, width), dtype=np.float32)
    vs = np.empty((height, width), dtype=np.float32)
    us[:] = np.arange(width, dtype=np.float32)
    vs[:] = np.arange(height, dtype=np.float32)[:, None]
    _PIXEL_GRIDS[key] = (us, vs)
    return us, vs


def align_depth_to_color(
    depth: np.ndarray,
    depth_scale: float,
    depth_intr: Pinhole,
    color_intr: Pinhole,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Project depth Z into the color image. Stores the depth-camera Z, like rs.align."""
    height, width = depth.shape
    us, vs = _pixel_grid(height, width)
    z_image = depth.astype(np.float32) * np.float32(depth_scale)
    valid = z_image > np.float32(0.05)
    if not np.any(valid):
        return np.zeros((color_intr.height, color_intr.width), dtype=np.uint16)
    z = z_image[valid]
    x = (us[valid] - np.float32(depth_intr.ppx)) * (np.float32(1.0) / np.float32(depth_intr.fx)) * z
    y = (vs[valid] - np.float32(depth_intr.ppy)) * (np.float32(1.0) / np.float32(depth_intr.fy)) * z
    rot = np.asarray(rotation, dtype=np.float32)
    trans = np.asarray(translation, dtype=np.float32)
    x_c = rot[0, 0] * x + rot[1, 0] * y + rot[2, 0] * z + trans[0]
    y_c = rot[0, 1] * x + rot[1, 1] * y + rot[2, 1] * z + trans[1]
    z_c = rot[0, 2] * x + rot[1, 2] * y + rot[2, 2] * z + trans[2]
    good = z_c > np.float32(0.05)
    inv_z = np.zeros_like(z_c)
    np.divide(np.float32(1.0), z_c, out=inv_z, where=good)
    uc = np.rint(np.float32(color_intr.fx) * x_c * inv_z + np.float32(color_intr.ppx)).astype(np.int32)
    vc = np.rint(np.float32(color_intr.fy) * y_c * inv_z + np.float32(color_intr.ppy)).astype(np.int32)
    inside = good & (uc >= 0) & (uc < color_intr.width) & (vc >= 0) & (vc < color_intr.height)
    uc = uc[inside]
    vc = vc[inside]
    z = z[inside]
    flat = np.full(color_intr.height * color_intr.width, np.inf, dtype=np.float32)
    np.minimum.at(flat, vc * np.int32(color_intr.width) + uc, z)
    nearest = flat.reshape(color_intr.height, color_intr.width)
    nearest[~np.isfinite(nearest)] = 0
    return np.clip(np.rint(nearest / np.float32(depth_scale)), 0, 65535).astype(np.uint16)


def _bgr_from_frame(color: np.ndarray, fmt: rs.format) -> np.ndarray:
    if fmt == rs.format.bgr8:
        return color
    if fmt == rs.format.rgb8:
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    if fmt in (rs.format.yuyv, rs.format.uyvy):
        code = cv2.COLOR_YUV2BGR_YUYV if fmt == rs.format.yuyv else cv2.COLOR_YUV2BGR_UYVY
        if color.ndim == 3 and color.shape[2] == 2:
            return cv2.cvtColor(color, code)
        if color.ndim == 2:
            return cv2.cvtColor(color, code)
    if fmt == rs.format.nv12 and color.ndim == 2:
        return cv2.cvtColor(color, cv2.COLOR_YUV2BGR_NV12)
    return color


class RealSenseCamera:
    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
    ) -> None:
        self.requested = (int(width), int(height), int(fps))
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial = serial or ""
        self.name = "RealSense"
        self.usb = ""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.depth_scale = 0.001
        self.intrinsics: Pinhole | rs.intrinsics | None = None
        self._started = False
        self._color_cap: cv2.VideoCapture | None = None
        self._color_format: rs.format | None = None
        self._depth_intr: Pinhole | None = None
        self._color_rotation: np.ndarray | None = None
        self._color_translation: np.ndarray | None = None

        self.spatial.set_option(rs.option.filter_magnitude, 1)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.6)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 0)

    def start(self) -> None:
        dev = _select_device(self.serial or None)
        self.name = _info(dev, rs.camera_info.name) or "RealSense"
        self.serial = _info(dev, rs.camera_info.serial_number)
        self.usb = _info(dev, rs.camera_info.usb_type_descriptor)
        self.product_line = _info(dev, rs.camera_info.product_line)
        usb2 = self.usb.startswith("2")

        depth_modes = _collect_modes(dev, rs.stream.depth, (rs.format.z16,))
        color_formats = _collect_color_formats(dev)
        v4l_node: str | None = None
        if color_formats:
            color_modes = set(color_formats)
        else:
            v4l_node, color_modes = _dedicated_color_node(dev)
        depth_mode, color_mode = pick_stream_modes(
            depth_modes, color_modes, requested=self.requested, usb2=usb2
        )

        self.config.enable_device(self.serial)
        self.config.enable_stream(
            rs.stream.depth, depth_mode[0], depth_mode[1], rs.format.z16, depth_mode[2]
        )
        if v4l_node is None:
            self._color_format = color_formats[color_mode]
            self.config.enable_stream(
                rs.stream.color,
                color_mode[0],
                color_mode[1],
                self._color_format,
                color_mode[2],
            )
        profile = None
        try:
            profile = self.pipeline.start(self.config)
        except RuntimeError as exc:
            self.config.disable_all_streams()
            self.config.enable_device(self.serial)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            if v4l_node is None:
                self._color_format = rs.format.bgr8
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
                color_mode = (640, 480, 15)
            try:
                profile = self.pipeline.start(self.config)
            except RuntimeError as exc2:
                raise RuntimeError(
                    f"Could not start {self.name} ({self.serial}) at "
                    f"depth {depth_mode} / color {color_mode}: {exc}; fallback failed: {exc2}"
                ) from exc2
            depth_mode = (640, 480, 15)
        if profile is None:
            raise RuntimeError(f"Could not start {self.name} ({self.serial})")

        self._started = True
        try:
            self._configure_depth(profile)
            if v4l_node is None:
                self._finish_sdk_color(profile)
            else:
                self._finish_v4l_color(dev, v4l_node, color_mode)
        except Exception:
            self.stop()
            raise

    def _configure_depth(self, profile: rs.pipeline_profile) -> None:
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        if self.product_line.startswith("D4"):
            try:
                depth_sensor.set_option(
                    rs.option.visual_preset, int(rs.rs400_visual_preset.high_accuracy)
                )
            except Exception:
                pass
        try:
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1)
        except Exception:
            pass
        try:
            if depth_sensor.supports(rs.option.laser_power):
                rng = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power, rng.max)
        except Exception:
            pass
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intr = depth_stream.get_intrinsics()
        self._depth_intr = Pinhole(
            int(depth_stream.width()),
            int(depth_stream.height()),
            float(depth_intr.fx),
            float(depth_intr.fy),
            float(depth_intr.ppx),
            float(depth_intr.ppy),
        )
        self.depth_mode = (
            int(depth_stream.width()),
            int(depth_stream.height()),
            int(depth_stream.fps()),
        )

    def _finish_sdk_color(self, profile: rs.pipeline_profile) -> None:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()
        self.width = int(color_stream.width())
        self.height = int(color_stream.height())
        self.fps = int(color_stream.fps())
        self.color_mode = (self.width, self.height, self.fps)

    def _finish_v4l_color(
        self,
        dev: rs.device,
        node: str,
        color_mode: tuple[int, int, int],
    ) -> None:
        cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, color_mode[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, color_mode[1])
        cap.set(cv2.CAP_PROP_FPS, color_mode[2])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open D585 color node {node}")
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"D585 color node {node} did not deliver a frame")
        self._color_cap = cap
        self._color_node = node
        self.width = int(frame.shape[1])
        self.height = int(frame.shape[0])
        self.fps = color_mode[2]
        self.color_mode = (self.width, self.height, self.fps)
        table = _read_rgb_calibration(dev)
        intr, rotation, translation = _d500_color_pose(table, self.width, self.height)
        self.intrinsics = intr
        self._color_rotation = rotation
        self._color_translation = translation

    def describe(self) -> str:
        usb = f" USB {self.usb}" if self.usb else ""
        depth = getattr(self, "depth_mode", (self.width, self.height, self.fps))
        color = getattr(self, "color_mode", (self.width, self.height, self.fps))
        if depth == color:
            streams = f"{color[0]}x{color[1]} @{color[2]}"
        else:
            streams = (
                f"color {color[0]}x{color[1]}@{color[2]} + "
                f"depth {depth[0]}x{depth[1]}@{depth[2]} (aligned to color)"
            )
        if self._color_cap is not None:
            streams += ", color aligned from dedicated RGB"
        return f"{self.name} ({self.serial}){usb}  {streams}"

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        frames = self.pipeline.wait_for_frames(timeout_ms=2500)
        if self._color_cap is not None:
            return self._read_v4l(frames)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        color = _bgr_from_frame(np.asanyarray(color_frame.get_data()), self._color_format or rs.format.bgr8)
        depth = np.asanyarray(depth_frame.get_data())
        return color, depth

    def _read_v4l(self, frames: rs.composite_frame) -> tuple[np.ndarray | None, np.ndarray | None]:
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None, None
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        assert self._color_cap is not None
        ok, color = self._color_cap.read()
        if not ok or color is None:
            return None, None
        if color.ndim == 2 or (color.ndim == 3 and color.shape[2] == 2):
            color = cv2.cvtColor(color, cv2.COLOR_YUV2BGR_YUYV)
        depth = np.asanyarray(depth_frame.get_data())
        assert self._depth_intr is not None
        assert isinstance(self.intrinsics, Pinhole)
        assert self._color_rotation is not None and self._color_translation is not None
        aligned = align_depth_to_color(
            depth,
            self.depth_scale,
            self._depth_intr,
            self.intrinsics,
            self._color_rotation,
            self._color_translation,
        )
        return color, aligned

    def stop(self) -> None:
        if self._color_cap is not None:
            self._color_cap.release()
            self._color_cap = None
        if self._started:
            self.pipeline.stop()
            self._started = False

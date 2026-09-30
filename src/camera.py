"""RealSense D400 color+depth capture (D415, D436, and similar)."""

from __future__ import annotations

import numpy as np
import pyrealsense2 as rs

# Same WxH@fps on both sensors when possible. 848x480 is D435/D436; D415 has no
# 848x480 depth and typically lands on 640x480 or 1280x720.
PREFERRED_MODES = (
    (848, 480, 30),
    (640, 480, 30),
    (1280, 720, 30),
    (960, 540, 30),
    (640, 480, 15),
    (1280, 720, 15),
    (640, 360, 30),
    (424, 240, 30),
)

USB2_MODES = (
    (640, 480, 15),
    (640, 360, 15),
    (424, 240, 15),
    (640, 480, 6),
)


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

    for fps in (30, 15, 10, 6):
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
            "No RealSense camera found. Plug in a D415 or D436 on a USB 3 port."
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
        self.intrinsics: rs.intrinsics | None = None
        self._started = False

        self.spatial.set_option(rs.option.filter_magnitude, 1)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.6)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 0)

    def start(self) -> None:
        dev = _select_device(self.serial or None)
        self.name = _info(dev, rs.camera_info.name) or "RealSense"
        self.serial = _info(dev, rs.camera_info.serial_number)
        self.usb = _info(dev, rs.camera_info.usb_type_descriptor)
        usb2 = self.usb.startswith("2")

        depth_modes = _collect_modes(dev, rs.stream.depth, (rs.format.z16,))
        color_modes = _collect_modes(
            dev,
            rs.stream.color,
            (rs.format.bgr8, rs.format.rgb8, rs.format.yuyv),
        )
        depth_mode, color_mode = pick_stream_modes(
            depth_modes, color_modes, requested=self.requested, usb2=usb2
        )

        self.config.enable_device(self.serial)
        self.config.enable_stream(
            rs.stream.depth, depth_mode[0], depth_mode[1], rs.format.z16, depth_mode[2]
        )
        self.config.enable_stream(
            rs.stream.color, color_mode[0], color_mode[1], rs.format.bgr8, color_mode[2]
        )
        try:
            profile = self.pipeline.start(self.config)
        except RuntimeError as exc:
            # Last-resort: 640x480@15, which almost every D400 can do.
            self.config.disable_all_streams()
            self.config.enable_device(self.serial)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            try:
                profile = self.pipeline.start(self.config)
            except RuntimeError as exc2:
                raise RuntimeError(
                    f"Could not start {self.name} ({self.serial}) at "
                    f"depth {depth_mode} / color {color_mode}: {exc}; fallback failed: {exc2}"
                ) from exc2

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
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

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()
        self.width = int(color_stream.width())
        self.height = int(color_stream.height())
        self.fps = int(color_stream.fps())
        self.depth_mode = (int(depth_stream.width()), int(depth_stream.height()), int(depth_stream.fps()))
        self.color_mode = (self.width, self.height, self.fps)
        self._started = True

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
        return f"{self.name} ({self.serial}){usb}  {streams}"

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        frames = self.pipeline.wait_for_frames(timeout_ms=2500)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return color, depth

    def stop(self) -> None:
        if self._started:
            self.pipeline.stop()
            self._started = False

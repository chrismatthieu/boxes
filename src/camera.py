"""RealSense D436 color+depth capture with alignment and depth filters."""

from __future__ import annotations

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(self, width: int = 848, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.depth_scale = 0.001
        self.intrinsics: rs.intrinsics | None = None
        self._started = False

        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 1)

    def start(self) -> None:
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        try:
            profile = self.pipeline.start(self.config)
        except RuntimeError:
            self.config.disable_all_streams()
            self.width, self.height = 640, 480
            self.config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
            self.config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            profile = self.pipeline.start(self.config)

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
            if depth_sensor.supports(rs.option.laser_power):
                rng = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power, rng.max)
        except Exception:
            pass

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()
        self._started = True

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        frames = self.pipeline.wait_for_frames(timeout_ms=2500)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return color, depth

    def stop(self) -> None:
        if self._started:
            self.pipeline.stop()
            self._started = False

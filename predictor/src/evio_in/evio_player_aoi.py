#!/usr/bin/env python3
import argparse  # noqa: INP001
import time

import cv2
import numpy as np

from src.evio_in.pacer import Pacer
from src.evio_in.dat_file import BatchRange, DatFileSource

from src.evio_in.ev_rpm_pipeline import (
    EvTachAOIConfig,
    EvTachAOIDetector,
    PipelineConfig,
    EE3PConfig,
    StreamingRpmPipeline,
)


# ----------------------------------------------------------------------
# Utility: decode one time window of events into x,y,polarity
# ----------------------------------------------------------------------
def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    timestamps: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0
    t_coords = timestamps[event_indexes].astype(np.int64, copy=False)

    return x_coords, y_coords, pixel_polarity, t_coords


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (127, 127, 127),  # gray
    on_color: tuple[int, int, int] = (255, 255, 255),  # white
    off_color: tuple[int, int, int] = (0, 0, 0),  # black
) -> np.ndarray:
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (0, 0, 0),  # black by default
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  drops/ms={pacer.instantaneous_drop_rate:.2f}"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    else:
        first_row_str = (
            f"(target) speed={pacer.speed:.2f}x  force_speed = False, no drops"
        )

    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    # first row
    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # second row
    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


# --------------------------------------------------------------------
# ----------------------------------------------------------------------
# Utility: decode one time window of events into x,y,polarity
# ----------------------------------------------------------------------


# Live AOI tracker – uses EvTachAOIDetector once we have enough data
# ----------------------------------------------------------------------
class LiveAOITracker:
    """
    Accumulates events for an initial AOI window, runs EvTachAOIDetector once,
    then keeps the AOIs fixed and returns them for each subsequent frame.
    """

    def __init__(
        self,
        sensor_width: int,
        sensor_height: int,
        num_clusters: int = 1,
        aoi_window_us: int = 150_000,  # 150 ms similar to PipelineConfig.aoi_window_us
        heatmap_bin_size: int = 4,
        max_kmeans_iters: int = 20,
        outlier_thresh_factor: float = 3.0,
    ):
        self.cfg = EvTachAOIConfig(
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            num_clusters=num_clusters,
            heatmap_bin_size=heatmap_bin_size,
            max_kmeans_iters=max_kmeans_iters,
            outlier_thresh_factor=outlier_thresh_factor,
        )
        self.detector = EvTachAOIDetector(self.cfg)
        self.aoi_window_us = aoi_window_us

        self._reset_state()

    def _reset_state(self) -> None:
        self._acc_x: list[np.ndarray] = []
        self._acc_y: list[np.ndarray] = []
        self._t_start_us: int | None = None
        self._have_aois: bool = False
        self.aois = []  # type: list

    def consume_window(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_range: BatchRange,
    ):
        """
        Feed one window of events (x,y) and update / compute AOIs if needed.
        Returns the current AOI list (possibly empty).
        """
        if not self._have_aois:
            if self._t_start_us is None:
                self._t_start_us = batch_range.start_ts_us

            # accumulate events
            self._acc_x.append(x.astype(np.int16, copy=False))
            self._acc_y.append(y.astype(np.int16, copy=False))

            # check whether we've covered the AOI detection window
            if batch_range.end_ts_us - self._t_start_us >= self.aoi_window_us:
                xs = (
                    np.concatenate(self._acc_x)
                    if self._acc_x
                    else np.zeros(0, dtype=np.int16)
                )
                ys = (
                    np.concatenate(self._acc_y)
                    if self._acc_y
                    else np.zeros(0, dtype=np.int16)
                )

                if xs.size > 0:
                    print(
                        f"[INFO] AOI detection on {xs.size} events "
                        f"from {self._t_start_us} to {batch_range.end_ts_us} us"
                    )
                    self.aois = self.detector.detect_aois(xs, ys)
                    print(f"[INFO] Detected {len(self.aois)} AOIs")
                else:
                    print("[WARN] No events accumulated for AOI detection.")

                self._have_aois = True

        # Once AOIs are computed, we just keep returning them
        return self.aois


def draw_aois(
    frame: np.ndarray, aois, rpms: dict[int, float] | None = None, color=(0, 0, 255)
) -> None:
    if rpms is None:
        rpms = {}
    for idx, aoi in enumerate(aois):
        x1, y1, x2, y2 = aoi.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"AOI {idx}"
        if idx in rpms:
            label += f" {rpms[idx]:.0f} RPM"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


# ----------------------------------------------------------------------
# Main GUI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument(
        "--window", type=float, default=10, help="Window duration in ms"
    )
    parser.add_argument(
        "--speed", type=float, default=1, help="Playback speed (1 is real time)"
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Sensor width in pixels"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Sensor height in pixels"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=1,
        help="Number of rotating objects (AOIs) to find",
    )
    parser.add_argument(
        "--aoi-window",
        type=float,
        default=150.0,
        help="AOI detection window in ms (integration time before running k-means)",
    )
    args = parser.parse_args()

    src = DatFileSource(
        args.dat,
        width=args.width,
        height=args.height,
        window_length_us=int(args.window * 1000),
    )

    # Enforce playback speed via dropping:
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    # --- create streaming RPM pipeline *after* args exist ---
    ee3p_cfg = EE3PConfig(symmetry_order=3)  # or make this a CLI arg
    pipe_cfg = PipelineConfig(
        source_type="evio",  # unused in streaming mode
        sensor_width=args.width,
        sensor_height=args.height,
        num_clusters=args.clusters,
        aoi_window_us=int(args.aoi_window * 1000.0),
        rpm_window_us=30_000,
        rpm_step_us=10_000,
        ee3p_config=ee3p_cfg,
    )
    rpm_pipeline = StreamingRpmPipeline(pipe_cfg)

    cv2.namedWindow("Evio Player", cv2.WINDOW_NORMAL)
    for batch_range in pacer.pace(src.ranges()):
        window = get_window(
            src.event_words,
            src.order,
            src.timestamps_raw,
            batch_range.start,
            batch_range.stop,
        )
        x_coords, y_coords, polarities_on, t_coords = window

        # Feed batch into streaming pipeline
        aois, rpm_dict = rpm_pipeline.consume_batch(x_coords, y_coords, t_coords)

        frame = get_frame(
            (x_coords, y_coords, polarities_on), width=args.width, height=args.height
        )

        # Draw AOIs and RPMs
        if aois:
            draw_aois(frame, aois, rpm_dict)

        draw_hud(frame, pacer, batch_range)

        cv2.imshow("Evio Player", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

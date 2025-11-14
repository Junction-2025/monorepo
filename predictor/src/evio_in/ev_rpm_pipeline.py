#!/usr/bin/env python3
"""
ev_rpm_pipeline.py

Event-based RPM estimation pipeline:

    EVIO/FRED/NeRDD -> EV-Tach-style AOI detection -> EE3P-style RPM estimator

Designed for:
- Fan scenarios (static + moving)
- Drone idle / moving (props visible)
"""

import argparse
import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Literal

import numpy as np

# Optional imports; you may need to install these.
from src.evio_in.dat_file import DatFileSource
from src.evio_in.recording import open_dat


try:
    import h5py
except ImportError:
    h5py = None

# ----------------------------------------------------------------------
# 1. Data sources
# ----------------------------------------------------------------------

class EventSourceBase:
    """Abstract base class for event sources."""

    def iter_packets(self):
        """Yield packets (x, y, t, p) as numpy arrays, time-sorted within packet."""
        raise NotImplementedError

    def iter_all(self):
        """Yield all events in a single tuple of arrays (x,y,t,p)."""
        xs, ys, ts, ps = [], [], [], []
        for x, y, t, p in self.iter_packets():
            xs.append(x)
            ys.append(y)
            ts.append(t)
            ps.append(p)
        if not xs:
            return (np.array([], dtype=np.int16),
                    np.array([], dtype=np.int16),
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.int8))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        t = np.concatenate(ts)
        p = np.concatenate(ps)
        # Ensure sorted by time
        order = np.argsort(t)
        return x[order], y[order], t[order], p[order]



class EvioDatSource(EventSourceBase):
    """Read .dat files via EVIO (open_dat), returning a single time-sorted packet."""

    def __init__(self, path: str, sensor_width: int, sensor_height: int):
        if open_dat is None:
            raise RuntimeError("evio not installed or not importable.")
        self.path = path
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height

    def iter_packets(self):
        # Load recording; timestamps are already sorted by time
        rec = open_dat(self.path, width=self.sensor_width, height=self.sensor_height)

        # t: int64 [N], already in time order
        t = np.asarray(rec.timestamps, dtype=np.int64)

        # event_words: uint32 [N_raw]; rec.order permutes into time order
        words_time_ordered = rec.event_words[rec.order].astype(np.uint32, copy=False)

        # Decode packed words into x, y, polarity (same logic as play_dat.get_window)
        x = (words_time_ordered & 0x3FFF).astype(np.int16, copy=False)
        y = ((words_time_ordered >> 14) & 0x3FFF).astype(np.int16, copy=False)
        p_bool = ((words_time_ordered >> 28) & 0xF) > 0
        p = p_bool.astype(np.int8, copy=False)

        # Sanity: enforce equal length
        n = min(len(x), len(y), len(t), len(p))
        x, y, t, p = x[:n], y[:n], t[:n], p[:n]

        # Yield one big packet; EventSourceBase.iter_all() will just pass it through
        yield x, y, t, p




class Hdf5EventSource(EventSourceBase):
    """
    Generic HDF5 event source for FRED/NeRDD-style files.
    You must adjust the dataset keys to match your specific file structure.
    """

    def __init__(self, path: str,
                 x_key: str = "events/x",
                 y_key: str = "events/y",
                 t_key: str = "events/t",
                 p_key: str = "events/p"):
        if h5py is None:
            raise RuntimeError("h5py not installed.")
        self.path = path
        self.x_key = x_key
        self.y_key = y_key
        self.t_key = t_key
        self.p_key = p_key

    def iter_packets(self):
        with h5py.File(self.path, "r") as f:
            x = f[self.x_key][:]
            y = f[self.y_key][:]
            t = f[self.t_key][:]  # usually microseconds
            p = f[self.p_key][:]
        # Single big packet
        yield x.astype(np.int16), y.astype(np.int16), t.astype(np.int64), p.astype(np.int8)


# ----------------------------------------------------------------------
# 2. EV-Tach-style AOI detection (heatmap + k-means + outlier removal)
# ----------------------------------------------------------------------

@dataclass
class AOI:
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray  # boolean mask over events


@dataclass
class EvTachAOIConfig:
    sensor_width: int
    sensor_height: int
    num_clusters: int
    heatmap_bin_size: int = 4  # 4x4 pixel bins
    max_kmeans_iters: int = 20
    outlier_thresh_factor: float = 3.0  # Dm * factor


class EvTachAOIDetector:
    """
    Simplified EV-Tach rotating-object extractor.

    Steps:
      - Accumulate heatmap over all events
      - Initialize centroids from highest-density bins
      - Run k-means clustering on XY
      - Remove outliers based on distance to centroid
      - Build AOI bounding boxes
    """

    def __init__(self, config: EvTachAOIConfig):
        self.cfg = config

    def _build_heatmap(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bs = self.cfg.heatmap_bin_size
        gw = self.cfg.sensor_width // bs + 1
        gh = self.cfg.sensor_height // bs + 1
        heat = np.zeros((gh, gw), dtype=np.int32)

        bin_x = np.clip(x // bs, 0, gw - 1)
        bin_y = np.clip(y // bs, 0, gh - 1)
        for bx, by in zip(bin_x, bin_y):
            heat[by, bx] += 1

        ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        return heat, xs, ys

    def _init_centroids_from_heatmap(self, heat: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Simple variant of EV-Tach heatmap-based centroid init:
        - pick the hottest bin
        - then farthest high-valued bins.
        """
        k = self.cfg.num_clusters
        flat_idx = np.argsort(heat.ravel())[::-1]  # descending
        coords = np.column_stack((xs.ravel()[flat_idx], ys.ravel()[flat_idx]))
        vals = heat.ravel()[flat_idx]

        # Only keep bins with nonzero events
        valid = vals > 0
        coords = coords[valid]
        vals = vals[valid]
        if len(coords) == 0:
            raise RuntimeError("Heatmap empty, no events?")

        # Use first bin as first centroid
        chosen = [coords[0]]

        # For subsequent centroids, choose farthest high-valued bins
        for _ in range(1, k):
            # compute distance to nearest chosen centroid
            dists = np.min(np.linalg.norm(coords[:, None, :] - np.array(chosen)[None, :, :], axis=-1), axis=1)
            # prefer high-value & far bins
            scores = dists * (vals / (vals.max() + 1e-9))
            idx = np.argmax(scores)
            chosen.append(coords[idx])

        # Convert bin indices back to pixel space (center of bin)
        bs = self.cfg.heatmap_bin_size
        centroids_pix = np.array(chosen, dtype=np.float32)
        centroids_pix = (centroids_pix + 0.5) * bs
        return centroids_pix  # shape (k,2) as (x,y)

    def _kmeans_xy(self, points: np.ndarray, init_centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Basic k-means on 2D points.

        points: [N,2]
        init_centroids: [K,2]
        Returns:
            labels: [N] cluster ids
            centroids: [K,2]
        """
        centroids = init_centroids.copy()
        K = centroids.shape[0]
        labels = np.zeros(points.shape[0], dtype=np.int32)

        for _ in range(self.cfg.max_kmeans_iters):
            # assign
            dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=-1)  # [N,K]
            new_labels = np.argmin(dists, axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            # update centroids
            for k in range(K):
                mask = labels == k
                if np.any(mask):
                    centroids[k] = points[mask].mean(axis=0)

        return labels, centroids

    def detect_aois(self, x: np.ndarray, y: np.ndarray) -> List[AOI]:
        """
        x,y: arrays for a time chunk (~150ms).
        Returns AOIs with centroid, bbox, and event mask.
        """
        cfg = self.cfg
        # Build heatmap
        heat, xs, ys = self._build_heatmap(x, y)
        # Init centroids in bin-space then convert to pixel coords
        centroids_init = self._init_centroids_from_heatmap(heat, xs, ys)

        # Run k-means on actual XY points
        pts = np.stack([x.astype(np.float32), y.astype(np.float32)], axis=1)
        labels, centroids = self._kmeans_xy(pts, centroids_init)

        aoi_list: List[AOI] = []
        for k in range(cfg.num_clusters):
            mask_k = labels == k
            if not np.any(mask_k):
                continue
            pts_k = pts[mask_k]
            c = pts_k.mean(axis=0)

            # Outlier removal
            dists = np.linalg.norm(pts_k - c[None, :], axis=1)
            Dm = np.median(dists)
            inlier_mask_local = dists <= (cfg.outlier_thresh_factor * Dm)
            if not np.any(inlier_mask_local):
                continue
            # Map back to global mask
            mask_indices = np.where(mask_k)[0]
            inlier_indices = mask_indices[inlier_mask_local]
            global_mask = np.zeros_like(x, dtype=bool)
            global_mask[inlier_indices] = True

            # Bbox
            x_in = x[global_mask]
            y_in = y[global_mask]
            x1 = int(max(0, x_in.min() - 5))
            y1 = int(max(0, y_in.min() - 5))
            x2 = int(min(cfg.sensor_width, x_in.max() + 5))
            y2 = int(min(cfg.sensor_height, y_in.max() + 5))

            aoi = AOI(
                centroid=(float(c[0]), float(c[1])),
                bbox=(x1, y1, x2, y2),
                mask=global_mask
            )
            aoi_list.append(aoi)

        return aoi_list


# ----------------------------------------------------------------------
# 3. EE3P-style RPM estimator
# ----------------------------------------------------------------------

@dataclass
class EE3PConfig:
    slice_us: int = 500        # time length of each aggregation slice (microseconds)
    min_slices: int = 10       # minimum slices needed to do correlation
    symmetry_order: int = 3    # number of identical repeats per revolution (e.g. blades)
    peak_prominence: float = 0.1
    max_peaks: int = 10


class EventAggregator:
    """
    Aggregates ROI events into a sequence of frames over time.

    Each frame is an HxW image of event counts within the ROI for a given time slice.
    """

    def __init__(self, sensor_width: int, sensor_height: int, cfg: EE3PConfig):
        self.sw = sensor_width
        self.sh = sensor_height
        self.cfg = cfg

    def aggregate_roi(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      t: np.ndarray,
                      bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        x,y,t: ROI-filtered events (global coords)
        bbox: (x1,y1,x2,y2)
        Returns:
          frames: [S,H,W] uint16
          times:  [S] float64, center time (seconds) of each slice
        """
        if len(t) == 0:
            return np.zeros((0, 1, 1), dtype=np.uint16), np.zeros(0, dtype=np.float64)

        cfg = self.cfg
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        t_min = t.min()
        t_max = t.max()
        total_us = t_max - t_min
        num_slices = int(np.ceil(total_us / cfg.slice_us))
        if num_slices < cfg.min_slices:
            return np.zeros((0, 1, 1), dtype=np.uint16), np.zeros(0, dtype=np.float64)

        frames = np.zeros((num_slices, h, w), dtype=np.uint16)
        times = np.zeros(num_slices, dtype=np.float64)

        # Assign each event to a slice
        slice_idx = ((t - t_min) // cfg.slice_us).astype(int)
        slice_idx = np.clip(slice_idx, 0, num_slices - 1)
        # Local ROI coordinates
        xr = x - x1
        yr = y - y1

        for xi, yi, ti, si in zip(xr, yr, t, slice_idx):
            if 0 <= xi < w and 0 <= yi < h:
                frames[si, yi, xi] += 1

        # Slice times (center of slice)
        for s in range(num_slices):
            t_start = t_min + s * cfg.slice_us
            t_end = t_start + cfg.slice_us
            times[s] = 0.5 * (t_start + t_end) * 1e-6  # convert to seconds

        return frames, times


class EE3PRPMEstimator:
    """
    Implements an EE3P-like periodicity estimator based on correlation
    of aggregated event frames.

    Process:
      - pick reference frame (e.g. first slice)
      - compute normalized cross-correlation with each frame
      - detect peaks in correlation signal
      - estimate RPM from peak intervals
    """

    def __init__(self, cfg: EE3PConfig):
        self.cfg = cfg

    @staticmethod
    def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
        """Normalized cross-correlation between two images."""
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        am = a.mean()
        bm = b.mean()
        a -= am
        b -= bm
        a_std = a.std() + 1e-6
        b_std = b.std() + 1e-6
        num = (a * b).sum()
        den = (a_std * b_std * a.size)
        return float(num / den)

    def _find_local_peaks(self, corr: np.ndarray) -> np.ndarray:
        """
        Simple peak detector: local maxima above mean + prominence*std.
        Returns indices of peaks.
        """
        if corr.size < 3:
            return np.array([], dtype=int)
        mean = corr.mean()
        std = corr.std() + 1e-6
        thresh = mean + self.cfg.peak_prominence * std
        peaks = []
        for i in range(1, corr.size - 1):
            if corr[i] > corr[i - 1] and corr[i] > corr[i + 1] and corr[i] > thresh:
                peaks.append(i)
        return np.array(peaks, dtype=int)

    def estimate_rpm(self, frames: np.ndarray, times: np.ndarray) -> Optional[float]:
        """
        frames: [S,H,W]
        times: [S] in seconds
        Returns RPM or None if not enough structure.
        """
        cfg = self.cfg
        S = frames.shape[0]
        if S < cfg.min_slices:
            return None

        ref = frames[0]
        corr = np.zeros(S, dtype=np.float32)
        for s in range(S):
            corr[s] = self._norm_corr(ref, frames[s])

        peaks = self._find_local_peaks(corr)
        if peaks.size < 2:
            return None

        # Limit number of peaks for robust estimation
        if peaks.size > cfg.max_peaks:
            peaks = peaks[:cfg.max_peaks]

        peak_times = times[peaks]
        # intervals between successive peaks
        dt = np.diff(peak_times)
        # Filter out weird intervals
        dt = dt[dt > 0]
        if dt.size == 0:
            return None

        median_period = np.median(dt)  # seconds between identical appearances
        if median_period <= 0:
            return None

        f_corr = 1.0 / median_period       # Hz of pattern repetition
        f_rot = f_corr / cfg.symmetry_order  # correct for symmetry (e.g. blades)
        rpm = 60.0 * f_rot
        return float(rpm)


# ----------------------------------------------------------------------
# 4. Pipeline orchestration
# ----------------------------------------------------------------------

@dataclass
class PipelineConfig:
    source_type: Literal["evio", "hdf5"]
    sensor_width: int
    sensor_height: int
    num_clusters: int
    aoi_window_us: int = 150_000    # time range used to detect AOI (~150ms)
    rpm_window_us: int = 30_000     # each RPM estimation window (~30ms)
    rpm_step_us: int = 10_000       # step between RPM windows (~10ms)
    ee3p_config: EE3PConfig = dataclasses.field(default_factory=EE3PConfig)
    evtach_config: Optional[EvTachAOIConfig] = None



class StreamingRpmPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        # set up AOI, EE3P and aggregator as before
        if cfg.evtach_config is None:
            self.evtach_cfg = EvTachAOIConfig(
                sensor_width=cfg.sensor_width,
                sensor_height=cfg.sensor_height,
                num_clusters=cfg.num_clusters,
            )
        else:
            self.evtach_cfg = cfg.evtach_config

        self.aoi_detector = EvTachAOIDetector(self.evtach_cfg)
        self.ee3p = EE3PRPMEstimator(cfg.ee3p_config)
        self.aggregator = EventAggregator(
            cfg.sensor_width, cfg.sensor_height, cfg.ee3p_config
        )

        # streaming state
        self._x_buf = np.empty(0, dtype=np.int16)
        self._y_buf = np.empty(0, dtype=np.int16)
        self._t_buf = np.empty(0, dtype=np.int64)

        self._have_aois = False
        self._aois: list[AOI] = []
        self._last_aoi_update_us: int | None = None

        self.latest_rpms: dict[int, float] = {}  # aoi_idx -> rpm

    def _append_batch(self, x: np.ndarray, y: np.ndarray, t: np.ndarray):
        # append new events
        self._x_buf = np.concatenate([self._x_buf, x.astype(np.int16, copy=False)])
        self._y_buf = np.concatenate([self._y_buf, y.astype(np.int16, copy=False)])
        self._t_buf = np.concatenate([self._t_buf, t.astype(np.int64, copy=False)])

        # drop events older than max(aoi_window_us, rpm_window_us)
        if self._t_buf.size == 0:
            return
        t_now = self._t_buf.max()
        max_window = max(self.cfg.aoi_window_us, self.cfg.rpm_window_us)
        keep_mask = self._t_buf >= (t_now - max_window)
        self._x_buf = self._x_buf[keep_mask]
        self._y_buf = self._y_buf[keep_mask]
        self._t_buf = self._t_buf[keep_mask]

    def _update_aois_if_needed(self):
        """Run EV-Tach AOI detection on the last aoi_window_us if needed."""
        if self._t_buf.size == 0:
            return

        t_now = self._t_buf.max()
        if self._last_aoi_update_us is not None:
            # e.g. re-run AOI every 50 ms
            if t_now - self._last_aoi_update_us < 50_000:
                return

        # select events from last aoi_window_us
        win_start = t_now - self.cfg.aoi_window_us
        mask = self._t_buf >= win_start
        xs = self._x_buf[mask]
        ys = self._y_buf[mask]
        if xs.size == 0:
            return

        self._aois = self.aoi_detector.detect_aois(xs, ys)
        self._have_aois = len(self._aois) > 0
        self._last_aoi_update_us = t_now

    def _update_rpm(self):
        """Estimate RPM for each AOI from the last rpm_window_us."""
        self.latest_rpms.clear()
        if not self._have_aois or self._t_buf.size == 0:
            return

        t_now = self._t_buf.max()
        win_start = t_now - self.cfg.rpm_window_us
        mask_win = self._t_buf >= win_start
        xw = self._x_buf[mask_win]
        yw = self._y_buf[mask_win]
        tw = self._t_buf[mask_win]

        for a_idx, aoi in enumerate(self._aois):
            x1, y1, x2, y2 = aoi.bbox
            m_roi = (xw >= x1) & (xw < x2) & (yw >= y1) & (yw < y2)
            if not np.any(m_roi):
                continue

            xr, yr, tr = xw[m_roi], yw[m_roi], tw[m_roi]
            frames, times = self.aggregator.aggregate_roi(xr, yr, tr, aoi.bbox)
            if frames.shape[0] == 0:
                continue

            rpm = self.ee3p.estimate_rpm(frames, times)
            if rpm is not None:
                self.latest_rpms[a_idx] = float(rpm)

    def consume_batch(self, x: np.ndarray, y: np.ndarray, t: np.ndarray):
        """
        Main entry point for streaming use.
        - x,y,t are a 10 ms batch of events (sorted by t).
        - Updates AOIs & RPMs using sliding windows.
        Returns: (aois, latest_rpms_dict)
        """
        self._append_batch(x, y, t)
        self._update_aois_if_needed()
        self._update_rpm()
        return self._aois, self.latest_rpms




class RpmPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        if cfg.evtach_config is None:
            self.evtach_cfg = EvTachAOIConfig(
                sensor_width=cfg.sensor_width,
                sensor_height=cfg.sensor_height,
                num_clusters=cfg.num_clusters,
            )
        else:
            self.evtach_cfg = cfg.evtach_config

        self.aoi_detector = EvTachAOIDetector(self.evtach_cfg)
        self.ee3p = EE3PRPMEstimator(cfg.ee3p_config)
        self.aggregator = EventAggregator(cfg.sensor_width, cfg.sensor_height, cfg.ee3p_config)

    def _make_source(self, path: str) -> EventSourceBase:
        if self.cfg.source_type == "evio":
            return EvioDatSource(
                path,
                self.cfg.sensor_width,
                self.cfg.sensor_height,
            )
        elif self.cfg.source_type == "hdf5":
            return Hdf5EventSource(path)
        else:
            raise ValueError(f"Unknown source_type: {self.cfg.source_type}")


    def run_on_file(self, path: str) -> List[Dict]:
        """
        Run pipeline on a single event file.

        Returns list of dicts:
          {
            "t_center": float (seconds),
            "aoi_idx": int,
            "rpm": float
          }
        """
        src = self._make_source(path)
        x, y, t, p = src.iter_all()
        if len(t) == 0:
            print("No events in file.")
            return []

        t_min = t.min()
        t_max = t.max()

        # 1) Use initial AOI window
        aoi_end = t_min + self.cfg.aoi_window_us
        mask_aoi = (t >= t_min) & (t <= aoi_end)
        if not np.any(mask_aoi):
            print("No events in AOI window.")
            return []

        print(f"[INFO] AOI detection window: {t_min} to {aoi_end} us ({mask_aoi.sum()} events)")

        aoi_list = self.aoi_detector.detect_aois(x[mask_aoi], y[mask_aoi])
        print(f"[INFO] Detected {len(aoi_list)} AOIs")

        # 2) Sliding RPM windows over entire sequence
        results = []
        w = self.cfg.rpm_window_us
        step = self.cfg.rpm_step_us

        t_center = t_min + w // 2
        while t_center + w // 2 <= t_max:
            win_start = t_center - w // 2
            win_end = t_center + w // 2
            mask_win = (t >= win_start) & (t <= win_end)
            if not np.any(mask_win):
                t_center += step
                continue

            xw, yw, tw, pw = x[mask_win], y[mask_win], t[mask_win], p[mask_win]

            for a_idx, aoi in enumerate(aoi_list):
                x1, y1, x2, y2 = aoi.bbox
                # ROI mask
                m_roi = (xw >= x1) & (xw < x2) & (yw >= y1) & (yw < y2)
                if not np.any(m_roi):
                    continue

                xr, yr, tr = xw[m_roi], yw[m_roi], tw[m_roi]
                frames, times = self.aggregator.aggregate_roi(xr, yr, tr, aoi.bbox)
                if frames.shape[0] == 0:
                    continue

                rpm = self.ee3p.estimate_rpm(frames, times)
                if rpm is None:
                    continue

                results.append({
                    "t_center": float(t_center * 1e-6),  # seconds
                    "aoi_idx": a_idx,
                    "rpm": float(rpm),
                })

            t_center += step

        return results


# ----------------------------------------------------------------------
# 5. CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Event-based RPM estimation (EVIO + EV-Tach AOI + EE3P).")
    parser.add_argument("path", type=str, help="Path to .dat or .h5 event file")
    parser.add_argument("--source-type", type=str, choices=["evio", "hdf5"], default="evio",
                        help="Type of source: 'evio' for .dat, 'hdf5' for FRED/NeRDD")
    parser.add_argument("--width", type=int, default=1280, help="Sensor width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Sensor height in pixels")
    parser.add_argument("--clusters", type=int, default=1, help="Number of rotating objects (AOIs) to find")
    parser.add_argument("--symmetry", type=int, default=3,
                        help="Symmetry order (e.g. number of blades; 3 for many props, 2 for typical drone blades)")
    args = parser.parse_args()

    ee3p_cfg = EE3PConfig(symmetry_order=args.symmetry)
    pipe_cfg = PipelineConfig(
        source_type=args.source_type,
        sensor_width=args.width,
        sensor_height=args.height,
        num_clusters=args.clusters,
        ee3p_config=ee3p_cfg,
    )
    pipeline = RpmPipeline(pipe_cfg)
    results = pipeline.run_on_file(args.path)

    if not results:
        print("No RPM estimates produced.")
        return

    print("t_center_s,aoi_idx,rpm")
    for r in results:
        print(f"{r['t_center']:.6f},{r['aoi_idx']},{r['rpm']:.2f}")


if __name__ == "__main__":
    main()

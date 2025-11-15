"""
Logging utilities for RPM detection pipeline.

Provides functions for:
- Creating timestamped output directories
- Saving frames, heatmaps, and labels as images
- Writing structured log files
- Managing file I/O in functional style
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import LOG_DIR


class RunContext(NamedTuple):
    """Context for a single pipeline run."""

    run_dir: Path
    frames_dir: Path
    heatmaps_dir: Path
    labels_dir: Path
    log_file: Path
    logger: logging.Logger


def create_run_directory(base_dir: Path = LOG_DIR) -> Path:
    """
    Create timestamped directory for current run.

    Args:
        base_dir: Base directory for logs

    Returns:
        Path to created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_run_context(base_dir: Path = LOG_DIR) -> RunContext:
    """
    Setup complete logging context for a pipeline run.

    Creates directory structure:
    logs/<timestamp>/
        ├── frames/      # Annotated frames
        ├── heatmaps/    # Spatial heatmaps
        ├── labels/      # Cluster labels
        └── run.log      # Text log file

    Returns:
        RunContext with all paths and logger configured
    """
    run_dir = create_run_directory(base_dir)

    # Create subdirectories
    frames_dir = run_dir / "frames"
    heatmaps_dir = run_dir / "heatmaps"
    labels_dir = run_dir / "labels"

    frames_dir.mkdir(exist_ok=True)
    heatmaps_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Setup logger
    log_file = run_dir / "run.log"
    logger = configure_logger(log_file)

    logger.info(f"Run started at {datetime.now().isoformat()}")
    logger.info(f"Output directory: {run_dir}")

    return RunContext(
        run_dir=run_dir,
        frames_dir=frames_dir,
        heatmaps_dir=heatmaps_dir,
        labels_dir=labels_dir,
        log_file=log_file,
        logger=logger,
    )


def configure_logger(log_file: Path) -> logging.Logger:
    """
    Configure logger with both file and console handlers.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("rpm_pipeline")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (warnings only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_frame(
    frame: NDArray[np.uint8],
    frame_number: int,
    output_dir: Path,
) -> Path:
    """
    Save annotated frame as PNG.

    Args:
        frame: RGB/BGR image array
        frame_number: Sequential frame number
        output_dir: Directory to save in

    Returns:
        Path to saved file
    """
    filename = f"frame_{frame_number:06d}.png"
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), frame)
    return output_path


def save_heatmap(
    heatmap: NDArray[np.int32],
    frame_number: int,
    output_dir: Path,
) -> Path:
    """
    Save heatmap as colorized PNG.

    Args:
        heatmap: 2D array of event counts
        frame_number: Sequential frame number
        output_dir: Directory to save in

    Returns:
        Path to saved file
    """
    # Normalize to 0-255
    if heatmap.max() > 0:
        normalized = (heatmap * 255.0 / heatmap.max()).astype(np.uint8)
    else:
        normalized = np.zeros_like(heatmap, dtype=np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    filename = f"heatmap_{frame_number:06d}.png"
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), colored)
    return output_path


def save_labels(
    labels: NDArray[np.int32],
    frame_number: int,
    output_dir: Path,
) -> Path:
    """
    Save cluster labels as colorized PNG.

    Args:
        labels: 2D array of cluster assignments
        frame_number: Sequential frame number
        output_dir: Directory to save in

    Returns:
        Path to saved file
    """
    # Map labels to colors (cycling through palette)
    unique_labels = np.unique(labels)
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ]

    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for i, label in enumerate(unique_labels):
        if label >= 0:  # Skip background (-1)
            mask = labels == label
            color = colors[i % len(colors)]
            colored[mask] = color

    filename = f"labels_{frame_number:06d}.png"
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), colored)
    return output_path


def log_aoi_detection(
    logger: logging.Logger,
    frame_number: int,
    num_aois: int,
    centroids: list[tuple[float, float]],
) -> None:
    """Log AOI detection results."""
    logger.info(f"Frame {frame_number}: Detected {num_aois} AOIs")
    for idx, (cx, cy) in enumerate(centroids):
        logger.debug(f"  AOI {idx}: centroid=({cx:.1f}, {cy:.1f})")


def log_rpm_estimate(
    logger: logging.Logger,
    frame_number: int,
    aoi_idx: int,
    rpm: float,
) -> None:
    """Log RPM estimation result."""
    logger.info(f"Frame {frame_number}: AOI {aoi_idx} RPM={rpm:.2f}")


def log_error(logger: logging.Logger, error_msg: str) -> None:
    """Log error message."""
    logger.error(error_msg)


def finalize_run(
    ctx: RunContext,
    total_frames: int,
    total_time_s: float,
) -> None:
    """
    Write final statistics and close logging.

    Args:
        ctx: Run context
        total_frames: Total frames processed
        total_time_s: Total processing time in seconds
    """
    ctx.logger.info("=" * 60)
    ctx.logger.info("Run complete")
    ctx.logger.info(f"Total frames processed: {total_frames}")
    ctx.logger.info(f"Total time: {total_time_s:.2f}s")
    ctx.logger.info(f"Average FPS: {total_frames / total_time_s:.2f}")
    ctx.logger.info(f"Output directory: {ctx.run_dir}")
    ctx.logger.info("=" * 60)

    # Close handlers
    for handler in ctx.logger.handlers:
        handler.close()

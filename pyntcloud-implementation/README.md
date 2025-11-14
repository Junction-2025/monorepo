# Pyntcloud Event Camera to Point Cloud Converter

Convert event camera .dat files to .ply point cloud format using pyntcloud.

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
uv sync
```

## Usage

### Basic usage (with defaults):
```bash
uv run python src/main.py
```

This will:
- Read from `data/drone_idle.dat`
- Output to `src/events.ply`
- Use timestamps as Z coordinates (normalized to 0-1 range)

### Custom input/output:
```bash
uv run python src/main.py --input path/to/input.dat --output path/to/output.ply
```

### Advanced options:

```bash
# Custom sensor dimensions
uv run python src/main.py --width 1920 --height 1080

# Use 2D sheet (Z=0) instead of timestamps
uv run python src/main.py --no-timestamp-as-z

# Don't normalize timestamps
uv run python src/main.py --no-normalise-time
```

## Options

- `--input`: Input .dat file path (default: `data/drone_idle.dat`)
- `--output`: Output .ply file path (default: `src/events.ply`)
- `--width`: Sensor width in pixels (default: 1280)
- `--height`: Sensor height in pixels (default: 720)
- `--use-timestamp-as-z`: Use timestamp as Z coordinate (default: True)
- `--no-timestamp-as-z`: Set Z coordinate to 0 (2D sheet)
- `--normalise-time`: Normalize timestamps to 0-1 range (default: True)
- `--no-normalise-time`: Don't normalize timestamps

## Output

The script generates a .ply point cloud file where:
- X, Y coordinates come from event pixel positions
- Z coordinate is either timestamp (normalized) or 0
- Colors represent polarity:
  - White (255,255,255) = positive polarity
  - Red (255,0,0) = negative polarity
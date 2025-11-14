from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280

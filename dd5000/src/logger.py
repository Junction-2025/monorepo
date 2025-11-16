from src.config import LOG_DIR
import logging
from datetime import datetime
import sys

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"{timestamp}_log.txt"

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler(str(log_file))
file_handler.setLevel(logging.DEBUG)

# Create stream handler for stderr (so tests can capture output)
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_logger():
    return logger

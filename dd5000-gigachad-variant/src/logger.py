from src.config import LOG_DIR
import logging
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"{timestamp}_log.txt"
logging.basicConfig(filename=str(log_file))
logging.getLogger().setLevel(logging.DEBUG)

logger = logging.getLogger()


def get_logger():
    return logger

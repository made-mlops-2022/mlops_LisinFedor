import logging
import sys

logger = logging.getLogger("online_inference")
str_handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter(
    "%(asctime)s\t%(levelname)s\t[%(name)s]: %(message)s",
)
logger.setLevel(logging.INFO)
str_handler.setFormatter(fmt)
logger.addHandler(str_handler)

import logging
from typing import Optional


def get_logger(filename: Optional[str] = None) -> logging.Logger:
    fmt = logging.Formatter(
        fmt="[%(asctime)s] :%(name)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # A sequential experiment runner creates several trainers in one process.
    # Use an independent logger per trainer so later FileHandlers cannot append
    # their records to earlier experiments' log files.
    logger = logging.Logger(name=__name__, level=logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    if filename is not None:
        handler = logging.FileHandler(filename=filename)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger

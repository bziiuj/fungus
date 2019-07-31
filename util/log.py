"""Utility functions regarding logging."""
import logging
import sys
from functools import partial


def get_logger(name, level=logging.DEBUG):
    """Builds a logger with provided name and adds file & console handlers to it."""
    log = logging.getLogger(name)
    log.setLevel(level)
    fh = logging.FileHandler('tmp/{}.log'.format(name))
    fh.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    log.addHandler(ch)
    log.addHandler(fh)
    return log


def _log_top_level_exceptions(exc_type, exc_value, exc_traceback, logger):
    """Exception handler to log all uncaught exceptions using provided logger."""
    logger.exception('Uncaught exception', exc_info=(
        exc_type, exc_value, exc_traceback))


def set_excepthook(logger):
    """Sets exception handler to a function that logs everything using provided logger."""
    handler = partial(_log_top_level_exceptions, logger=logger)
    sys.excepthook = handler

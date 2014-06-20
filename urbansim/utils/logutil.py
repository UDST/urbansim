import contextlib
import logging


@contextlib.contextmanager
def log_start_finish(msg, logger, level=logging.DEBUG):
    """
    A context manager to log messages with "start: " and "finish: "
    prefixes before and after a block.

    Parameters
    ----------
    msg : str
        Will be prefixed with "start: " and "finish: ".
    logger : logging.Logger
    level : int, optional
        Level at which to log, passed to ``logger.log``.

    """
    logger.log(level, 'start: ' + msg)
    yield
    logger.log(level, 'finish: ' + msg)

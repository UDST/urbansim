import contextlib
import logging

US_LOG_FMT = ('%(asctime)s|%(levelname)s|%(name)s|'
              '%(funcName)s|%(filename)s|%(lineno)s|%(message)s')
US_LOG_DATE_FMT = '%Y-%m-%d %H:%M:%S'
US_FMT = logging.Formatter(fmt=US_LOG_FMT, datefmt=US_LOG_DATE_FMT)


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


def set_log_level(level):
    """
    Set the logging level for urbansim.

    Parameters
    ----------
    level : int
        A supporting logging level. Use logging constants like logging.DEBUG.

    """
    logging.getLogger('urbansim').setLevel(level)


def log_to_stream(level=None):
    """
    Send log messages to the console.

    """
    handler = logging.StreamHandler()
    handler.setFormatter(US_FMT)

    if level is not None:
        handler.setLevel(level)

    logger = logging.getLogger('urbansim')
    logger.addHandler(handler)
    logger.propagate = False


def log_to_file(filename, level=None):
    """
    Send log output to the given file.

    Parameters
    ----------
    filename : str
    level : int, optional
        Optional logging level for the file handler.

    """
    handler = logging.FileHandler(filename)
    handler.setFormatter(US_FMT)

    if level is not None:
        handler.setLevel(level)

    logger = logging.getLogger('urbansim')
    logger.addHandler(handler)

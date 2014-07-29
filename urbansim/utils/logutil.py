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


def _add_urbansim_handler(handler, level=None, fmt=None, datefmt=None,
                          propagate=None):
    """
    Add a logging handler to urbansim.

    Parameters
    ----------
    handler : logging.Handler subclass
    level : int, optional
        An optional logging level that will apply only to this stream
        handler.
    fmt : str, optional
        An optional format string that will be used for the log
        messages.
    datefmt : str, optional
        An optional format string for formatting dates in the log
        messages.
    propagate : bool, optional
        Whether the urbansim logger should propagate. If None the
        propagation will not be modified, otherwise it will be set
        to this value.

    """
    if not fmt:
        fmt = US_LOG_FMT
    if not datefmt:
        datefmt = US_LOG_DATE_FMT

    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    if level is not None:
        handler.setLevel(level)

    logger = logging.getLogger('urbansim')
    logger.addHandler(handler)

    if propagate is not None:
        logger.propagate = propagate


def log_to_stream(level=None, fmt=None, datefmt=None):
    """
    Send log messages to the console.

    Parameters
    ----------
    level : int, optional
        An optional logging level that will apply only to this stream
        handler.
    fmt : str, optional
        An optional format string that will be used for the log
        messages.
    datefmt : str, optional
        An optional format string for formatting dates in the log
        messages.

    """
    _add_urbansim_handler(
        logging.StreamHandler(), fmt=fmt, datefmt=datefmt, propagate=False)


def log_to_file(filename, level=None, fmt=None, datefmt=None):
    """
    Send log output to the given file.

    Parameters
    ----------
    filename : str
    level : int, optional
        An optional logging level that will apply only to this stream
        handler.
    fmt : str, optional
        An optional format string that will be used for the log
        messages.
    datefmt : str, optional
        An optional format string for formatting dates in the log
        messages.

    """
    _add_urbansim_handler(
        logging.FileHandler(filename), fmt=fmt, datefmt=datefmt)

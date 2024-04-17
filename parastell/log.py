import logging
import time


class NullLogger(object):
    """Creates a pseudo logger object mimicking an actual logger object whose
    methods do nothing when called.
    """
    def __init__(self, *args):
        pass

    def hasHandlers(self):
        return True

    def info(self, message):
        current_time = time.localtime()
        current_time = time.strftime('%H:%M:%S', current_time)
        print(f'{current_time}: {message}')

    def warning(self, *args):
        pass

    def error(self, *args):
        pass


def init():
    """Creates and configures logger with separate stream and file handlers.

    Returns:
        logger (object): logger object.
    """
    # Create logger
    logger = logging.getLogger('log')
    # Configure base logger message level
    logger.setLevel(logging.INFO)
    # Configure stream handler
    s_handler = logging.StreamHandler()
    # Configure file handler
    f_handler = logging.FileHandler(
        filename='stellarator.log',
        mode='w'
    )
    # Define and set logging format
    format = logging.Formatter(
        fmt = '%(asctime)s: %(message)s',
        datefmt = '%H:%M:%S'
    )
    s_handler.setFormatter(format)
    f_handler.setFormatter(format)
    # Add handlers to logger
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    return logger


def check_init(logger_obj, null_logger=True):
    """Checks if a logger object has been instantiated, and if not,
    instantiates one.

    Arguments:
        logger_obj (object or None): logger object input.
        null_logger (bool): flag to indicate whether a NullLogger object should
            be returned (optional, defaults to True).
    Returns:
        logger_obj (object): logger object.
    """
    if logger_obj != None and logger_obj.hasHandlers():
        return logger_obj
    elif null_logger:
        return NullLogger()
    else:
        return init()

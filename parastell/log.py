import logging
import time


def check_init(logger_obj):
    """Checks if a logger object has been instantiated, and if not,
    instantiates one.

    Arguments:
        logger_obj (object or None): logger object input.

    Returns:
        logger_obj (object): logger object.
    """
    if logger_obj != None and logger_obj.hasHandlers():
        return logger_obj
    else:
        return NullLogger()


class NullLogger(object):
    """Creates a pseudo logger object mimicking an actual logger object whose
    methods do nothing when called.
    """

    def __init__(self):
        pass

    def hasHandlers(self):
        return True

    def info(self, message):
        current_time = time.localtime()
        current_time = time.strftime("%H:%M:%S", current_time)
        print(f"{current_time}: {message}")

    def warning(self, *args):
        pass

    def error(self, *args):
        pass


def init():
    """Creates and configures logger with separate stream and file handlers.

    Returns:
        logger (object): logger object.
    """
    logger = logging.getLogger("log")

    logger.setLevel(logging.INFO)

    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename="stellarator.log", mode="w")

    format = logging.Formatter(
        fmt="%(asctime)s: %(message)s", datefmt="%H:%M:%S"
    )
    s_handler.setFormatter(format)
    f_handler.setFormatter(format)

    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    return logger

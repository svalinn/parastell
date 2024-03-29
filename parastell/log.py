import logging

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
    f_handler = logging.FileHandler('stellarator.log')
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

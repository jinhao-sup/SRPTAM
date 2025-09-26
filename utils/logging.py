import logging


_FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'

""" 模型的日志器，区别于kcore_interactions数据处理的日志器 """
class _LoggerHolder(object):
    """
    Logger singleton instance holder.
    """
    INSTANCE = None


def get_logger():
    """
    Returns library scoped logger.
    :returns: Library logger.
    """
    if _LoggerHolder.INSTANCE is None:
        formatter = logging.Formatter(_FORMAT)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger('srtam')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _LoggerHolder.INSTANCE = logger
    return _LoggerHolder.INSTANCE


def enable_verbose_logging():
    """ Enable tensorflow logging. """
    logger = get_logger()
    logger.setLevel(logging.DEBUG)

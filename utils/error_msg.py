# 自定义的Mojito对象异常，异常消息从外面传入。写在mojito包下的__init__.py，主要包含通用方法，不需要再创建额外的异常类包，方便调用。
class Error(Exception):
    """ Custom exception for srtam related error. """
    pass

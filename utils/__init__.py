import numpy as np
from collections import defaultdict


def int_dd():
    return defaultdict(int)


"""
defaultdict 是 Python 标准库 collections 模块中的一个字典类，它提供了一个具有默认值的字典。
返回一个新的 defaultdict 对象。float 作为构造函数的参数传入，指定了字典值的默认类型为浮点数。
这意味着，任何在这个字典中尚未显式设置的键都将自动关联到一个默认值 0.0
"""
def float_dd():
    return defaultdict(float)


""" 
随机抽取一个负样本，fs应该是该user交互过的样本集
"""
def random_neg(lh, rh, fs):
    """
    random negative item
    :param lh:
    :param rh:
    :param fs: forbiden set
    :return:
    """
    t = np.random.randint(lh, rh)
    while t in fs:
        t = np.random.randint(lh, rh)
    return t

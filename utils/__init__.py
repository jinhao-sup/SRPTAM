import numpy as np
from collections import defaultdict


def int_dd():
    return defaultdict(int)

def float_dd():
    return defaultdict(float)


""" 
随机抽取一个负样本，fs是该user交互过的样本集
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

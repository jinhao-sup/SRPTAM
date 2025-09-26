import json
import os
from utils.error_msg import Error


""" 
 加载给出路径的配置文件，即configs中的3个配置文件，从程序执行指令中指定， 包含对数据集，训练超参，测试超参，cache还不清楚作用，日志配置等
 区分kcore中的默认加载ml1m的配置，这里从指令中给出
"""
def load_configuration(descriptor):  # 传入路径
    """
    Load configuration from the given descriptor.
    Args:
        descriptor:
    Returns:
    """
    if not os.path.exists(descriptor):
        raise Error(f'Configuration file {descriptor} '
                          f'not found')
    with open(descriptor, 'r') as stream:
        return json.load(stream)

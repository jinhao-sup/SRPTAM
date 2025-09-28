import json
import os
from utils.error_msg import Error


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

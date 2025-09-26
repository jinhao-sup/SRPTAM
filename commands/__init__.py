from argparse import ArgumentParser

# 这些字典定义了一些通用的命令行参数选项
# -p opt specification (train, evaluate and denoise).
# 定义了-p选项，用于指定配置文件名，其类型为字符串，默认值是'srtam'，并存储了帮助信息
OPT_PARAMS = {
    'dest': 'configuration',
    'default': 'srtam:srtam',
    'type': str,
    'action': 'store',
    'help': 'JSON filename that contains params'
}

# 定义了--best_epoch选项（测试集评估时使用），用于指定最佳训练轮次，其类型为整数，默认值为-1
OPT_BEST_EPOCH = {
    'type': int,
    'default': -1,
    'help': 'Best epoch on validation set'
}

# -a opt specification (train, evaluate and denoise).
# 定义了--verbose选项，启用此选项将显示详细日志
OPT_VERBOSE = {
    'action': 'store_true',
    'help': 'Shows verbose logs'
}


""" # 创建整体的命令行解析器 """
def create_argument_parser():
    """ Creates overall command line parser for srtam.
    :returns: Created argument parser.
    """
    parser = ArgumentParser(prog='srtam')
    subparsers = parser.add_subparsers()
    subparsers.dest = 'command'
    subparsers.required = True
    # 添加子命令解析器，每个子命令分别处理不同的功能
    _create_train_parser(subparsers.add_parser)
    _create_eval_parser(subparsers.add_parser)
    _create_analyse_parser(subparsers.add_parser)
    _create_extract_parser(subparsers.add_parser)
    return parser


"""
_create_train_parser, _create_eval_parser, _create_analyse_parser, 和 _create_extract_parser分别为训练、评估、分析和提取功能创建了子命令解析器。
每个函数都接收一个“parser factory”，即一个工厂函数，用来实际创建子解析器。
这些子解析器都添加了一些共通的参数，如-p、-ep和--verbose，这是通过调用_add_common_options函数实现的
"""
def _create_train_parser(parser_factory):  # 对应sh中的train命令
    """ Creates an argparser for training command
    :param parser_factory: Factory to use to create parser instance.
    :returns: Created and configured parser.
    """
    parser = parser_factory('train', help='Train a recommendation model')
    _add_common_options(parser)
    return parser


def _create_eval_parser(parser_factory):  # 对应sh中的eval命令
    """ Creates an argparser for evaluation command
    :param parser_factory: Factory to use to create parser instance.
    :returns: Created and configured parser.
    """
    parser = parser_factory('eval', help='Evaluate a model on the musDB test datasets')
    _add_common_options(parser)
    return parser


def _create_analyse_parser(parser_factory):  # 预定义了两个指令analyse、extract，但是好像没有具体实现
    """ Creates an argparser for evaluation command
    :param parser_factory: Factory to use to create parser instance.
    :returns: Created and configured parser.
    """
    parser = parser_factory('analyse', help='Analyse a model on a test datasets')
    _add_common_options(parser)
    return parser


def _create_extract_parser(parser_factory):
    """ Creates an argparser for extract command
    :param parser_factory: Factory to use to create parser instance.
    :returns: Created and configured parser.
    """
    parser = parser_factory('extract', help='Extract user/item embeddings')
    _add_common_options(parser)
    return parser


# 即train/eval指令都可以指定-p（参数配置文件）
def _add_common_options(parser):  # 为各个解析器添加定义好的通用参数选项
    """ Add common option to the given parser.
    :param parser: Parser to add common opt to.
    """
    parser.add_argument('-p', '--params_filename', **OPT_PARAMS)
    parser.add_argument('-ep', '--best_epoch', **OPT_BEST_EPOCH)
    parser.add_argument('--verbose', **OPT_VERBOSE)
    # 通过上述代码，程序可以根据用户输入的命令（例如 train、eval、analyse、extract）和相关选项来执行相应的操作

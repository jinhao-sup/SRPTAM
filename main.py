import sys
import warnings

from utils.error_msg import Error  # 自定义的异常消息封装类
from commands import create_argument_parser
from utils.configuration import load_configuration
from utils.logging import get_logger, enable_verbose_logging


def main(argv):
    try:
        parser = create_argument_parser()
        arguments = parser.parse_args(argv[1:])
        if arguments.verbose:
            enable_verbose_logging()
        # 确定是执行训练 (train) 还是评估 (eval)。从相应的模块动态导入 entrypoint 函数
        if arguments.command == 'train':
            from commands.train import entrypoint
        elif arguments.command == 'eval':
            from commands.eval import entrypoint
        else:
            raise Error(f'does not support command {arguments.command}')
        # 根据指令后的-p设置configuration参数作为路径，加载对应的配置文件中的配置
        params = load_configuration(arguments.configuration)
        entrypoint(params)  # 将指令参数和配置参数传入训练或测试的入口函数
    except Error as e:
        get_logger().error(e)


def entrypoint():
    """ Command line entrypoint. """
    warnings.filterwarnings('ignore')  # 禁用了警告，防止某些库产生的不重要警告影响输出的清晰度
    main(sys.argv)


if __name__ == '__main__':
    entrypoint()

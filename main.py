import sys
import warnings

from utils.error_msg import Error  # 自定义的异常类，用来包装Mojito对象返回的异常，在mojito包下的__init__.py文件
from commands import create_argument_parser  # commands包下的__init.py__中的方法
from utils.configuration import load_configuration
from utils.logging import get_logger, enable_verbose_logging


""" 作用：指令参数，配置参数解析，根据指令判断执行训练还是测试 """
def main(argv):
    try:
        # 基于Python argparse 库的自定义命令行参数解析器，即commands包下的__init__.py文件下的create_argument_parser()函数
        parser = create_argument_parser()
        arguments = parser.parse_args(argv[1:])  # 去除了第一个元素，因为 sys.argv 的第一个元素通常是脚本名或模块名，不需要进行参数解析
        if arguments.verbose:  # 即运行指令中的--verbose，记录mojito对象的详细日志,意味着日志级别会被设置得更低
            enable_verbose_logging()
        # 这段代码确定是执行训练 (train) 还是评估 (eval)。它从相应的模块动态导入 entrypoint 函数
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
    warnings.filterwarnings('ignore')  # 禁用了警告，可能是为了防止某些库产生的不重要警告影响输出的清晰度
    # argv 传递给程序的命令行参数列表（通常是 sys.argv，包含了从命令行传入的所有元素）
    main(sys.argv)


if __name__ == '__main__':
    entrypoint()

"""
要注意整个mojito文件是python package而不是directory
PyCharm之python package和directory的区别
python作为一门解释性的脚本语言。python中模块就是指一个py文件，如果我们将所有相关的代码都放在一个py文件中，则该py文件既是程序又是是模块，
但是程序和模块的设计目的是不同的，程序的目的是为了运行，而模块的目的是为了其他程序进行调用。

Directory:
Dictionary在pycharm中就是一个文件夹，放置资源文件等，该文件夹其中并不包含__init.py__文件

Python Package:
对于Python package 文件夹而言，与Dictionary不同之处在于其会自动创建 __init.py__ 文件。
简单的说，python package就是一个目录，其中包括一组模块和一个 __init.py__文件。
目录下具有init.py文件，这样可以通过from…import的方式进行.py文件的导入。

__init__.py是编辑器用来将不同的目录标识为package包
"""

"""
计算图的定义和图的运算是分开的.tensorflow是一个符号主义的库.编程模式分为两类,命令式(imperative style)和符号式(symbolic style).
命令式的程序很容易理解和调试,它按照原有的逻辑运行.符号式则相反,在现有的深度学习框架中,torch是命令式的,Caffe,Mxnet是两种模式的混合,
tensorflow完全采用符号式.符号式编程,一般先定义各种变量,然后建立一个计算图(数据流图),计算图指定了各个变量之间的计算关系,此时对计算图进行编译,没有任何输出,
计算图还是一个空壳,只有把需要运算的输入放入后,才会在模型中形成数据流.形成输出.

"""

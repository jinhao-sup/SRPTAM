from utils.error_msg import Error
from models.srtam import SRTAM

# 一个字典，用于存储支持的模型类。将字符串 'srtam' 映射到 Mojito 类
_SUPPORTED_MODELS = {
    'srtam': SRTAM
}


class ModelFactory:
    @classmethod  # 类方法，直接通过类名调用
    def generate_model(cls, sess, params, n_users, n_items, command='train'):
        """
        Factory method to generate a model
        :param sess:  TensorFlow 会话
        :param params:  模型参数字典，应该是json的training部分
        :param n_users:  用户总数
        :param n_items:  item总数
        :param command: 字符串，指定"train"或"eval"
        """
        model_name = params['model']['name']  # 字符串"srtam"
        try:
            # TODO：单独运行sh中的eval指令时，会报AttributeError: 'Mojito' object has no attribute 'user_ids'的错误，怀疑是这行代码问题
            # create a new model
            mdl = _SUPPORTED_MODELS[model_name](sess=sess, params=params, n_users=n_users, n_items=n_items)

            # 如果是 'train'，则调用模型的 build_graph 方法来构建计算图；如果是 'eval'，则调用 restore 方法来恢复模型状态
            if command == 'train':
                # build computation graph
                mdl.build_graph(name=model_name)
            # 从exp/model加载保存的最佳模型
            elif command == 'eval':
                mdl.restore(name=model_name)
            return mdl  # 返回模型实例
        except KeyError:
            raise Error(f'Currently not support model {model_name}')

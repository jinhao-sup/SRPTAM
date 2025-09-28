from utils.error_msg import Error
from models.srtam import SRTAM

# srtam是论文写作前暂定的模型名称
_SUPPORTED_MODELS = {
    'srtam': SRTAM
}


class ModelFactory:
    @classmethod
    def generate_model(cls, sess, params, n_users, n_items, command='train'):
        """
        Factory method to generate a model
        :param sess:  TensorFlow 会话
        :param params:  模型参数字典
        :param n_users:  用户总数
        :param n_items:  item总数
        :param command: 字符串，指定"train"或"eval"
        """
        model_name = params['model']['name']  # 字符串"srtam"
        try:
            # create a new model
            mdl = _SUPPORTED_MODELS[model_name](sess=sess, params=params, n_users=n_users, n_items=n_items)
            if command == 'train':
                # build computation graph
                mdl.build_graph(name=model_name)
            elif command == 'eval':
                mdl.restore(name=model_name)
            return mdl
        except KeyError:
            raise Error(f'Currently not support model {model_name}')

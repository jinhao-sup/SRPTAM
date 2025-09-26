import os
import tensorflow as tf
from utils.logging import get_logger
from utils.params import process_params
from datasets.tempo_dataset import TempoDataset
from models import ModelFactory
from commands.trainer import Trainer


# 此函数负责处理命令行提供的参数，设置训练环境，并启动模型训练过程
def entrypoint(params):
    """ Command entrypoint
    :param params: Deserialized JSON configuration file provided in CLI args.
    反序列化后的JSON配置文件（字典），由命令行接口参数提供（在__main__中处理后传入）
    """
    logger = get_logger()
    # process params
    # 调用process_params()函数来解析并组织params中的数据（嵌套字典），返回训练参数training_params和模型参数model_params（字典）
    """ 这个函数process_params会将params和training_params中的model_dir修改为拼接超参后的详细路径 """
    training_params, model_params = process_params(params)

    # create model directory if not exist
    if not os.path.exists(training_params['model_dir']):
        os.makedirs(training_params['model_dir'], exist_ok=True)
    logger.info(training_params['model_dir'])

    # TODO：这里不再使用dataset_factory，而是直接返回TempoDataset（原来的作用就是根据json的dataloader返回dataset/tempo_dataset的data）
    # load datasets
    # 将训练，验证，测试集，聚合信息，训练集items流行度，原始uid和iid列表，训练交互索引，user和item总数，最先时间戳等保存到dataset_factory的self.data字典中
    # data = dataset_factory(params=params)
    data = TempoDataset(params=params).data
    # start model training
    # 创建一个TensorFlow会话配置，允许显存增长（避免一次性分配全部显存）并允许在必要时软件放置计算操作
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    # 使用上面配置的会话参数创建一个TensorFlow会话，使用with语句确保会话能正确关闭
    with tf.compat.v1.Session(config=sess_config) as sess:
        # create model
        # 根据会话sess，训练参数training_params和数据集中的用户数n_users与物品数n_items创建模型实例
        model = ModelFactory.generate_model(sess=sess, params=training_params,
                                            n_users=data['n_users'], n_items=data['n_items'])
        # 在TensorFlow会话中运行全局变量初始化操作，确保所有模型变量都被正确初始化
        sess.run(tf.compat.v1.global_variables_initializer())
        # create a trainer to train model
        trainer = Trainer(sess, model, params)  # 将会话，模型和参数传入Trainer类实例
        logger.info('Start model training')
        trainer.fit(data=data)  # 训练100个epoch，每个epoch都会创建对应的train/valid dataloader，加载数据以及对应的fism-item
        logger.info('Model training done')

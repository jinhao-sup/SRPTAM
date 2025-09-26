import os
import numpy as np
import tensorflow as tf
from commands.evaluator import Evaluator
from datasets.tempo_dataset import TempoDataset
from loaders import dataloader_factory
from models import ModelFactory
from utils.logging import get_logger
from utils.params import process_params


def entrypoint(params):
    """ Command entrypoint
    :param params: Deserialized JSON configuration file provided in CLI args.
    反序列化后的JSON配置文件（字典），由命令行接口参数提供（在__main__中处理后传入）
    """
    logger = get_logger()
    tf.compat.v1.disable_eager_execution()
    # process params
    # 调用process_params()函数来解析并组织params中的数据，返回训练参数training_params和模型参数model_params（字典）
    training_params, model_params = process_params(params)
    model_name = model_params['name']
    dataset_params = params['dataset']
    cache_path = params['cache']['path']  # 字符串"cache/ml1m/10ucore-5icore"
    if model_params['type'] == 'tempo' and model_params['name'] == 'tisasrec':
        # eval指令中拼接的这个路径后续好像没再使用，并且也没出现类似的文件夹
        params['cache']['path'] = os.path.join(
            params['cache']['path'], f'seqlen{model_params["params"]["seqlen"]}')
    # create model directory if not exist
    if not os.path.exists(training_params['model_dir']):
        os.makedirs(training_params['model_dir'], exist_ok=True)
    logger.info(training_params['model_dir'])
    timespan = model_params['params'].get('timespan', 256)  # 时间间隔参数，没有用到

    # TODO：这里不再使用dataset_factory，而是直接返回TempoDataset（原来的作用就是根据json的dataloader返回dataset/tempo_dataset的data）
    # load dataset
    # 将训练，验证，测试集，聚合信息，训练集items流行度，原始uid和iid列表，训练实例索引，user和item总数，最先时间戳等保存到data字典中
    # data = dataset_factory(params=params)
    data = TempoDataset(params=params).data
    # 创建一个TensorFlow会话配置，允许显存增长（避免一次性分配全部显存）并允许在必要时软件放置计算操作
    # start model training
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    # 使用上面配置的会话参数创建一个TensorFlow会话，使用with语句确保会话能正确关闭
    with tf.compat.v1.Session(config=sess_config) as sess:
        # 因为exp/model下保存的是训练过程中最佳的模型（只会保存一个最佳模型，比较各个epoch的ndcg），这里指定eval则会读取这个模型
        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            command='eval')
        # generate users for test
        scores = []
        batch_size = training_params['batch_size']
        num_scored_users = params['eval'].get('n_users')  # 测试过程中涉及的用户数（TestDataLoader中缩小数据集）
        random_seeds = params['eval'].get('random_seeds')  # json用于评估的随机种子列表

        seqlen = model_params['params'].get('seqlen', 50)

        # 候选集中负样本的采样方式
        neg_sampling = 'uniform'
        if 'negative_sampling' in params['eval']:
            neg_sampling = params['eval']['negative_sampling']['type']
        """
        每次循环使用不同的种子创建测试数据加载器，即json中的"random_seeds": [1013, 2791, 4357, 6199, 7907]，对应5次测试评估过程
        每次缩减测试集时抽取的user不同，首先对用户 ID 列表进行随机洗牌，然后选取前 num_scored_users 个用户
        """
        for step, seed in enumerate(random_seeds):
            logger.info(f'EVALUATION for #{step + 1} COHORT')
            # 返回随机抽取并缩小后的测试集数据加载器，虽然这里传入的data包含整个数据集的信息，但是工厂类只会返回对应的测试数据集
            test_dataloader = dataloader_factory(
                data=data,
                batch_size=batch_size,
                seqlen=seqlen,
                mode='test',
                random_seed=seed,
                num_scored_users=num_scored_users,
                model_name=model_name,
                timespan=timespan,
                cache_path=cache_path,
                neg_sampling=neg_sampling)
            # 调用evaluator.py中的eval方法，返回在测试集的平均ndcg，hr，以列表的形式返回
            # TODO：返修后，返回结果会在多出NDCG@5，HR@5，MAP@5@10，即返回NDCG@10，HR@10，MAP@10，NDCG@5，HR@5，MAP@5，需要拼接到message在控制台打印
            score = Evaluator.eval(test_dataloader, model, item_pops=None)
            # TODO：这里给原来的NDCG和HR拼上@10
            message = [f'Step #{step + 1}', f'NDCG@10 {score[0]:8.5f} ', f'HR@10 {score[1]:8.5f} ', f'MAP@10 {score[2]:8.5f} ',
                       f'NDCG@5 {score[3]:8.5f} ', f'HR@5 {score[4]:8.5f} ', f'MAP@5 {score[5]:8.5f} ']
            # 先在控制台打印当前轮次COHORT的测试结果。测试时没有将输出信息重定向到日志文件，所以直接在控制台打印
            logger.info(','.join(message))
            # 保存每次测试结果
            scores.append(score)
        # TODO：新增MAP@10，NDCG@5，HR@5，MAP@5
        ndcg10, hr10, map10, ndcg5, hr5, map5 = zip(*scores)
        # 计算5次测试指标（ndcg，hr）的平均值和标准差
        # TODO：需要添加新指标的平均结果，以及标准差。这里给原来的NDCG和HR拼上@10
        message = ['RESULTS:',
                   f'NDCG@10: {np.mean(ndcg10):8.5f} +/- {np.std(ndcg10):8.5f}',
                   f'HR@10: {np.mean(hr10):8.5f} +/- {np.std(hr10):8.5f}',
                   f'MAP@10: {np.mean(map10):8.5f} +/- {np.std(map10):8.5f}',
                   f'NDCG@5: {np.mean(ndcg5):8.5f} +/- {np.std(ndcg5):8.5f}',
                   f'HR@5: {np.mean(hr5):8.5f} +/- {np.std(hr5):8.5f}',
                   f'MAP@5: {np.mean(map5):8.5f} +/- {np.std(map5):8.5f}']
        # 最后在控制台打印5次测试的平均结果以及标准差
        logger.info('\n'.join(message))

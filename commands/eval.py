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
    """
    logger = get_logger()
    tf.compat.v1.disable_eager_execution()
    training_params, model_params = process_params(params)
    model_name = model_params['name']
    dataset_params = params['dataset']
    cache_path = params['cache']['path']  # 字符串"cache/ml1m/10ucore-5icore"
    if model_params['type'] == 'tempo' and model_params['name'] == 'tisasrec':
        params['cache']['path'] = os.path.join(
            params['cache']['path'], f'seqlen{model_params["params"]["seqlen"]}')
    # create model directory if not exist
    if not os.path.exists(training_params['model_dir']):
        os.makedirs(training_params['model_dir'], exist_ok=True)
    logger.info(training_params['model_dir'])
    timespan = model_params['params'].get('timespan', 256)  # 时间间隔参数，没有用到
    data = TempoDataset(params=params).data

    # start model training
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.compat.v1.Session(config=sess_config) as sess:
        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            command='eval')
        # generate users for test
        scores = []
        batch_size = training_params['batch_size']
        num_scored_users = params['eval'].get('n_users')
        random_seeds = params['eval'].get('random_seeds')  # json用于评估的随机种子列表

        seqlen = model_params['params'].get('seqlen', 50)

        # 候选集中负样本的采样方式
        neg_sampling = 'uniform'
        if 'negative_sampling' in params['eval']:
            neg_sampling = params['eval']['negative_sampling']['type']

        for step, seed in enumerate(random_seeds):
            logger.info(f'EVALUATION for #{step + 1} COHORT')
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

            score = Evaluator.eval(test_dataloader, model, item_pops=None)
            message = [f'Step #{step + 1}', f'NDCG@10 {score[0]:8.5f} ', f'HR@10 {score[1]:8.5f} ', f'MAP@10 {score[2]:8.5f} ',
                       f'NDCG@5 {score[3]:8.5f} ', f'HR@5 {score[4]:8.5f} ', f'MAP@5 {score[5]:8.5f} ']
            logger.info(','.join(message))
            # 保存每次测试结果
            scores.append(score)
        # TODO：新增MAP@10，NDCG@5，HR@5，MAP@5
        ndcg10, hr10, map10, ndcg5, hr5, map5 = zip(*scores)
        # 计算5次测试指标的平均值和标准差
        message = ['RESULTS:',
                   f'NDCG@10: {np.mean(ndcg10):8.5f} +/- {np.std(ndcg10):8.5f}',
                   f'HR@10: {np.mean(hr10):8.5f} +/- {np.std(hr10):8.5f}',
                   f'MAP@10: {np.mean(map10):8.5f} +/- {np.std(map10):8.5f}',
                   f'NDCG@5: {np.mean(ndcg5):8.5f} +/- {np.std(ndcg5):8.5f}',
                   f'HR@5: {np.mean(hr5):8.5f} +/- {np.std(hr5):8.5f}',
                   f'MAP@5: {np.mean(map5):8.5f} +/- {np.std(map5):8.5f}']
        # 最后在控制台打印5次测试的平均结果以及标准差
        logger.info('\n'.join(message))

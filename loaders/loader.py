import os
import pickle
import numpy as np
from utils.logging import get_logger


""" loader的作用是根据传入的dataset对象（包含处理好的数据，之前只知道loader调用sampler，这里串联了loader和dataset）采样一个batch的训练，验证，测试实例。而sampler的作用是采样一个训练，测试实例 """
class DataLoader:
    """
    DataLoader is responsible for train/valid batch data generation 负责生成训练和验证阶段所需的批次数据
    """
    def __init__(self, dataset, n_users, n_items,
                 batch_size, seqlen, random_seed=2022, **kwargs):
        """
        Initialization
        :param dataset: 传入的为训练集，验证集或测试集
        :param n_users: 用户的总数
        :param n_items: items总数
        :param batch_size:
        :param seqlen:
        :param random_seed:
        :param kwargs: 除了json外，额外的一些超参
        """
        self.dataset = dataset
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.random_seed = random_seed
        self.kwargs = kwargs  # 将DataLoader工厂中传入的kwargs设置到self.kwargs
        self.kwargs['n_users'] = n_users
        self.rng = np.random.RandomState(self.random_seed)  # 随机数生成器
        self.logger = get_logger()  # mojito对象的日志记录器
        self.current_batch_idx = 0  # 当前数据批次的索引
        self.n_batches = 0  # 总批次数
        self.model_name = kwargs['model_name']

    """ 获取下一个批次数据 """
    def next_batch(self):
        # 如果当前批次索引 current_batch_idx 等于总批次数 n_batches，则将索引重置为 0，实现循环遍历所有批次
        if self.current_batch_idx == self.n_batches:
            self.current_batch_idx = 0
        batch_samples = self._batch_sampling(self.current_batch_idx)
        self.current_batch_idx += 1
        return batch_samples

    # 返回数据加载器中配置的总批次数
    def get_num_batches(self):
        return self.n_batches

    # 这是一个抽象方法，需要在具体使用的数据加载器子类中实现。该方法根据传入的批次索引 batch_index，生成相应的数据批次
    def _batch_sampling(self, batch_index):
        """
        Batch sampling
        :param batch_index:
        :return:
        """
        raise NotImplementedError('process method should be implemented in concrete model')

    # TODO：可以优化，因为没有使用
    # 加载在指定路径的时间关系矩阵。没有被调用
    def _get_time_matrix(self, data_path, mode, spec):
        fin_path = os.path.join(
            data_path,
            f'{mode}_time_relation_matrix_{spec}.pkl')
        self.logger.info(f'Load time matrix from {fin_path}')
        time_matrix = pickle.load(open(fin_path, 'rb'))
        return time_matrix

from utils.error_msg import Error
from loaders.train_loader import TrainDataLoader
from loaders.test_loader import TestDataLoader


"""
            DataLoader                       dataloader_factory实例化及管理具体的loader.将dataset类返回的data传入到TrainDataLoader/TestDataLoader
TrainDataLoader         TestDataLoader       trainer需要dataloader_factory返回的TrainDataLoader以及model用于训练
one_train_sample        one_test_sample      evaluator需要dataloader_factory返回的TestDataLoader以及model用于测试
具体的模型train-sampler   test-sampler         train负责创建dataset_factory，模型，trainer，并执行模型训练  eval负责创建test_dataloader，加载最佳模型，进行5次评估

dataset负责：将训练，验证，测试集，各训练实例采样元组列表（uid，nextid），交互统计信息，训练集items流行度，原始uid和iid列表，训练交互索引，user和item总数，最先时间戳等保存到data字典中
"""
# 定义一个字典，用于存储支持的数据加载器类。这个字典的键是数据加载器的名称，值是对应的类。
_SUPPORTED_DATALOADERS = {
    'train': TrainDataLoader,
    'valid': TestDataLoader,
    'test': TestDataLoader
}

# todo：这里不再使用的超参的键（前三个）可以优化'train_interactions_agg_user_key_dict','train_interactions_agg_item_key_dict','min_ts',
# 部分支持传递给数据加载器的关键字参数
_SUPPORTED_KWARGS = ['num_test_negatives', 'num_valid_users',
                     'time_relation_matrix', 'train_item_popularities']

# TODO：需要优化dataloader_factory中的sse参数
""" loader工厂的函数，其目的是基于输入参数和模式（train, valid, 或 test）来创建相应的数据加载器。这个函数使用了配置映射和支持的参数列表来动态配置和实例化具体的 DataLoader类 """
# 主要就是封装参数到kwargs，创建并返回对应的数据加载器实例
# model_name在loader下的__init__.py下（加载器工厂），由外部传入 -》 loader.py
def dataloader_factory(data, batch_size, seqlen, mode='train',
                       random_seed=2022, num_scored_users=-1,
                       cache_path='', spec='',
                       model_name=None, timespan=256,
                       frac=0.05,
                       ctx_size=6,
                       neg_sampling='uniform',
                       train_interaction_indexes=None):
    """
    Create a data loader for train/valid/test
    :param data:   （dataset的self.data）包含各种数据集（如训练集、验证集、测试集）和其他相关数据，看mode传入对应数据集，和处理后的时间特征字典（时间戳：7个特征的元组），完整数据集user总数，item总数等配置信息
    :param batch_size:
    :param seqlen:
    :param mode:   指定创建哪种类型的数据加载器（train, valid, test），默认是train。后面两种都是TestDataLoader
    :param random_seed:   随机数生成器的种子值。作用: 确保实验可重复，数据洗牌和其他随机过程的一致性
    :param num_scored_users: 默认值: -1(不使用)。控制在验证和测试过程中评估的用户数量（TestDataLoader中缩小数据集），有助于减小评分计算的规模或者针对特定用户子集进行评估
    :param cache_path:  指定加载或保存处理过的数据（如 FISM 特征、时间矩阵）的位置
    :param spec:   字符串，构建特定文件名，比如区分mojito或fism，应该在cache_path后拼接
    # :param epoch:  指定需要哪个epoch的fism数据
    :param model_name:   根据模型名称定制加载器的行为
    :param timespan:   时间间隔，控制时间间隔矩阵的阀值，默认256。实际上还没用到
    :param frac: 应该是在dataset.py中指定使用多少lfm1b的数据。作用: 在数据处理中使用，如随机选择部分数据进行训练或测试，0.05 表示使用 5% 的数据
    :param ctx_size: 使用的时间信息种类（不使用年就是6，使用就是7）
    :param neg_sampling:    负采样策略，默认是uniform，即均匀随机采样，也可以是流行度采样
    :param train_interaction_indexes:  训练交互索引列表，应该就是对应训练集中的每一次交互记录，uid，iid，ratings，timestamp。如果与dataset中的一致，则是在最后L次交互中等步长采样而来
    :return:
    """
    # 遍历 _SUPPORTED_KWARGS 列表中的每一个 key，对于每一个 key，代码首先检查 data 字典中是否存在该 key
    # 如果 data 字典中存在该 key，则将 data[key] 的值分配给 kwargs[key]，如果 data 中不存在该 key，则将 None 分配给 kwargs[key]
    """ 模型的超参除了在json配置，还会在这里根据传入的参数配置到kwargs """
    kwargs = {key: data[key] if key in data else None for key in _SUPPORTED_KWARGS}
    # 根据传入的参数更新参数字典
    kwargs['frac'] = frac
    kwargs['num_scored_users'] = num_scored_users
    kwargs['mode'] = mode
    # kwargs['epoch'] = epoch  # 指定需要哪个epoch的fism数据
    kwargs['cache_path'] = cache_path
    kwargs['spec'] = spec
    kwargs['model_name'] = model_name
    kwargs['timespan'] = timespan
    kwargs['neg_sampling'] = neg_sampling

    # 如果模式是训练，在参数字典中设置训练交互索引
    if mode == 'train':
        kwargs['train_interaction_indexes'] = train_interaction_indexes
    if 'time_dict' in data:   # 如果数据字典中存在时间字典（ts:7个时间特征的元组），则将其分配给参数字典
        kwargs['time_dict'] = data['time_dict']
    # 根据传入的参数更新参数字典
    kwargs['ctx_size'] = ctx_size  # 使用的时间信息种类

    # 验证模式，验证集作为gt
    if mode == 'valid':
        kwargs['train_set'] = data['train_set']

    # 测试模式，合并训练集和验证集数据作为新的训练集，并更新训练集的交互次数统计字典
    elif mode == 'test':
        kwargs['train_set'] = {uid: iids + data['valid_set'][uid] for uid, iids in data['train_set'].items()}

    # 从上面定义的字典取出对应模式的数据加载器，并传入参数和参数字典实例化，并返回
    try:
        return _SUPPORTED_DATALOADERS[mode](data[f'{mode}_set'],
                                            n_users=data['n_users'], n_items=data['n_items'],
                                            batch_size=batch_size, seqlen=seqlen,
                                            random_seed=random_seed, **kwargs)
    except KeyError as err:
        raise Error(f'{err}')

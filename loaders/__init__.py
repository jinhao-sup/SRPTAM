from utils.error_msg import Error
from loaders.train_loader import TrainDataLoader
from loaders.test_loader import TestDataLoader


_SUPPORTED_DATALOADERS = {
    'train': TrainDataLoader,
    'valid': TestDataLoader,
    'test': TestDataLoader
}

_SUPPORTED_KWARGS = ['num_test_negatives', 'num_valid_users',
                     'time_relation_matrix', 'train_item_popularities']

def dataloader_factory(data, batch_size, seqlen, mode='train',
                       random_seed=2022, num_scored_users=-1,
                       cache_path='', spec='',
                       model_name=None, timespan=256,
                       frac=0.05,
                       ctx_size=6,
                       neg_sampling='uniform',
                       train_interaction_indexes=None):
    kwargs = {key: data[key] if key in data else None for key in _SUPPORTED_KWARGS}
    # 根据传入的参数更新参数字典
    kwargs['frac'] = frac
    kwargs['num_scored_users'] = num_scored_users
    kwargs['mode'] = mode
    # kwargs['epoch'] = epoch
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

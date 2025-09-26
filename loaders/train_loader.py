from loaders.loader import DataLoader
from loaders.tempo_sampler import train_sample  # TODO：注意这里不同于baseline的时导入的是mojito_sampler（因为需要抽取时间序列对应的7种时间特征序列）


class TrainDataLoader(DataLoader):  # 父类是DataLoader
    def __init__(self, dataset, n_users, n_items,
                 batch_size, seqlen, need_neg_dataset=False, random_seed=2022, **kwargs):
        super(TrainDataLoader, self).__init__(
            dataset, n_users, n_items, batch_size, seqlen,
            need_neg_dataset=need_neg_dataset, random_seed=random_seed, **kwargs)  # 父类没有need_neg_dataset参数，是否需要负样本集
        # 获取训练交互索引列表，从下面的_batch_sampling_seq函数看，每个元素包含(uid, idx)，即userid，下一次交互的itemid
        # 因为每个user的交互都已经按时间排序，所以可以直接将idx作为结束位，取前面L次交互作为seq
        self.interaction_indexes = kwargs['train_interaction_indexes']
        self.rng.shuffle(self.interaction_indexes)  # 使用随机数生成器 self.rng（从DataLoader类继承，种子确定）来随机打乱交互索引
        """ 
        以训练实例总数来计算batch总数
        这里的train_interaction_indexes=interaction_indexes是一个列表，存的是（uid，交互在该user的interactions中的索引）
        """
        self.n_batches = int(len(self.interaction_indexes) / batch_size)
        # 如果交互总数不能被批次大小整除，批次数将增加一，以包含所有数据
        if self.n_batches * self.batch_size < len(self.interaction_indexes):
            self.n_batches += 1

    # 采样一个批次的交互索引用于生成一个批次的训练实例，其中当前的current_batch_idx由外部传入，由DataLoader类维护，_batch_sampling在next_batch函数中调用
    def _batch_sampling(self, batch_index):
        batch_interaction_indexes = self.interaction_indexes[
                                    batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        # 再调用下面的_batch_sampling_seq函数，实际可以合并
        return self._batch_sampling_seq(batch_interaction_indexes)

    def _batch_sampling_seq(self, batch_interaction_indexes):
        """
        Batch sampling
        :param batch_interaction_indexes:
        """
        output = []
        # 遍历每个交互索引，为每个用户-idx对生成一个训练实例，根据模型名称返回不同格式的训练实例
        """
        如果是mojito模型一个训练样本：包含uid, seq, pos, neg，seq对应的7个时间特征序列，pos对应的7个时间特征序列，item_fism_seq（已修改mojito_sampler，不再训练实例中返回）
        其他模型一个训练样本：包含uid, seq, pos, neg
        
        这里的idx是交互在该user的interactions中的索引，以idx作为最后一个pos，取前L次交互作为一个训练实例
        即一个user的整个交互序列，会产生多个训练实例，而不是像sasrec中只使用每个user训练集中的最后L次交互作为训练实例
        """
        for uid, idx in batch_interaction_indexes:
            # TODO：修改了源代码，不再使用one_train_sample接口（直接调用mojito_sampler.py）
            one_sample = train_sample(uid, idx, self.dataset, self.seqlen, self.n_items, **self.kwargs)
            output.append(one_sample)
        # 将一个batch所有的uid zip后打包成列表，seq，pos，neg等数据同理，方便使用同一个下标取出一个训练实例的所有数据，而不用再迭代实例。常规做法
        return list(zip(*output))


"""
    zip函数例子：
    zip(
    [2021, 1, 1, 4, 1, 53, 0], 
    [2021, 2, 1, 0, 32, 5, 0])，得到
    所有的年份：(2021, 2021)
    所有的月份：(1, 2)
    所有的日期：(1, 1)
"""

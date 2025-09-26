from loaders.loader import DataLoader
from loaders.tempo_sampler import test_sample  # TODO：注意这里不同于baseline的时导入的是mojito_sampler（因为需要抽取时间序列对应的7种时间特征序列）


class TestDataLoader(DataLoader):  # 父类是DataLoader
    def __init__(self, dataset, n_users, n_items,
                 batch_size, seqlen, need_neg_dataset=False, random_seed=2022, **kwargs):
        super(TestDataLoader, self).__init__(
            dataset, n_users, n_items, batch_size, seqlen,
            need_neg_dataset=need_neg_dataset, random_seed=random_seed, **kwargs)  # 父类没有need_neg_dataset参数，指示是否需要负样本集
        # 划分后数据集的格式为uid: [(item_id, time_stamp), ...]的列表。取出所有用户id
        self.user_ids = list(dataset.keys())
        # 选择特定数量的用户。实际上这个参数的值为-1（在loaders.tempo_dataset.py），所以这段if代码不会执行
        if kwargs['num_scored_users'] > 0:  # 如果指定了 num_scored_users（需要评分的用户数量），缩小测试数据集
            # 首先对用户 ID 列表进行随机洗牌，然后选取前 num_scored_users 个用户的 ID
            self.rng.shuffle(self.user_ids)
            self.user_ids = self.user_ids[:kwargs['num_scored_users']]
            self.dataset = {uid: self.dataset[uid] for uid in self.user_ids}  # 缩小数据集，仅保留前num_scored_users个用户的数据
        # 计算总的batch数，如果交互总数不能被批次大小整除，批次数将增加一，以包含所有数据
        self.n_batches = int(len(self.dataset) / self.batch_size)
        if self.n_batches * self.batch_size < len(self.dataset):
            self.n_batches += 1

    # 采样一个批次的user索引用于生成一个批次的测试实例，其中当前的current_batch_idx由外部传入，由DataLoader类维护，_batch_sampling在next_batch函数中调用
    def _batch_sampling(self, batch_index):
        # 根据批次索引从 user_ids 列表中抽取对应的用户 ID，调用下面的函数用于生成该批次的测试数据
        batch_user_ids = self.user_ids[batch_index * self.batch_size:
                                       (batch_index + 1) * self.batch_size]
        return self._batch_sampling_seq(batch_user_ids)

    # 生成测试实例批次数据，在上面的函数调用
    def _batch_sampling_seq(self, batch_user_ids):
        """
        Batch sampling
        :param batch_user_ids:
        :return:
        """
        output = []
        # 遍历用户列表，因为在生成每个batch的测试实例时，不需要idx，所以和train_loader的不同
        for uid in batch_user_ids:
            # TODO：修改了源代码，不再使用one_test_sample接口（直接调用mojito_sampler.py）
            """
            如果是mojito模型一个测试实例：包含uid, seq（用于求Ai）, 候选集test_item_ids，seq对应的7个时间特征序列（融合后用于求Ac），测试item的7个时间特征（融合后与候选集item拼接），item_fism_seq（不再返回）
            其他模型一个测试：包含uid，交互序列seq，候选集test_item_ids（一个正样本（最后一次交互）+ num_negatives个负样本）
            """
            one_sample = test_sample(uid, self.dataset, self.seqlen, self.n_items, **self.kwargs)
            output.append(one_sample)
        # 将一个batch所有的uid zip后打包成列表，seq，test_item_ids等数据同理，方便使用同一个下标取出一个测试实例的所有数据，而不用再迭代实例。常规做法
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
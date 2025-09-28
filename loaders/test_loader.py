from loaders.loader import DataLoader
from loaders.tempo_sampler import test_sample


class TestDataLoader(DataLoader):  # 父类是DataLoader
    def __init__(self, dataset, n_users, n_items,
                 batch_size, seqlen, need_neg_dataset=False, random_seed=2022, **kwargs):
        super(TestDataLoader, self).__init__(
            dataset, n_users, n_items, batch_size, seqlen,
            need_neg_dataset=need_neg_dataset, random_seed=random_seed, **kwargs)
        self.user_ids = list(dataset.keys())
        if kwargs['num_scored_users'] > 0:
            self.rng.shuffle(self.user_ids)
            self.user_ids = self.user_ids[:kwargs['num_scored_users']]
            self.dataset = {uid: self.dataset[uid] for uid in self.user_ids}
        # 计算总的batch数，如果交互总数不能被批次大小整除，批次数将增加一，以包含所有数据
        self.n_batches = int(len(self.dataset) / self.batch_size)
        if self.n_batches * self.batch_size < len(self.dataset):
            self.n_batches += 1

    def _batch_sampling(self, batch_index):
        # 根据批次索引从 user_ids 列表中抽取对应的用户 ID，用于生成该批次的测试数据
        batch_user_ids = self.user_ids[batch_index * self.batch_size:
                                       (batch_index + 1) * self.batch_size]
        return self._batch_sampling_seq(batch_user_ids)

    # 生成测试实例批次数据
    def _batch_sampling_seq(self, batch_user_ids):
        """
        Batch sampling
        :param batch_user_ids:
        :return:
        """
        output = []
        for uid in batch_user_ids:
            one_sample = test_sample(uid, self.dataset, self.seqlen, self.n_items, **self.kwargs)
            output.append(one_sample)
        return list(zip(*output))

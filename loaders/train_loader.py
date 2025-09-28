from loaders.loader import DataLoader
from loaders.tempo_sampler import train_sample


class TrainDataLoader(DataLoader):  # 父类是DataLoader
    def __init__(self, dataset, n_users, n_items,
                 batch_size, seqlen, need_neg_dataset=False, random_seed=2022, **kwargs):
        super(TrainDataLoader, self).__init__(
            dataset, n_users, n_items, batch_size, seqlen,
            need_neg_dataset=need_neg_dataset, random_seed=random_seed, **kwargs)
        self.interaction_indexes = kwargs['train_interaction_indexes']
        self.rng.shuffle(self.interaction_indexes)
        self.n_batches = int(len(self.interaction_indexes) / batch_size)
        # 如果交互总数不能被批次大小整除，批次数将增加一，以包含所有数据
        if self.n_batches * self.batch_size < len(self.interaction_indexes):
            self.n_batches += 1

    # 采样一个批次的交互索引用于生成一个批次的训练实例，其中当前的current_batch_idx由外部传入
    def _batch_sampling(self, batch_index):
        batch_interaction_indexes = self.interaction_indexes[
                                    batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        return self._batch_sampling_seq(batch_interaction_indexes)

    def _batch_sampling_seq(self, batch_interaction_indexes):
        """
        Batch sampling
        :param batch_interaction_indexes:
        """
        output = []
        for uid, idx in batch_interaction_indexes:
            one_sample = train_sample(uid, idx, self.dataset, self.seqlen, self.n_items, **self.kwargs)
            output.append(one_sample)
        return list(zip(*output))


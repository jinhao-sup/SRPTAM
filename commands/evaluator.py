import numpy as np
from tqdm import tqdm
from utils.logging import get_logger


class Evaluator:
    @classmethod
    def eval(cls, dataloader, model, item_pops=None):
        """
        Get score on valid/test dataset
        :param dataloader:
        :param model:
        :param item_pops:
        """
        ndcg = 0.0  # ndcg10
        hr = 0.0  # hr10
        n_users = 0  # 记录评估的用户数量
        n_batches = dataloader.get_num_batches()
        bad = 0
        pop = 0.0
        ndcg5 = 0.0
        hr5 = 0.0
        map5 = 0.0
        map10 = 0.0

        # for each batch
        for _ in tqdm(range(1, n_batches), desc='Evaluating...'):
            # get batch data
            batch_data = dataloader.next_batch()
            feed_dict = model.build_feedict(batch_data, is_training=False)
            # get prediction from model
            predictions = -model.predict(feed_dict)  # model.predict函数返回的是模型的self.test_logits，形状为 [batch_size, num_test_negatives+1]
            # calculate evaluation metrics
            test_item_ids = batch_data[2]  # 候选集
            for i, pred in enumerate(predictions):
                if np.all(pred == pred[0]):
                    bad += 1
                n_users += 1
                rank = pred.argsort().argsort()[0]  # 找到gt item在预测中的排名
                if rank < 10:
                    ndcg += 1 / np.log2(rank + 2)
                    hr += 1
                    map10 += 1 / (rank + 1)

                # TODO：新增计算NDCG@5，HR@5，MAP@5@10指标。注意在next item推荐MAP@K=MRR@K
                if rank < 5:
                    ndcg5 += 1 / np.log2(rank + 2)  # 注意不要与原始ndcg，hr（10）指标混淆，否则会错误累加
                    hr5 += 1
                    map5 += 1 / (rank + 1)

        out = ndcg / n_users, hr / n_users, map10 / n_users, ndcg5 / n_users, hr5 / n_users, map5 / n_users
        if item_pops is not None:
            out += (pop / hr,)
        logger = get_logger()
        logger.info(f'Number of bad results: {bad}')
        return out

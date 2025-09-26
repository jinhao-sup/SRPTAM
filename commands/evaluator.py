import numpy as np
from tqdm import tqdm
from utils.logging import get_logger


""" 评估模型在验证集/测试集上的，平均ndcg，hr """
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
        # 验证集或测试集包含的batch总数，虽然在评估的时候一般不说batch
        n_batches = dataloader.get_num_batches()
        bad = 0  # 用于记录无法正常评估的批次数量（比如所有预测值都相同）
        pop = 0.0
        # TODO：返修需要新增的评价指标，NDCG@5，HR@5，MAP@5@10
        ndcg5 = 0.0
        hr5 = 0.0
        map5 = 0.0
        map10 = 0.0

        # for each batch
        """ 以batch为单位，迭代验证集/测试集，并计算整个验证集/测试集的评估指标 """
        for _ in tqdm(range(1, n_batches), desc='Evaluating...'):
            # get batch data
            """
            从验证或测试数据加载器获取当前批次的数据，为什么在评估阶段还要使用批次，大概率是为了符合模型的输入。
            防止直接将整个验证/测试数据集看成一个batch时需要的显存过大
            """
            batch_data = dataloader.next_batch()
            # 将当前批次的验证/测试数据转换成模型输入字典，并设置到self
            feed_dict = model.build_feedict(batch_data, is_training=False)
            # get prediction from model
            # TODO：如果使用余弦距离作为预测得分，后续这里应该就不需要乘负号
            """ 先理解为进行前向推断，得到预测结果（为什么要乘负号？因为预测得分越大越好，为了好的排在前面，乘负号升序排序后，好/得分高负值小在前面），形状为 (batch_size, num_test_negatives+1) """
            predictions = -model.predict(feed_dict)  # model.predict函数返回的就是模型的self.test_logits，形状为 [batch_size, num_test_negatives+1]
            # calculate evaluation metrics
            test_item_ids = batch_data[2]  # 候选集
            for i, pred in enumerate(predictions):  # 迭代一个batch的各个候选集预测结果
                # 对于每个用户的预测，检查是否所有预测值都相同（这可能表明模型未能有效区分项目），并更新bad计数器
                if np.all(pred == pred[0]):
                    bad += 1
                n_users += 1
                rank = pred.argsort().argsort()[0]  # 找到gt item在预测中的排名
                # 如果gt排名在前10内，则更新NDCG和HR统计。否则本次预测对指标的贡献为0，但是因为有效用户数+1，所以实际会拉低指标
                if rank < 10:
                    ndcg += 1 / np.log2(rank + 2)
                    hr += 1
                    # TODO：新增MAP@10，且应该满足hr > ndcg > map
                    map10 += 1 / (rank + 1)
                    # 如果提供了item_pops，计算前10个推荐项目的平均流行度（实际所有的调用都不会走这个if）
                    if item_pops is not None:
                        indices = pred.argsort()[:10]
                        top_item_ids = [test_item_ids[i][idx] for idx in indices]
                        top_item_pops = np.mean([item_pops[iid] for iid in top_item_ids])
                        pop += top_item_pops
                # TODO：新增计算NDCG@5，HR@5，MAP@5@10指标。注意在next item推荐MAP@K=MRR@K
                if rank < 5:
                    ndcg5 += 1 / np.log2(rank + 2)  # 注意不要与原始ndcg，hr（10）指标混淆，否则会错误累加
                    hr5 += 1
                    map5 += 1 / (rank + 1)

        # 迭代完整个测试/验证集，计算对所有测试/验证用户的平均结果
        # TODO：将本次测试集新增指标结果拼接到out，修改后out包含（NDCG@10，HR@10，MAP@10，NDCG@5，HR@5，MAP@5）
        out = ndcg / n_users, hr / n_users, map10 / n_users, ndcg5 / n_users, hr5 / n_users, map5 / n_users
        if item_pops is not None:
            out += (pop / hr,)
        # 日志器记录无效结果的数量
        logger = get_logger()
        logger.info(f'Number of bad results: {bad}')
        # 返回模型在验证集/测试集上的，平均ndcg，hr
        return out

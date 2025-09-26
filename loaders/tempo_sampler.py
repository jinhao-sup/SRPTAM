import numpy as np
from utils import random_neg


""" 返回一个训练样本：包含uid, seq, pos, neg，seq对应的7个时间特征序列，pos对应的7个时间特征序列，item_fism_seq """
def train_sample(uid, nxt_idx, dataset, seqlen, n_items, **kwargs):
    """
    Sampling train data for a given user
    :param uid: user id
    :param nxt_idx: next interaction index  下一次交互在训练数据集元组列表的下标，即只取nxt_idx前的L次交互构造训练实例
    :param dataset: dataset                 大概率是传入训练集
    :param seqlen: sequence length
    :param n_items: number of items         完整数据集中物品的总数
    :param kwargs: additional parameters
    :return:
    """
    # sequence of previous items  第nxt_idx次交互前的L次交互
    seq = np.zeros([seqlen], dtype=np.int32)
    # their corresponding timestamps  对应交互时间戳序列
    in_ts_seq = np.zeros([seqlen], dtype=np.int32)
    # sequence of target timestamps   pos序列对应的时间戳序列
    nxt_ts_seq = np.zeros([seqlen], dtype=np.int32)
    # sequence of target items        对应的next item序列pos
    pos = np.zeros([seqlen], dtype=np.int32)  # 初始化为全0，默认做了0填充
    # sequence of random negative items  对应的负样本序列
    neg = np.zeros([seqlen], dtype=np.int32)
    nxt = dataset[uid][nxt_idx][0]  # 使用第nxt_idx次交互的item作为pos的最后一个元素
    nxt_time = dataset[uid][nxt_idx][1]  # nxt对应的时间戳

    idx = seqlen - 1  # 从后面开始填充各个序列，所以idx初始化为seqlen-1
    # list of historic items  训练集中该user交互过的所有items，构成favs_list
    favs = set(map(lambda x: x[0], dataset[uid]))

    """ 遍历训练交互实例索引，从对应user的第nxt_idx次（不含）交互前，逆序取前L次交互的items """
    for interaction in reversed(dataset[uid][:nxt_idx]):
        iid, ts = interaction  # 这次交互的item ID和时间戳
        # 从seq，in_ts_seq，pos，nxt_ts_seq，neg后面往前填充元素
        seq[idx] = iid
        in_ts_seq[idx] = ts
        pos[idx] = nxt
        nxt_ts_seq[idx] = nxt_time
        if nxt != 0:
            neg[idx] = random_neg(1, n_items + 1, favs)  # 指抽到的负样本不会出现在favs
        nxt = iid
        nxt_time = ts
        idx -= 1
        if idx == -1:
            break

    out = uid, seq, pos, neg

    # 如果已经将完整数据集的时间戳都转换为时间特征（tempo_dataset.py对时间进行预处理并保存到timedict.pkl文件，并设置到Dataset的self.data['time_dict']的数据开始作用），
    # 则开始处理seq和pos对应的时间特征序列
    if 'time_dict' in kwargs and kwargs['time_dict'] is not None:
        # 交互序列对应的7个时间特征序列，即每次交互都有7个时间特征，即year，month，day，dayofweek，dayofyear，week，hour
        seq_year = np.zeros([seqlen], dtype=np.int32)
        seq_month = np.zeros([seqlen], dtype=np.int32)
        seq_day = np.zeros([seqlen], dtype=np.int32)
        seq_dayofweek = np.zeros([seqlen], dtype=np.int32)
        seq_dayofyear = np.zeros([seqlen], dtype=np.int32)
        seq_week = np.zeros([seqlen], dtype=np.int32)
        seq_hour = np.zeros([seqlen], dtype=np.int32)
        # next item序列对应的7个时间特征序列
        nxt_year = np.zeros([seqlen], dtype=np.int32)
        nxt_month = np.zeros([seqlen], dtype=np.int32)
        nxt_day = np.zeros([seqlen], dtype=np.int32)
        nxt_dayofweek = np.zeros([seqlen], dtype=np.int32)
        nxt_dayofyear = np.zeros([seqlen], dtype=np.int32)
        nxt_week = np.zeros([seqlen], dtype=np.int32)
        nxt_hour = np.zeros([seqlen], dtype=np.int32)
        # 开始处理seq对应的时间序列，转为交互序列对应的7个时间特征序列
        for i, ts in enumerate(in_ts_seq):
            if ts > 0:
                seq_year[i], seq_month[i], seq_day[i], seq_dayofweek[i], \
                    seq_dayofyear[i], seq_week[i], seq_hour[i] = kwargs['time_dict'][ts]  # 直接从moji获取处理后的时间特征
        # 开始处理pos对应的时间序列，转为pos对应的7个时间特征序列
        for i, ts in enumerate(nxt_ts_seq):
            if ts > 0:
                nxt_year[i], nxt_month[i], nxt_day[i], nxt_dayofweek[i], \
                    nxt_dayofyear[i], nxt_week[i], nxt_hour[i] = kwargs['time_dict'][ts]
        # 将seq和pos对应的时间序列，追加到输出元组out中，此时out有uid, seq, pos, neg和下面7+7个时间特征序列
        out = out + (seq_year, seq_month, seq_day,
                     seq_dayofweek, seq_dayofyear, seq_week, seq_hour,
                     nxt_year, nxt_month, nxt_day,
                     nxt_dayofweek, nxt_dayofyear, nxt_week, nxt_hour)

    return out


""" 返回一个测试实例：包含uid, seq, 候选集test_item_ids，seq对应的7个时间特征序列，测试item的7个时间特征，item_fism_seq """
def test_sample(uid, dataset, seqlen, n_items, **kwargs):
    """
    Sampling test data for a given user
    :param uid:
    :param dataset: 这里传入的应该是测试集（验证时则是验证集），即每个user的最后一次交互，所构成的记录集
    :param seqlen:
    :param n_items:
    :param kwargs:
    :return:
    """
    # 从 kwargs 中提取 train_set（训练集数据，并且怀疑合并了验证集）和 num_test_negatives（生成的负样本数）
    train_set = kwargs['train_set']
    num_negatives = kwargs['num_test_negatives']
    seq = np.zeros([seqlen], dtype=np.int32)
    ts_seq = np.zeros([seqlen], dtype=np.int32)
    # 从训练集最后元素开始后往前填充seq
    idx = seqlen - 1
    for interaction in reversed(train_set[uid]):
        seq[idx] = interaction[0]
        ts_seq[idx] = interaction[1]
        idx -= 1
        if idx == -1:
            break
    # rated即该user交互过的所有items id集合，并且验证集的那次交互应该也在里面
    rated = set([x[0] for x in train_set[uid]])
    rated.add(dataset[uid][0][0])  # 最后一次交互的item，即测试集的item也添加带rated
    rated.add(0)  # 填充item id
    # list of test items, beginning with positive and then negative items
    # 候选集，首先添加测试集的item，即最后一次交互的item
    test_item_ids = [dataset[uid][0][0]]
    neg_sampling = kwargs['neg_sampling']  # 负采样策略
    # 均匀采样，从负样本中随机采样num_negatives个负样本
    if neg_sampling == 'uniform':
        for _ in range(num_negatives):
            t = np.random.randint(1, n_items + 1)
            while t in rated:
                t = np.random.randint(1, n_items + 1)
            test_item_ids.append(t)
    # 根据流行度从整个负样本集中采样num_negatives个负样本
    else:
        # 将 rated 转换成列表，然后转换成 numpy 数组，之后对每个元素减去1。这是因为 Python 的索引是从0开始的，而物品ID通常是从1开始的。这样的操作是为了将物品ID转换成数组索引
        zeros = np.array(list(rated)) - 1
        p = kwargs['train_item_popularities'].copy()  # 数据集中每个物品的流行度 这里首先复制这个流行度数组，以避免修改原始数据
        p[zeros] = 0.0  # 将已评价过的物品对应的流行度设置为0，确保在后续的抽样过程中这些物品不会被选择作为负样本，比sampler一直循环那种写法好
        # 将流行度数组 p 中的所有值除以它们的总和，进行归一化处理。这一步确保 p 变成一个有效的概率分布，用于指导随机抽样
        p = p / p.sum()
        # 从1到 n_items + 1（包含所有物品的ID范围）中根据概率分布 p 抽取 num_negatives 个不重复的负样本。replace=False即每个样本都是完全不同的，没有任何重复
        neg_item_ids = np.random.choice(range(1, n_items + 1),
                                        size=num_negatives, p=p, replace=False)
        test_item_ids = test_item_ids + neg_item_ids.tolist()
    out = uid, seq, test_item_ids

    # 追加交互序列和gt的时间特征
    if 'time_dict' in kwargs and kwargs['time_dict'] is not None:
        test_ts = dataset[uid][0][1]  # 最后一次交互，即测试集的item的时间戳
        # 交互序列对应的时间特征序列
        seq_year = np.zeros([seqlen], dtype=np.int32)
        seq_month = np.zeros([seqlen], dtype=np.int32)
        seq_day = np.zeros([seqlen], dtype=np.int32)
        seq_dayofweek = np.zeros([seqlen], dtype=np.int32)
        seq_dayofyear = np.zeros([seqlen], dtype=np.int32)
        seq_week = np.zeros([seqlen], dtype=np.int32)
        seq_hour = np.zeros([seqlen], dtype=np.int32)
        for i, ts in enumerate(ts_seq):  # 遍历seq对应的时间戳序列
            if ts > 0:  # 若时间戳不为0，则提取时间特征，否则直接为0
                seq_year[i], seq_month[i], seq_day[i], seq_dayofweek[i], \
                    seq_dayofyear[i], seq_week[i], seq_hour[i] = kwargs['time_dict'][ts]
        # 测试item对应的时间特征，都是一个值
        test_year, test_month, test_day, test_dayofweek, \
            test_dayofyear, test_week, test_hour = kwargs['time_dict'][test_ts]
        # 将交互序列对应的7个时间特征序列，以及测试item对应的时间特征，追加到out元组中
        out = out + (seq_year, seq_month, seq_day,
                     seq_dayofweek, seq_dayofyear, seq_week, seq_hour,
                     test_year, test_month, test_day,
                     test_dayofweek, test_dayofyear, test_week, test_hour)

    return out

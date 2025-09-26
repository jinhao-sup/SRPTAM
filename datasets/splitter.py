""" 将传入的数据集，划分为训练集、验证集、测试集 """
def split_data(sorted_user_interactions):
    """
    Split data into train/valid/test
    The last item is for test
    The second last item is for validation
    The remaining items are for train
    :param sorted_user_interactions: dictionary, values are tuples (iid, timestamp)，字典格式是userid：元组列表[(itemid,时间)...]
    :return:
    """
    train_set = {}
    valid_set = {}
    test_set = {}  # 都是字典
    for uid, interactions in sorted_user_interactions.items():  # 遍历数据集中所有user的交互记录
        nfeedback = len(interactions)  # 统计数据集中该用户交互的次数
        if nfeedback < 3:  # 若该user的交互次数小于3（0,1,2），不足以按规定的方法划分，则直接将该user的交互记录作为训练集
            train_set[uid] = interactions
            valid_set[uid] = []
            test_set[uid] = []
        else:  # 将该user的交互记录添加到训练集、验证集、测试集
            train_set[uid] = interactions[:-2]
            valid_set[uid] = [interactions[-2]]
            test_set[uid] = [interactions[-1]]
    # 返回划分后的训练集，验证集，测试集，格式均为userid：元组列表[(itemid,时间)...]，只是验证集，测试集，每个user的列表只有一个元组
    return train_set, valid_set, test_set

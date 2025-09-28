import numpy as np
import pandas as pd
from tqdm import tqdm


""" 将时间戳集合转换为7种时间特征，年、月、日、一周中的第几天、一年中的第几天、week（一年的第几周）和小时数 """
def ts_to_high_level_tempo(time_set):
    df = pd.DataFrame(list(time_set), columns=['ts'])
    df['dt'] = pd.to_datetime(df['ts'], unit='s')
    df = df.sort_values(by=['ts'])  # 对传入的时间集合进行排序
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'], df['week'], df['hour'] = zip(
        *df['dt'].map(lambda x: [x.year, x.month, x.day, x.dayofweek, x.dayofyear, x.week, x.hour]))
    df['year'] -= df['year'].min()
    df['year'] += 1  # year列的值减去最小年份，再加1，使年份从1开始
    df['dayofweek'] += 1  # 将 dayofweek 列的每个值增加1，使星期几从1开始
    res = {}  # 结果字典
    for _, row in df.iterrows():  # 遍历dataframe
        ts = row['ts']
        year = row['year']
        month = row['month']
        day = row['day']
        dayofweek = row['dayofweek']
        dayofyear = row['dayofyear']
        week = row['week']
        hour = row['hour']
        # 结果字典格式为 {ts: (year, month, day, dayofweek, dayofyear, week, hour)}
        res[ts] = year, month, day, dayofweek, dayofyear, week, hour
    return res


""" 将每个时间戳对应的时间特征，均标准化到[0,1]区间，并转换为一个数组"""
def ts_to_carca_tempo(time_set):
    df = pd.DataFrame(list(time_set), columns=['ts'])
    df['dt'] = pd.to_datetime(df['ts'], unit='s')
    df = df.sort_values(by=['ts'])
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'], df['week'] = zip(
        *df['dt'].map(lambda x: [x.year, x.month, x.day, x.dayofweek, x.dayofyear, x.week]))
    df['year'] -= df['year'].min()
    df['year'] /= df['year'].max()
    df['month'] /= 12
    df['day'] /= 31
    df['dayofweek'] /= 7
    df['dayofyear'] /= 365
    df['week'] /= 4
    res = {}
    for _, row in df.iterrows():
        ts = row['ts']
        year = row['year']
        month = row['month']
        day = row['day']
        dayofweek = row['dayofweek']
        dayofyear = row['dayofyear']
        week = row['week']
        res[ts] = np.array([year, month, day, dayofweek, dayofyear, week])
    return res


""" 对一个时间戳列表进行规范化处理。 """
def normalize_time(time_list):
    time_diff = set()
    for i in range(len(time_list) - 1):
        if time_list[i + 1] - time_list[i] != 0:
            time_diff.add(time_list[i + 1] - time_list[i])
    # 确定时间尺度 time_scale，如果所有时间戳相同，则设置=1，否则设置为最小时间差
    if len(time_diff) == 0:
        time_scale = 1
    else:
        time_scale = min(time_diff)
    time_min = min(time_list)  # 时间列表中最小的时间戳
    # 计算每个时间戳（减去最小时间戳后）相对于时间尺度（最小时间间隔）的倍数，+1使倍数从1开始。即得到规范化时间
    res = [int(round((t - time_min) / time_scale) + 1) for t in time_list]
    return res


""" 计算（一个user）任意两次时间戳之间的相对时间间隔矩阵，并限制最大时间间隔为time_span """
def compute_time_matrix(time_seq, time_span):
    """
    Compute temporal relation matrix for the given time sequence
    :param time_seq: Timestamp sequence
    :param time_span: threshold  限制的最大时间间隔阈值
    :return:
    """
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if time_span is not None and span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


# 计算训练数据中所有user的任意两次时间戳之间的相对时间间隔矩阵，并限制最大时间间隔为time_span
def compute_relation_matrix(user_train, usernum, maxlen, time_span):
    """
    Compute temporal relation matrix for all users
    """
    data_train = dict()
    # 开始遍历每个user的训练数据
    for user in tqdm(range(1, usernum + 1),
                     desc='Preparing temporal relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)  # 时间序列，限制长度为maxlen
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):  # 倒序遍历用户训练数据，但是不包括最后一次
            time_seq[idx] = i[1]  # 取出该次交互的时间戳
            idx -= 1
            if idx == -1:
                break
        # 将该user的训练数据的最后maxlen次交互（逆序，不含训练集最后一次）的时间序列，用于计算相对时间间隔矩阵
        data_train[user] = compute_time_matrix(time_seq, time_span)
    return data_train

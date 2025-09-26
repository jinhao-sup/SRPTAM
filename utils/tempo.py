import numpy as np
import pandas as pd
from tqdm import tqdm


""" tempo即temporal缩写。将时间戳集合转换为一组更高级的时间特征，如年、月、日、一周中的第几天、一年中的第几天、week（应该是一年的第几周）和小时数 """
def ts_to_high_level_tempo(time_set):
    # 将输入的时间戳集合 time_set 转换为列表，然后创建一个包含单个列ts的DataFrame。有许多行，每行有若干列属性
    df = pd.DataFrame(list(time_set), columns=['ts'])
    # 把ts列中的时间戳转换为日期时间格式datetime对象，作为新的列dt，这里的 unit='s' 参数表明时间戳的单位是秒
    df['dt'] = pd.to_datetime(df['ts'], unit='s')
    df = df.sort_values(by=['ts'])  # 对传入的时间集合进行排序
    """
    map() 函数对 dt 列中的每个元素应用一个函数,lambda 函数接受一个日期时间对象 x 并返回一个列表，包含该x对应的年，月份，日等时间信息
    * 解包操作符确实是用来将每个 x 对应的年份、月份、日期等时间组件进行解包到 zip() 函数
    当 zip() 函数应用于解包后的列表时，它会产生一系列元组，每个元组包含从每个列表中相同位置的元素
    例子：
    zip(
    [2021, 1, 1, 4, 1, 53, 0], 
    [2021, 2, 1, 0, 32, 5, 0])，得到
    所有的年份：(2021, 2021)
    所有的月份：(1, 2)
    所有的日期：(1, 1)，此时df的列为[ts，dt，year，month，day，dayofweek，dayofyear，week，hour]
    """
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'], df['week'], df['hour'] = zip(
        *df['dt'].map(lambda x: [x.year, x.month, x.day, x.dayofweek, x.dayofyear, x.week, x.hour]))
    df['year'] -= df['year'].min()
    df['year'] += 1  # year列的值减去最小年份，再加1，使年份从1开始
    df['dayofweek'] += 1  # 将 dayofweek 列的每个值增加1，使星期几从1开始（通常在Python中，星期一从0开始）
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
        # 结果字典格式为 {ts: (year, month, day, dayofweek, dayofyear, week, hour)}，即每个键对应一个元组，即每个时间戳对应的时间特征
        res[ts] = year, month, day, dayofweek, dayofyear, week, hour
    return res


""" 将每个时间戳对应的时间特征，均标准化到[0,1]区间，并转换为一个数组，
结果字典格式为 {ts: np.array([year, month, day, dayofweek, dayofyear, week])}，但是这个函数并没有被调用"""
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


""" 对一个时间戳列表进行规范化处理。但是没有调用 """
def normalize_time(time_list):
    time_diff = set()  # 存储时间列表中相邻元素之间的非零时间差
    # 如果相邻时间戳之间的差异不为零，则将该差值添加到 time_diff 集合,使用集合保证存储的时间差是唯一的
    for i in range(len(time_list) - 1):
        if time_list[i + 1] - time_list[i] != 0:
            time_diff.add(time_list[i + 1] - time_list[i])
    # 确定时间尺度 time_scale，如果所有时间戳相同，则设置=1，否则设置为最小时间差
    if len(time_diff) == 0:
        time_scale = 1
    else:
        time_scale = min(time_diff)
    time_min = min(time_list)  # 时间列表中最小的时间戳
    # round对商进行四舍五入
    # 计算每个时间戳（减去最小时间戳后）相对于时间尺度（最小时间间隔）的倍数，+1使倍数从1开始。即得到所谓的规范化时间
    res = [int(round((t - time_min) / time_scale) + 1) for t in time_list]
    return res


""" 计算（一个user）任意两次时间戳之间的相对时间间隔矩阵，并限制最大时间间隔为time_span。只在下面的函数中调用，实际还是没有使用 """
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


# 计算训练数据中所有user的任意两次时间戳之间的相对时间间隔矩阵，并限制最大时间间隔为time_span。没有使用
def compute_relation_matrix(user_train, usernum, maxlen, time_span):
    """
    Compute temporal relation matrix for all users  这里的参数解释是自己写的，原来没有
    :param user_train:  所有user的训练数据。字典或矩阵
    :param usernum:
    :param maxlen: 应该是考虑的序列最大长度
    :param time_span: 限制的最大时间间隔阈值
    :return:
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

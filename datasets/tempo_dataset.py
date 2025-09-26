import os
import pickle
from tqdm import tqdm
from datasets.dataset import Dataset
from utils.tempo import ts_to_high_level_tempo  # 将时间戳集合转换为一组更高级的时间特征


# 实际就是data.py上加了两个函数功能
class TempoDataset(Dataset):  # TempoDataset是一个抽象类，用于定义tempo数据集的通用方法，继承自Dataset类
    def __init__(self, params):
        super(TempoDataset, self).__init__(params)  # 调用基类（Dataset）的构造函数
        self.model_params = params['training']['model']['params']
        self.dataset_params = params['dataset']
        self.cache_params = params['cache']
        # 统计所有训练实例，验证集，测试集出现的时间戳集合，然后将时间戳转化为对应的7种时间特征，{ts: (year, month, day, dayofweek, dayofyear, week, hour)}
        # 以字典形式设置到self.data['time_dict']，并保存到文件timedict.pkl
        # fetch additional temporal data
        self._fetch_tempo_data()

    # def _fetch_tempo_data(self):  # 由具体的tempo数据集子类实现
    #     raise NotImplementedError('_fetch_tempo_data method should be '
    #                               'implemented in concrete model')

    # 对所有训练实例，测试和验证数据的时间戳转换为7种时间特征，以字典的形式保存，并设置到类的data['time_dict'] 属性中
    # 将时间集合中的时间戳转换为时间特征，time_dict字典，{ts：年、月、日、一周中的第几天、一年中的第几天、week（应该是一年的第几周）和小时数}
    def _fetch_tempo_data(self):
        self.logger.debug('Fetch additional temporal data')
        tempo_data_path = os.path.join(self.cache_path, self.model_name)  # 保存时间数据的文件夹
        if not os.path.exists(tempo_data_path):
            os.makedirs(tempo_data_path)
        timedict_path = os.path.join(tempo_data_path, 'timedict.pkl')  # 时间数据，即时间字典的存储路径
        # 如果未对时间戳进行特征处理
        if not os.path.exists(timedict_path):
            train_set = self.data['train_set']
            valid_set = self.data['valid_set']
            test_set = self.data['test_set']
            """ 对训练集中的训练实例，验证集，测试集时间戳统计到集合，并转换为时间特征保存到文件，作用就是为了方便将时间戳序列转换为对应的7个时间特征序列，在mojito_sampler中使用 """
            # 对于训练集，只统计给定交互索引（所有训练实例）的时间戳。
            time_set = self._time_set(train_set, indexes=self.data['train_interaction_indexes'])
            # 然后将其合并验证集和测试集的时间集合
            time_set = time_set.union(self._time_set(valid_set), self._time_set(test_set))
            # 将时间集合中的时间戳转换为时间特征，time_dict字典，{ts：年、月、日、一周中的第几天、一年中的第几天、week（应该是一年的第几周）和小时数}
            # {ts: (year, month, day, dayofweek, dayofyear, week, hour)}
            time_dict = ts_to_high_level_tempo(time_set=time_set)
            pickle.dump(time_dict, open(timedict_path, 'wb'))
        # 已对时间集的时间戳进行特征处理，直接从文件中读取
        else:
            time_dict = pickle.load(open(timedict_path, 'rb'))
        self.data['time_dict'] = time_dict  # time_dict 保存到类的 data 属性中
        self.logger.debug('Finish fetch additional temporal data')

    # 两种功能：1.统计传入数据集的去重时间戳集合，2.如果给定了索引列表，则只记录每个索引及前面L次交互去重的时间戳集合
    # 这个函数功能不是很清晰的原因是，融合了两个不同的功能，一个是统计整个数据集的时间戳集合，另一个是统计给定索引列表的前L次交互的时间戳集合
    def _time_set(self, dataset, indexes=None):
        ts_set = set()
        n_users = len(dataset)  # 传入数据集的用户数量
        cnt = 0
        """ 如果传入验证集或测试集，则统计数据集中出现过的时间戳 """
        if indexes is None:
            for _, interactions in dataset.items():
                for interaction in interactions:
                    ts_set.add(interaction[1])  # 将数据集中的时间戳添加到集合中
                # 维护一个计数器 cnt 来追踪已处理的用户数量，并定期通过日志输出进度
                cnt += 1
                if cnt % 5000 == 0 or cnt == n_users:
                    self.logger.debug(f'----> {cnt} / {n_users} users')
        # 如果传入的是训练集，同时会传入train_interaction_indexes（uid，nextid）即训练实例抽样的索引，所以只会统计训练集中训练实例出现过的时间戳
        else:
            for uid, nxt_idx in tqdm(indexes, desc='Generate time set'):
                idx = self.seqlen - 1
                for _, ts in reversed(dataset[uid][:nxt_idx]):
                    if ts > 0:
                        ts_set.add(ts)
                    idx -= 1
                    if idx == -1:
                        break
                _, ts = dataset[uid][nxt_idx]
                ts_set.add(ts)
        return ts_set

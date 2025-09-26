import os
import pickle
import numpy as np
import pandas as pd
from utils.logging import get_logger
from datasets.splitter import split_data


# TODO：这个Dataset是baseline常用，优化后的
class Dataset:
    """
    Dataset
    """

    def __init__(self, params):  # 构造时需要传入json配置文件对象，从下面的调用分析也是一个字典
        cache_params = params['cache']  # 缓存设置，路径等，json中的xx_interactions大概率也是路径
        self.dataset_params = params['dataset']  # 数据集参数
        self.eval_params = params['eval']  # 评估参数

        dataset_name = self.dataset_params['name']
        self.model_name = params['training']['model']['name']
        self.u_ncore = self.dataset_params.get('u_ncore', 1)
        self.i_ncore = self.dataset_params.get('i_ncore', 1)
        self.item_type = self.dataset_params.get('item_type', 'item')  # 数据集中物品类型的标识，用于cache文件中路径拼接

        self.cache_path = cache_params['path']
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        # 每个训练或测试实例，考虑的交互序列长度，默认为 50。实际上训练时也会迭代子序列，因为有pos列表指示每个子序列的gt
        self.seqlen = params['training']['model']['params'].get('seqlen', 50)
        self.num_test_negatives = params['training'].get('num_test_negatives', 100)  # 测试时负采样的个数，默认为 100
        self.num_valid_users = params['training'].get('num_valid_users', -1)  # 限定在验证过程中应当考虑的用户数。则默认值为 -1，可能表示不限制用户数
        # 用于训练、验证和测试数据的文件名前缀
        train_interactions_prefix = cache_params['train_interactions']
        valid_interactions_prefix = cache_params['valid_interactions']
        test_interactions_prefix = cache_params['test_interactions']

        """
        指定在从用户交互数据中提取训练样本时应使用的步长。如果未在配置中明确设置，则默认值为 0，这意味着与sasrec中的一致，只使用训练集中每个user的最后L次交互构造训练实例
        在每个user的交互序列中，每隔100个交互采样一个。
        """
        self.samples_step = self.dataset_params.get('samples_step', 0)

        """ 对ml1m和amzb数据集进行处理。都是在拼接各种数据的完整路径，没有实际处理操作 """
        self.common = f'{self.u_ncore}core-users_{self.i_ncore}core-items'  # 拼接公共路径
        """ 拼接训练、验证和测试数据的完整路径。使用pkl格式，因为存储的是字典uid：[交互的元组列表(itemid,时间)...]，而不是具体的交互记录，方便直接加载为python对象。在cache文件夹下 """
        self.train_interactions_path = os.path.join(self.cache_path,
                                                    f'{train_interactions_prefix}_{self.common}.pkl')
        self.valid_interactions_path = os.path.join(self.cache_path,
                                                    f'{valid_interactions_prefix}_{self.common}.pkl')
        self.test_interactions_path = os.path.join(self.cache_path, f'{test_interactions_prefix}_{self.common}.pkl')
        # 对kcore处理后的数据集进行id重映射的数据集保存路径
        # 主要是将user和item的id映射为内部id（从1开始），然后保存为csv文件。在exp\data文件夹下，如ml1m_mapped_internal-ids_interactions_10core-users_5core-items（未排序）
        self.kcore_data_path = os.path.join(self.dataset_params['path'],
                                            f'{dataset_name}_mapped_internal-ids_interactions_{self.common}.csv')
        """
        保存的是去重的原始（未映射到从1开始）uid和iid列表（很多原始的uid和iid都是自字符串，而不是数值，作用就是统计去重的user和item总数）
        # npz 文件是一个多个数组数据的压缩文件格式，它是由 numpy 提供的一种数据存储格式。在处理大量的科学计算数据时，npz 是一个非常好的方式。
        当我们需要处理多个数组数据时，可以使用 npz 文件将它们压缩成一个文件，这样可以方便的存储和传输这些数组
        """
        self.entities_path = os.path.join(self.dataset_params['path'], f'{dataset_name}_entities_{self.common}.npz')
        """ 存储训练实例索引的文件路径。这些索引用于从用户的交互历史中选取训练实例 """
        self.train_interaction_indexes_path = os.path.join(self.cache_path,
                                                           f'train_interactions_indexes_{self.common}_samples-step{self.samples_step}.pkl')

        # 对数据集进行处理
        self.n_epochs = params['training'].get('num_epochs', 100)  # 训练迭代次数，默认为100
        self.logger = get_logger()
        # fetch data
        self._fetch_data()

    # Dataset类的对数据处理的主要操作函数，调用下面的方法，上面的__init__只是在拼接各种数据保存路径
    # 将训练，验证，测试集，各训练实例采样元组列表（uid，nextid），交互统计信息，训练集items流行度，原始uid和iid列表，训练交互索引，user和item总数，最先时间戳等保存到dataset对象的self.data字典中
    def _fetch_data(self):
        """
        Fetch data
        :return: data dictionary
        """
        #  检查训练集，验证集，测试集，数据实体(数据)是否存在，如果某个文件不存在，则进入数据处理流程
        if not os.path.exists(self.train_interactions_path) or not \
                os.path.exists(self.valid_interactions_path) or not \
                os.path.exists(self.test_interactions_path) or not \
                os.path.exists(self.entities_path):
            #  这里的作用不是对数据集进行kcore处理，而是kcore处理后判断是否将user和item的id映射为从1开始，若没有则处理
            # 保存到如/exp/data/ml1m/ml1m_mapped_internal-ids_interactions_10core-users_5core-items.csv
            if not os.path.exists(self.kcore_data_path):
                # 1. read original k-core
                # 读出来的数据已经经过kcore处理，但是还没有重新映射user和item id有可能是字符串，即/exp/data/ml1m下的ml1m_interactions_10core-users_5core-items.csv
                data = self._fetch_original_kcore_interactions()
                # 2. map original data to internal ids
                self.logger.debug(f'Map to internal ids')
                # 将数据中的原始ID映射到内部ID，data使用新的列名user，item从1开始，同时返回去重的原始uid和iid列表（不一定是数值，有可能是字符串）
                data, user_ids, item_ids = self._map_internal_ids(data)
                data.to_csv(self.kcore_data_path, sep=',', header=True, index=False)  # 保存kcore处理后的数据集到csv文件
                # 所以entities_path保存的是去重的原始uid和iid列表（不一定是数值，可能是加密后的字符串），作用就是统计去重的user和item总数
                np.savez(self.entities_path, user_ids=user_ids,
                         item_ids=item_ids)
            else:  # 已经对kcore处理后的整个数据集进行了id的重新映射
                self.logger.debug(f'Load data from {self.kcore_data_path}')
                data = pd.read_csv(self.kcore_data_path)  # 读取kcore处理后的数据集
                # 所以entities_path保存的是去重的原始uid和iid列表（不一定是数值，可能是加密后的字符串），作用就是统计去重的user和item总数
                entities = np.load(self.entities_path, allow_pickle=True)
                user_ids = entities['user_ids']  # 读取旧的去重user_ids和item_ids列表
                item_ids = entities['item_ids']
            # 3. sort each user interactions by timestamps
            self.logger.debug('Sort user interactions by timestamps')
            sorted_user_interactions = dict()
            grouped_data = data.groupby('user')  # 将处理后的数据集data按照user分组
            for uid, interactions in grouped_data:  # 遍历每个用户的交互数据
                # 将每个用户的交互数据按照时间戳进行排序，添加到字典sorted_user_interactions，格式是userid：元组列表[(itemid,时间)...]
                interactions = interactions.sort_values(by=['timestamp'])
                iids = interactions['item'].tolist()
                times = interactions['timestamp'].tolist()
                sorted_user_interactions[uid] = list(zip(iids, times))
            # 4. split data
            self.logger.debug('Split data into train/val/test')
            train_set, valid_set, test_set = split_data(sorted_user_interactions)  # 将排序好的交互数据进行划分为train/val/test
            # 5. store data to cache for next use
            self.logger.debug('Save data to cache')
            # 保存划分后的数据集到pickle文件，所以xx_interactions_path保存的是train/val/test的有序交互数据，都是字典，键是userid，值是元组列表[(itemid,时间)...]
            # 并且保存为pkl文件，而不是csv文件，因为不是交互记录，而是字典。位于/cache下
            pickle.dump(train_set, open(self.train_interactions_path, 'wb'))
            pickle.dump(valid_set, open(self.valid_interactions_path, 'wb'))
            pickle.dump(test_set, open(self.test_interactions_path, 'wb'))
        # 训练集，验证集，测试集，数据实体(数据)已经存在，直接读取
        else:
            train_set = pickle.load(open(self.train_interactions_path, 'rb'))
            valid_set = pickle.load(open(self.valid_interactions_path, 'rb'))
            test_set = pickle.load(open(self.test_interactions_path, 'rb'))
            # 读取旧的去重user_ids和item_ids列表（不一定是数值，有可能是字符串）
            entities = np.load(self.entities_path, allow_pickle=True)
            user_ids = entities['user_ids']
            item_ids = entities['item_ids']

        """ 对每个用户整个训练交互内的训练实例索引进行采样，交给sampler在训练集中提取训练实例（为了产生更多的训练实例，不同于sasrec只使用每个user训练集最后L次交互构造训练实例） """
        # train interaction indexes within each user
        # 如果训练实例采样索引数据不存在，开始提取。位于/cache下
        if not os.path.exists(self.train_interaction_indexes_path):
            self.logger.debug('Extract training interaction indexes')
            train_interaction_indexes = []  # 每个元组的元素是(uid,nxt_idx)，uid是userid，nxt_idx是每个训练实例pos中最后一个item在交互序列中的下标
            for uid, interactions in train_set.items():  # 遍历训练集中每个user的有序交互数据，train_set是字典，interactions是元组列表[(itemid,时间)...]
                # （训练集）该user最后一次交互在interactions中的索引
                last_idx = len(interactions) - 1
                """ 这几行的作用就是从训练集的最后一个交互开始，往前每samples_step步取一次交互的索引，但不超过前面的L个 """
                # 所以train_interaction_indexes是一个元组列表，保存每个（user，训练集最后一次交互的索引），每个user唯一
                train_interaction_indexes.append((uid, last_idx))
                if self.samples_step > 0:  # 如果定义了非零的采样步长
                    # # 计算从最后一个交互向前的采样索引列表，直到达到序列长度限制
                    """
                    range(start, stop, step) 函数：这是 Python 内置的函数，用于生成一个整数序列。从 start 开始，到 stop 结束（不包含 stop），步长为 step
                    当步长step 为负数时，序列会递减。生成的数字满足条件 start > current > stop。
                    假设：last_idx = 20，self.seqlen = 5，self.samples_step = 3
                    生成的 offsets 列表为：[20, 17, 14, 11, 8, 5]
                    就是说各个实例的结束索引（指向gt）前面的交互个数不会少于seqlen个，实际就是从后往前最后一个结束索引前面还会有seqlen个交互
                    """
                    offsets = list(range(last_idx, self.seqlen - 1, -self.samples_step))
                    for offset in offsets:
                        # # 将用户ID和计算出的各个偏移索引作为元组添加到列表中，offset即从训练集最后往前每samples_step步记录交互的索引
                        train_interaction_indexes.append((uid, offset))
            """
            所以train_interaction_indexes保存的是（uid，交互索引）元组列表，交互的索引即从训练集每个user的最后一个交互开始，往前每samples_step步取一次交互的索引，但不超过前面的L个（防止取前L个时越界）
            将train_interaction_indexes保存到pickle文件
            """
            pickle.dump(train_interaction_indexes, open(self.train_interaction_indexes_path, 'wb'))
        # 如果训练实例采样索引数据存在，直接读取
        else:
            train_interaction_indexes = pickle.load(open(self.train_interaction_indexes_path, 'rb'))

        # 在日志中打印数据集信息
        self.logger.info(f'Number of users: {len(user_ids)}')
        self.logger.info(f'Number of items: {len(item_ids)}')
        # 统计整个数据集的总交互次数，+2是因为算上验证集和测试集的2个交互
        n_interactions = np.sum([len(interactions) + 2 for _, interactions in train_set.items()])
        self.logger.info(f'Number of interactions: {n_interactions}')
        # 计算数据集的密度，即总交互次数除以（总用户数乘以总物品数）
        self.logger.info(f'Density: {n_interactions / (len(user_ids) * len(item_ids)):3.5f}')

        """ 将训练，验证，测试集，各训练实例采样元组列表（uid，nextid），交互统计信息，训练集items流行度，原始uid和iid列表，训练交互索引，user和item总数，最先时间戳等保存到data字典中 """
        self.data = {
            'train_set': train_set,
            'valid_set': valid_set,
            'test_set': test_set,
            'train_interaction_indexes': train_interaction_indexes,
            'user_ids': user_ids,  # 去重的原始用户id列表（可能不是数值，用于统计去重的user总数）
            'item_ids': item_ids,  # 去重的原始物品id列表
            'n_users': len(user_ids),
            'n_items': len(item_ids),
            'num_test_negatives': self.num_test_negatives,
            'num_valid_users': self.num_valid_users
        }

    """ 返回将完整数据集经过kcore处理后的整个数据集（还没重新映射user和item的id），还没划分训练，验证，测试集 """

    def _fetch_original_kcore_interactions(self):
        """ kcore处理后的数据集路径（还未重新映射user和item的id），位于exp/data/下 """
        org_kcore_data_path = os.path.join(
            self.dataset_params['path'],
            f'{self.dataset_params["name"]}_interactions_'
            f'{self.u_ncore}core-users_{self.i_ncore}core-items.csv')

        # 检查kcore处理后的数据集是否存在，如果不存在则进行kcore处理。实际上不会走这个if，因为在kcore_interactions.py已经对数据集进行了kcore处理
        # 所以不会走_fetch_interactions函数，这个函数并没有正确拼接原始数据集的完整路径（已修改源代码）
        if not os.path.exists(org_kcore_data_path):
            # 1. read original data
            data = self._fetch_interactions()
            # 2. select only interested columns
            """这里比源代码多加rating列，防止少保存一列，即使不使用"""
            data = data[['org_user', f'org_{self.item_type}', 'rating', 'timestamp']]
            # 3. kcore preprocessing
            self.logger.debug(f'Get {self.u_ncore}-core user, '
                              f'{self.i_ncore}-core item data')
            data = self._kcore(data, self.u_ncore, self.i_ncore)
            data.to_csv(org_kcore_data_path, sep=',', header=True, index=False)  # 对整个数据集kcore处理后保存到csv文件
        # 如果已经对整个数据集进行了kcore处理，则直接读取
        else:
            self.logger.debug(f'Read original k-core data from {org_kcore_data_path}')
            data = pd.read_csv(org_kcore_data_path)
        return data

    # 从指定的文件路径中获取用户的交互数据。它的主要功能是读取CSV文件中的交互数据（一条一条的交互数据，不是字典有序交互数据），在上面的函数调用
    def _fetch_interactions(self):
        """
        Fetch interactions from file
        """
        """ 修改源代码，拼接正确的原始数据集路径，从项目文件夹开始（参考了kcore_interactions.py的写法） """
        user_interactions_path = os.path.join(self.dataset_params['path'],
                                              f'{self.dataset_params["interactions"]}.{self.dataset_params["file_format"]}')

        self.logger.info(f'Fetch information from {user_interactions_path}')
        # 列名由配置文件中的 col_names 参数指定
        data = pd.read_csv(user_interactions_path, names=self.dataset_params['col_names'])
        return data

    # 从提供的数据集中生成一个k-core数据集，确保每个用户至少与 u_ncore 个不同的物品有交互，同时每个物品至少被 i_ncore 个不同的用户交互
    # 实际上就是kcore_interactions.py中的kcore代码，只是这里定义为一个私有方法，而没有调用kcore_interactions中的方法
    def _kcore(self, data, u_ncore, i_ncore):
        """
        Preprocessing data to get k-core dataset.
        Each user has at least u_ncore items in his preference
        Each item is interacted by at least i_ncore users
        :param data:
        :param u_ncore: min number of interactions for each user
        :param i_ncore: min number of interactions for each item
        """
        if u_ncore <= 1 and i_ncore <= 1:  # 都小于等于1，意味着没有核心性的限制，函数直接返回原始数据
            return data

        def filter_user(df):  # 过滤掉用户交互数小于 u_ncore 的用户
            """ Filter out users less than u_ncore interactions """
            tmp = df.groupby(['org_user'], as_index=False)[f'org_{self.item_type}'].count()
            tmp.rename(columns={f'org_{self.item_type}': 'cnt_item'},
                       inplace=True)
            df = df.merge(tmp, on=['org_user'])
            df = df[df['cnt_item'] >= u_ncore].reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)
            return df

        def filter_item(df):  # 过滤掉物品被交互次数小于 i_ncore 的物品
            """ Filter out items less than u_ncore interactions """
            tmp = df.groupby([f'org_{self.item_type}'], as_index=False)['org_user'].count()
            tmp.rename(columns={'org_user': 'cnt_user'},
                       inplace=True)
            df = df.merge(tmp, on=[f'org_{self.item_type}'])
            df = df[df['cnt_user'] >= i_ncore].reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)
            return df

        # because of repeat consumption, just count 1 for each user-item pair  消除重复的用户-物品对，因为重复消费在这里只计算一次
        unique_data = data[['org_user', f'org_{self.item_type}']].drop_duplicates()
        while 1:
            unique_data = filter_user(unique_data)
            unique_data = filter_item(unique_data)
            chk_u = unique_data.groupby('org_user')[f'org_{self.item_type}'].count()
            chk_i = unique_data.groupby(f'org_{self.item_type}')['org_user'].count()
            if len(chk_i[chk_i < i_ncore]) <= 0 and len(chk_u[chk_u < u_ncore]) <= 0:
                break

        unique_data = unique_data.dropna()
        data = pd.merge(data, unique_data, on=['org_user', f'org_{self.item_type}'])
        return data

    # 将数据集中的原始用户和物品标识符（ID）映射到内部使用的连续整数ID。data使用新的列名user，item即uid和iid从1开始，同时返回去重的原始uid和iid列表（不一定是数值，有可能是字符串）
    def _map_internal_ids(self, data):
        # 将名为 'org_{self.item_type}' 的列重命名为 'org_item'（因为对于lfm1b等数据集没有item，是使用track作为item_type）
        data.rename(columns={f'org_{self.item_type}': 'org_item'}, inplace=True)
        # 提取并去重用户ID (org_user) 和物品ID (org_item)，然后转换为 NumPy 数组
        user_ids = data['org_user'].drop_duplicates().to_numpy()
        item_ids = data['org_item'].drop_duplicates().to_numpy()
        # map to internal ids   遍历去重后的用户和物品ID，为每个ID分配一个从1开始的新索引
        user_id_map = {uid: idx + 1 for idx, uid in enumerate(user_ids)}
        item_id_map = {iid: idx + 1 for idx, iid in enumerate(item_ids)}
        # data 中的 org_user 列中的每个值替换为一个新的值（x即旧id），并作为新的列user，item列同理
        data.loc[:, 'user'] = data.org_user.apply(lambda x: user_id_map[x])
        data.loc[:, 'item'] = data.org_item.apply(lambda x: item_id_map[x])
        # 移除原始的org_user, org_item列
        data = data.drop(columns=['org_user', f'org_item'])
        return data, user_ids, item_ids

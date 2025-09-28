import os
import pickle
import numpy as np
import pandas as pd
from utils.logging import get_logger
from datasets.splitter import split_data


class Dataset:
    """
    Dataset
    """

    def __init__(self, params):
        cache_params = params['cache']
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
    
        self.seqlen = params['training']['model']['params'].get('seqlen', 50)
        self.num_test_negatives = params['training'].get('num_test_negatives', 100)
        self.num_valid_users = params['training'].get('num_valid_users', -1)
        # 用于训练、验证和测试数据的文件名前缀
        train_interactions_prefix = cache_params['train_interactions']
        valid_interactions_prefix = cache_params['valid_interactions']
        test_interactions_prefix = cache_params['test_interactions']

        self.samples_step = self.dataset_params.get('samples_step', 0)

        self.common = f'{self.u_ncore}core-users_{self.i_ncore}core-items'  # 拼接公共路径
        self.train_interactions_path = os.path.join(self.cache_path,
                                                    f'{train_interactions_prefix}_{self.common}.pkl')
        self.valid_interactions_path = os.path.join(self.cache_path,
                                                    f'{valid_interactions_prefix}_{self.common}.pkl')
        self.test_interactions_path = os.path.join(self.cache_path, f'{test_interactions_prefix}_{self.common}.pkl')
        # 对kcore处理后的数据集进行id重映射的数据集保存路径
        self.kcore_data_path = os.path.join(self.dataset_params['path'],
                                            f'{dataset_name}_mapped_internal-ids_interactions_{self.common}.csv')

        self.entities_path = os.path.join(self.dataset_params['path'], f'{dataset_name}_entities_{self.common}.npz')
        self.train_interaction_indexes_path = os.path.join(self.cache_path,
                                                           f'train_interactions_indexes_{self.common}_samples-step{self.samples_step}.pkl')

        # 对数据集进行处理
        self.n_epochs = params['training'].get('num_epochs', 100)  # 训练迭代次数，默认为100
        self.logger = get_logger()
        # fetch data
        self._fetch_data()

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
            # 保存到如/exp/data/ml1m/ml1m_mapped_internal-ids_interactions_10core-users_5core-items.csv
            if not os.path.exists(self.kcore_data_path):
                # 1. read original k-core
                data = self._fetch_original_kcore_interactions()
                # 2. map original data to internal ids
                self.logger.debug(f'Map to internal ids')
                data, user_ids, item_ids = self._map_internal_ids(data)
                data.to_csv(self.kcore_data_path, sep=',', header=True, index=False)  # 保存kcore处理后的数据集到csv文件
                np.savez(self.entities_path, user_ids=user_ids,
                         item_ids=item_ids)
            else:  # 已经对kcore处理后的整个数据集进行了id的重新映射
                self.logger.debug(f'Load data from {self.kcore_data_path}')
                data = pd.read_csv(self.kcore_data_path)  # 读取kcore处理后的数据集
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
            pickle.dump(train_set, open(self.train_interactions_path, 'wb'))
            pickle.dump(valid_set, open(self.valid_interactions_path, 'wb'))
            pickle.dump(test_set, open(self.test_interactions_path, 'wb'))
        # 训练集，验证集，测试集，数据实体(数据)已经存在，直接读取
        else:
            train_set = pickle.load(open(self.train_interactions_path, 'rb'))
            valid_set = pickle.load(open(self.valid_interactions_path, 'rb'))
            test_set = pickle.load(open(self.test_interactions_path, 'rb'))
            entities = np.load(self.entities_path, allow_pickle=True)
            user_ids = entities['user_ids']
            item_ids = entities['item_ids']

        # train interaction indexes within each user
        if not os.path.exists(self.train_interaction_indexes_path):
            self.logger.debug('Extract training interaction indexes')
            train_interaction_indexes = []
            for uid, interactions in train_set.items():
                last_idx = len(interactions) - 1
                train_interaction_indexes.append((uid, last_idx))
                if self.samples_step > 0:  # 如果定义了非零的采样步长
                    offsets = list(range(last_idx, self.seqlen - 1, -self.samples_step))
                    for offset in offsets:
                        train_interaction_indexes.append((uid, offset))

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
        self.logger.info(f'Density: {n_interactions / (len(user_ids) * len(item_ids)):3.5f}')

        self.data = {
            'train_set': train_set,
            'valid_set': valid_set,
            'test_set': test_set,
            'train_interaction_indexes': train_interaction_indexes,
            'user_ids': user_ids,
            'item_ids': item_ids,
            'n_users': len(user_ids),
            'n_items': len(item_ids),
            'num_test_negatives': self.num_test_negatives,
            'num_valid_users': self.num_valid_users
        }

    def _fetch_original_kcore_interactions(self):
        org_kcore_data_path = os.path.join(
            self.dataset_params['path'],
            f'{self.dataset_params["name"]}_interactions_'
            f'{self.u_ncore}core-users_{self.i_ncore}core-items.csv')

        if not os.path.exists(org_kcore_data_path):
            # 1. read original data
            data = self._fetch_interactions()
            # 2. select only interested columns
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

    def _fetch_interactions(self):
        """
        Fetch interactions from file
        """
        user_interactions_path = os.path.join(self.dataset_params['path'],
                                              f'{self.dataset_params["interactions"]}.{self.dataset_params["file_format"]}')
        self.logger.info(f'Fetch information from {user_interactions_path}')
        data = pd.read_csv(user_interactions_path, names=self.dataset_params['col_names'])
        return data

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

        # because of repeat consumption, just count 1 for each user-item pair
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

    def _map_internal_ids(self, data):
        data.rename(columns={f'org_{self.item_type}': 'org_item'}, inplace=True)
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


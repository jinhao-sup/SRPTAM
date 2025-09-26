import os
import json
import logging  # 不是mojito包下自定义日志类，而是python提供的日志库
import pandas as pd

# 定义了日志的输出格式，包括时间戳、日志级别、记录器名称和消息
_FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'


""" 前面加_,私有的单例日志类，一个单例模式的类，用来持有日志记录器的实例。即数据处理专属的logger """
class _LoggerHolder(object):
    """
    Logger singleton instance holder.
    """
    INSTANCE = None


""" 返回数据处理专属的logger，单例日志类 """
def get_logger():
    """
    Returns library scoped logger.
    :returns: Library logger.
    """
    if _LoggerHolder.INSTANCE is None:
        # 日志器格式
        formatter = logging.Formatter(_FORMAT)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger('repeatflow')  # 返回具有指定名称的Logger，并在必要时创建它
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _LoggerHolder.INSTANCE = logger
    return _LoggerHolder.INSTANCE


""" 加载给出路径的配置文件，默认是configs/ml1m.json， 包含对数据集，训练超参，测试超参，cache还不清楚作用，日志配置等"""
def load_configuration(descriptor):
    """
    Load configuration from the given descriptor.
    Args:
        descriptor:
    Returns:
    """
    if not os.path.exists(descriptor):
        raise IOError(f'Configuration file {descriptor} '
                      f'not found')
    with open(descriptor, 'r') as stream:
        return json.load(stream)


""" 对数据进行kcore处理，确保最后处理的记录中每个user的交互次数不小于u_ncore，每个item至少与i_ncore个user交互过 """
def kcore(data, u_ncore, i_ncore):
    """
    Preprocessing data to get k-core dataset.
    Each user has at least u_ncore items in his preference
    Each item is interacted by at least i_ncore users
    :param data:
    :param u_ncore: min number of interactions for each user
    :param i_ncore: min number of interactions for each item

    :return:
    """
    if u_ncore <= 1 and i_ncore <= 1:
        return data

    def filter_user(df):
        """ Filter out users less than u_ncore interactions """
        tmp = df.groupby(['org_user'], as_index=False)['org_item'].count()
        tmp.rename(columns={'org_item': 'cnt_item'},
                   inplace=True)
        df = df.merge(tmp, on=['org_user'])
        df = df[df['cnt_item'] >= u_ncore].reset_index(drop=True).copy()
        df.drop(['cnt_item'], axis=1, inplace=True)
        return df

    def filter_item(df):
        """ Filter out items less than u_ncore interactions """
        tmp = df.groupby(['org_item'], as_index=False)['org_user'].count()
        tmp.rename(columns={'org_user': 'cnt_user'},
                   inplace=True)
        df = df.merge(tmp, on=['org_item'])
        df = df[df['cnt_user'] >= i_ncore].reset_index(drop=True).copy()
        df.drop(['cnt_user'], axis=1, inplace=True)
        return df

    # because of repeat consumption, just count 1 for each user-item pair
    unique_data = data[['org_user', 'org_item']].drop_duplicates()
    while 1:
        unique_data = filter_user(unique_data)
        unique_data = filter_item(unique_data)
        chk_u = unique_data.groupby('org_user')['org_item'].count()
        chk_i = unique_data.groupby('org_item')['org_user'].count()
        if len(chk_i[chk_i < i_ncore]) <= 0 and len(chk_u[chk_u < u_ncore]) <= 0:
            break

    unique_data = unique_data.dropna()
    data = pd.merge(data, unique_data, on=['org_user', 'org_item'])
    return data


""" 这个数据处理程序的主入口 """
# change another configuration file to preprocess the corresponding dataset
# for example amazon book: configs/amzb.json   configs/ml1m.json   configs/ml10m.json   configs/amz_beauty.json
# configs/amz_movies_and_tv.json   configs/lastfm1k.json
""" 加载json配置文件，超参，数据集配置等，默认处理ml1m数据集 """
params = load_configuration(f'configs/lastfm1k.json')
dataset_params = params['dataset']
u_ncore = dataset_params.get('u_ncore', 1)  # 后面的1应该是默认值
i_ncore = dataset_params.get('i_ncore', 1)

logger = get_logger()  # 获取数据处理的日志记录器

# 拼接原始数据集路径，从项目文件夹开始
data_path = os.path.join(dataset_params['path'],
                         f'{dataset_params["interactions"]}.{dataset_params["file_format"]}')
# 拼接kcore处理后数据的保存路径，也是在exp/data/ml1m下
output_path = os.path.join(
            dataset_params['path'],
            f'{dataset_params["name"]}_interactions_'
            f'{u_ncore}core-users_{i_ncore}core-items.csv')
# 如果kcore处理后数据不存在，则读取原始数据，进行kcore处理，保存到kcore处理后的数据路径下
if not os.path.exists(output_path):
    logger.info(f'Read data from csv {data_path}')
    data = pd.read_csv(data_path, sep=dataset_params['sep'],
                       names=dataset_params['col_names'])
    # 进行kcore处理，并写到日志
    logger.info(f'k-core extraction with u_ncore={u_ncore} & i_ncore={i_ncore}')
    data = kcore(data, u_ncore=u_ncore, i_ncore=i_ncore)
    logger.info(f'Write to {output_path}')
    data.to_csv(output_path, sep=',', header=True, index=False)
    logger.info('Finish')
else:  # 对应的数据集已经处理过，直接读取
    logger.info(f'Read data from csv {output_path}')
    data = pd.read_csv(output_path)

# 打印kcore处理后的数据集信息，user，item，交互次数
logger.info(f'Number of users: {len(data["org_user"].unique())}')
logger.info(f'Number of items: {len(data["org_item"].unique())}')
logger.info(f'Number of interactions: {len(data)}')

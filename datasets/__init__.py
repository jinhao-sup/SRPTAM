# from datasets.dataset import Dataset
# from datasets.tempo_dataset import TempoDataset
#
#
# # 仅支持pos数据集和tempo_mojito数据集
# _SUPPORTED_DATASETS = {
#     'pos': Dataset,
#     'tempo_mojito': TempoDataset  # TODO：注意修改这里的时候需要同步修改json中的tempo_mojito
# }
#
#
# # 生成数据集的工厂  Factory that generate dataset
# def dataset_factory(params):
#     """
#     :param params: json配置文件对象
#     """
#     # 根据json配置，返回的是MojitoDataset的fism-item数据以及Dataset的data属性
#     dataloader_type = params['dataset'].get('dataloader', 'pos')
#     try:  # 返回数据集的data属性，即训练，验证，测试集等信息
#         return _SUPPORTED_DATASETS[dataloader_type](params).data
#     except KeyError:
#         raise KeyError(f'Not support {dataloader_type} dataset')

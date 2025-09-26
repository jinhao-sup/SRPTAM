import os
from utils.error_msg import Error


""" 在参数model_dir": "exp/model后拼接数据集，训练，模型参数，作为保存模型的文件路径，并返回json文件中的training训练参数和model模型参数 """
def process_params(params):
    # 读取json文件中的，数据集，训练，模型参数
    dataset_params = params['dataset']
    training_params = params['training']
    model_params = training_params['model']

    # 拼接数据集参数字符串，用作文件路径
    dataset_spec = f'{dataset_params["name"]}_' \
                   f'{dataset_params["u_ncore"]}ucore_' \
                   f'{dataset_params["i_ncore"]}icore'
    # 拼接训练参数字符串
    training_spec = f'lr{training_params["learning_rate"]}_' \
                    f'batch{training_params["batch_size"]}_' \
                    f'dim{training_params["embedding_dim"]}'
    model_type = model_params['type']
    model_name = model_params['name']

    """ 拼接保存模型权重路径/exp/model/...下的最后一层文件夹的参数字符串，即模型的各种超参 """
    # /exp/model/book_30ucore_20icore/samples_step5/nepoch100/tempo_mojito_lr0.001_batch512_dim64_seqlen50_l2emb0.0
    # _nblocks2_nheads2_dropout0.5_tempo-dim16-linspace8_lbdatrans0.5_mercer2_linspace8_residual-add_glob0.3_l2u0.0_l2i0.（最后少个0是否无法再长？）
    if model_name == 'srtam':
        # 先将model_name前的{model_type}_注释，即tempo_mojito的tempo_不要（防止文件名过长被linux截断）
        model_spec = f'{model_name}_' \
                     f'{training_spec}_' \
                     f'seqlen{model_params["params"]["seqlen"]}_' \
                     f'l2emb{model_params["params"]["l2_emb"]}_' \
                     f'nblocks{model_params["params"]["num_blocks"]}_' \
                     f'nheads{model_params["params"]["num_heads"]}_' \
                     f'dropout{model_params["params"]["dropout_rate"]}_' \
                     f'tempo-dim{training_params["tempo_embedding_dim"]}-' \
                     f'linspace{training_params["tempo_linspace"]}'

        if 'causality' in model_params['params'] and \
                model_params['params']['causality'] is False:
            model_spec = f'{model_spec}_noncausal'

        if 'residual' in model_params['params']:
            model_spec = f'{model_spec}_residual-{model_params["params"]["residual"]}'
        if 'use_year' in model_params['params'] and \
                model_params['params']['use_year'] is False:
            model_spec = f'{model_spec}_noyear'
        if 'lambda_glob' in model_params['params']:
            model_spec = f'{model_spec}_' \
                         f'glob{model_params["params"]["lambda_glob"]}'
        if 'lambda_user' in model_params['params']:
            model_spec = f'{model_spec}_' \
                         f'l2u{model_params["params"]["lambda_user"]}'
        if 'lambda_item' in model_params['params']:
            model_spec = f'{model_spec}_' \
                         f'l2i{model_params["params"]["lambda_item"]}'
        # TODO：用于调参
        if 'test_version' in model_params['params']:
            model_spec = f'{model_spec}_' \
                         f'test_version{model_params["params"]["test_version"]}'

    else:
        raise Error(f'Unknown model name {model_name}')

    # TODO：需要添加gnn-step
    # model_dir即exp/model，作用是根据参数创建模型文件路径，作为新的model_dir参数值，用于保存模型
    # 拼接更详细的模型保存路径，也可以理解为重置了json文件中'model_dir'的值，因为training_params是指向params['training']（浅拷贝）
    # training_params 成为 params['training'] 的一个引用，不是副本。因此，对 training_params 的任何修改都会反映到 params['training'] 上
    """ 实际上这里拼接的是保存模型的路径，即exp/model下很长的文件夹，以及保存metrics.csv（每个epoch训练后在验证集上的评估结果） """
    training_params['model_dir'] = os.path.join(
        training_params['model_dir'],
        dataset_spec,
        f'samples_step{dataset_params["samples_step"]}',
        f'nepoch{training_params["num_epochs"]}',
        model_spec)
    # 并且返回json文件中的训练参数和模型参数
    return training_params, model_params

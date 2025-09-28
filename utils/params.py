import os
from utils.error_msg import Error


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

    if model_name == 'srtam':
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

    training_params['model_dir'] = os.path.join(
        training_params['model_dir'],
        dataset_spec,
        f'samples_step{dataset_params["samples_step"]}',
        f'nepoch{training_params["num_epochs"]}',
        model_spec)
    # 返回json文件中的训练参数和模型参数
    return training_params, model_params

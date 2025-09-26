import os
import time
import numpy as np
from tqdm import tqdm
from utils.logging import get_logger
from loaders import dataloader_factory
from commands.evaluator import Evaluator


class Trainer:
    """ Trainer is responsible to estimate parameters for a given model """
    def __init__(self, sess, model, params):
        """
        Initialization a trainer. The trainer will be responsible to train the model
        :param sess: global session
        :param model: model to be trained
        :param params: hyperparameters for training
        """
        self.sess = sess
        self.model = model
        self.params = params
        # 模型保存路径，同时在下面保存每次训练后的评估指标。
        # 为什么在train.py中传入的params是对应json的配置，但是在下面的代码中却可以推出self.model_dir是拼接超参后的详细路径，而不是简单的"exp/model"
        # 因为process_params函数中training_params是params['training'] 的一个引用，不是副本。因此，对training_params的任何修改都会反映到 params['training'] 上
        self.model_dir = params['training']['model_dir']
        self.n_epochs = self.params['training'].get('num_epochs', 20)
        self.n_valid_users = self.params['training'].get('n_valid_users', 10000)  # 在每个epoch训练后/测试时，对验证/测试集进行缩小（取若干user）
        self.num_negtives = self.params['training'].get('num_negatives', 100)  # 配置文件没有这个参数
        self.logger = get_logger()

    def fit(self, data):
        """
        Training model
        :param data:训练，验证，测试集，聚合信息，训练集items流行度，原始uid和iid列表，训练实例索引，user和item总数，最先时间戳等保存到dataset_factory的self.data字典中
        """
        # create data loaders for train & validation
        cache_path = self.params['cache']['path']  # 缓存划分后数据，item流行度的文件路径
        dataset_params = self.params['dataset']
        u_ncore = dataset_params.get('u_ncore', 1)
        i_ncore = dataset_params.get('i_ncore', 1)
        item_type = dataset_params.get('item_type', 'item')
        # 公用路径字符串
        spec = f'{item_type}_{u_ncore}core-users_' \
               f'{i_ncore}core-items'
        training_params = self.params['training']
        model_params = training_params['model']
        """ 训练时，所有epoch的train_dataloader，都使用相同的随机种子。测试评估5次则是使用不同的随机种子 """
        random_seed = self.params['dataset'].get('random_state', 2022)
        model_name = model_params['name']
        timespan = model_params['params'].get('timespan', 256)  # todo：这个参数可以优化，但是其作为dataloader_factory的一个参数，可能需要一起改动

        # 保存训练每个epoch后在验证集的评估指标，如训练损失，验证损失，hdcg，保存到exp下metrics.csv
        metrics_path = '{}/metrics.csv'.format(self.model_dir)
        if os.path.isfile(metrics_path):
            os.remove(metrics_path)

        # 评估时，用于负样本的负采样方式。因为在训练过程中，一个epoch训完，就会在验证集上评估一次
        neg_sampling = 'uniform'
        if 'negative_sampling' in self.params['eval']:
            neg_sampling = self.params['eval']['negative_sampling']['type']
        best_valid_score = -1.0  # 用于保存在训练过程所有epoch中，验证集上得到的最好ndcg
        best_ep = -1  # 保存训练过程中ndcg最好的epoch
        seqlen = model_params['params'].get('seqlen', 50)  # 训练实例中各种序列长度，seq，pos，neg

        with open(metrics_path, 'w') as f:  # 每个epoch训练完成后，在验证集进行一次评估，评估指标都保存在metrics.csv中
            header = 'epoch,lr,train_loss,ndcg,hr,' \
                     'rep_ndcg,rep_hr,exp_ndcg,exp_hr'
            f.write(f'{header}\n')
            """ 开始迭代训练100个epoch，每个epoch都需要创建对应的train_dataloader，因为loader会根据不同的epoch加载对应的fism-item。但是随机种子相同，数据打乱一致 """
            for ep in range(self.n_epochs):
                start_time = time.time()
                # TODO：需要优化sse参数
                # 在每个epoch，传入训练集数据以及配置参数，通过工厂类返回训练数据加载器，用于训练
                # train_interaction_indexes是一个列表，存的是（uid，交互在该user的interactions中的索引），以idx作为最后一个pos，取前L次交互作为一个训练实例
                # 即一个user的整个交互序列，会产生多个训练实例，而不是像sasrec中只使用每个user训练集中的最后L次交互作为训练实例
                train_dataloader = dataloader_factory(
                    data=data,
                    batch_size=training_params['batch_size'],
                    seqlen=seqlen,
                    mode='train',
                    random_seed=random_seed,
                    cache_path=cache_path,
                    spec=spec,
                    model_name=model_name,
                    timespan=timespan,
                    train_interaction_indexes=data['train_interaction_indexes'])

                # calculate train loss
                """ 计算本次epoch的损失，并且在epoch里面的每个batch计算损失后都是进行模型权重的更新，达到训练模型的目的。
                实际上batch loss才是用于训练（标量），而epoch loss一般都是用于打印输出 """
                train_loss, train_reg_loss = self._get_epoch_loss(train_dataloader, ep)
                self.logger.info(f'Train loss: {train_loss}, reg_loss: {train_reg_loss}')  # 每个epoch训练后打印的训练损失，正则化损失都是0
                valid_batchsize = training_params['batch_size']

                # 在每个epoch训练后，传入验证集数据，通过工厂类返回验证数据加载器，用于在验证集上进行一次评估，并记录到metrics.csv中
                valid_dataloader = dataloader_factory(
                    data=data,
                    batch_size=valid_batchsize,
                    seqlen=seqlen,
                    mode='valid',
                    random_seed=random_seed,
                    cache_path=cache_path,
                    model_name=model_name,
                    num_scored_users=data['num_valid_users'],
                    timespan=timespan,
                    neg_sampling=neg_sampling)

                # get predictions on valid_set
                """ 每个epoch训练完后，在验证集进行一次评估。调用evaluator.py中的eval方法
                    因为返修需要添加评估指标，但是训练验证时仍然只在日志记录NDCG@10和HR@10，即保留原来的日志输出
                    虽然返回结果会在多出NDCG@5，HR@5，MAP@5@10，但是下面的代码没有使用，不影响"""
                score = Evaluator.eval(valid_dataloader, self.model)  # 返回在验证集的平均ndcg，hr，以列表的形式返回
                # 如果本次训练后，验证集上的ndcg比之前最好的ndcg要高，则保存模型，否则不保存
                if best_valid_score < score[0] or ep == 1:
                    save_model = True
                    best_valid_score = score[0]
                    best_ep = ep
                else:
                    save_model = False
                # 在日志文件，记录该epoch训练后的评估结果。即日志中Evaluating...后面的一行
                # 因为训练时指令将消息输出重定向到日志文件，所以验证结果在日志文件显示。而测试时则直接打印到控制台
                logged_message = self._get_message(
                    ep, self.model.learning_rate,
                    train_loss, score, start_time)
                self.logger.info(', '.join(logged_message))
                """ 同时将本次epoch训练后的模型参数，验证结果记录到metrics.csv中，注意metrics.csv中的val_loss应该是ndcg，只是上面的文件头名称写错。已修改 """
                metric_message = self._get_message(
                    ep, self.model.learning_rate,
                    train_loss, score, start_time, logged=False)
                f.write(','.join(metric_message) + '\n')
                f.flush()
                # 如果需要保存模型，则保存到exp/model下。
                # 但是在该文件夹下只会保存最佳模型，用于后续测试使用。按理来说，每次epoch更好都会保存一个模型，但是实际只保存最佳模型，因为在model的saver中设置了keep_checkpoint_max=1，只保存一个模型
                # 参考https://blog.csdn.net/qq_27825451/article/details/105819752
                if save_model:
                    save_path = f'{self.model_dir}/' \
                                f'{self.model.__class__.__name__.lower()}' \
                                f'-epoch_{ep}'
                    self.model.save(save_path=save_path, global_step=ep)
            # 所有epoch结束后，在日志文件的最后打印验证集上最好的ndcg，以及对应的epoch
            self.logger.info(f'Best validation : {best_valid_score}, '
                             f'on epoch {best_ep}')

    # 计算当前epoch的所有批次数据的平均损失，平均正则化损失（标量）
    def _get_epoch_loss(self, dataloader, epoch_id):
        """
        Forward pass for an epoch
        :param dataloader:
        :param epoch_id:
        """
        n_batches = dataloader.get_num_batches()  # 总批次数
        # 存储每个批次的损失和正则化损失
        losses = []
        reg_losses = []
        desc = f'Optimizing epoch #{epoch_id}'
        # for each batch  迭代各个批次
        for _ in tqdm(range(1, n_batches), desc=f'{desc}...'):
            # get batch data
            batch_data = dataloader.next_batch()
            # 计算当前批次的损失和正则化损失（列表），其中每个训练实例返回一个标量损失（所有时间步损失的平均值）
            batch_loss, batch_reg_loss = self._get_batch_loss(batch=batch_data)
            # 检查batch_loss是否为numpy.ndarray，如果是，则计算当前批次数据损失的平均值
            if type(batch_loss) == np.ndarray:
                batch_loss = np.mean(batch_loss)
            # 检查batch_loss和batch_reg_loss是否为无穷大或NaN，如果不是，则添加到对应的列表中
            # 仅仅是为了在日志打印，并不会对nan或inf的loss进行操作，因为在_get_batch_loss函数中已经通过损失进行模型权重的更新
            if not np.isinf(batch_loss) and not np.isnan(batch_loss):
                losses.append(batch_loss)
            if not np.isinf(batch_reg_loss) and not np.isnan(batch_reg_loss):
                reg_losses.append(batch_reg_loss)
        # 计算当前epoch所有批次数据的平均损失，平均正则化损失
        loss = np.mean(losses, axis=0)
        reg_loss = np.mean(reg_losses, axis=0)
        return loss, reg_loss

    """
    计算单个批次的损失（返回的应该是两个列表，代表各个训练实例的损失和正则化损失）
    并且在计算每个batch的损失后，都会进行模型权重的更新
    """
    def _get_batch_loss(self, batch):
        """
        Forward pass for a batch
        :param batch:
        """
        # 根据批次数据构建模型的输入字典
        feed_dict = self.model.build_feedict(batch, is_training=True)
        reg_loss = 0.0  # 初始化正则化损失为0（这个示例中似乎没有计算实际的正则化损失，因为在计算长期偏好预测损失时直接加到了里面）
        """
        使用sess.run执行损失计算和模型的训练操作。其中self.model.loss在_create_loss中计算（标量，会调用这个函数）
        因为在计算损失时已经是通过预测得分与标签比较得到损失值，所以在模型的优化过程只需要利用损失进行模型权重优化
        实际上指定返回self.model.train_ops，就是在通过优化器执行最小化损失优化模型的过程
        self.model.train_ops在model的_create_train_ops函数中预定义了一个最小化损失的操作
        """
        # 要注意这里算出的loss为inf时，就会传入到优化器进行模型权重更新，上面的检查loss是否为inf或nan只是为了计算一个epoch的loss来打印，
        # 并不是为了防止为inf或nan的loss影响模型的优化
        _, loss = self.sess.run(
            [self.model.train_ops, self.model.loss], feed_dict=feed_dict)

        return loss, reg_loss

    # 获取模型当前的信息，包含epoch，学习率，训练损失，NDCG，HR，时间，封装为字符串列表。应该是用于每次训练后接的一次评估
    @classmethod
    def _get_message(cls, ep, learning_rate,
                     train_loss, score, start_time, logged=True):
        duration = int(time.time() - start_time)
        ss, duration = duration % 60, duration // 60
        mm, hh = duration % 60, duration // 60
        if logged is True:
            message = [f'Epoch #{ep}',
                       f'LR {learning_rate:6.5f}',
                       f'Tr-Loss {train_loss:7.5f}',
                       f'Val NDCG {score[0]:7.5f}',
                       f'Val HR {score[1]:7.5f}',
                       f'Dur:{hh:0>2d}h{mm:0>2d}m{ss:0>2d}s']
        else:
            message = [f'{ep}:',
                       f'{learning_rate:6.7f}',
                       f'{train_loss:7.5f}',
                       f'{score[0]:7.5f}',
                       f'{score[1]:7.5f}']
        return message

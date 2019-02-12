import numpy as np
import tensorflow as tf
from new_lstm_keras_model import create_new_model
from lstm_keras_model import create_model
from data import DataSet, VideoReader
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model


def lr_schedule(epoch):
    """
    Learning Rate Schedule

    Learning rate is scheduled to be reduced after 10, 20
    :param epoch: The number of epochs
    :return: lr (float32) learning rate
    """
    lr = 0.001

    if epoch < 2:
        lr = 0.00001
    elif epoch < 10:
        lr = 0.001
    elif epoch < 20:
        lr = 0.0001
    elif epoch < 40:
        lr = 0.00001
    else:
        lr = 0.000001

    print('Learning rate: ', lr)
    return lr


# 1. read all data
database_name = 'THETIS'
seq_length = 50
feature_dims = 2048
data_set = DataSet(database_name, seq_length)
class_nums = 12
alpha=0.5
skf = StratifiedKFold(n_splits=2, random_state=1, shuffle=False)

epoch_nums = 1
batch_size = 32


# 2. define model
hidden_dimensions = [10, 10, 10]
# 多GPU使用
#parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=1)

for train_index, test_index in skf.split(data_set.data, data_set.label):
    train_reader = VideoReader(data_set,
                                train_index,
                                seq_length,
                                feature_dims,
                                class_nums,
                                True)
    test_reader = VideoReader(data_set,
                                test_index,
                                seq_length,
                                feature_dims,
                                class_nums,
                                False)
    model = create_model(seq_length, feature_dims, hidden_dimensions, class_nums)
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=lr_schedule(0)),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[tf.keras.metrics.categorical_accuracy])
    for epoch in range(epoch_nums):
        train_reader.reset()

        tr_loss_totally = 0.
        tr_acc_totally = 0.
        while train_reader.has_more():
            features, labels, current_minibatch = train_reader.next_minibatch(batch_size)
            tr_loss, tr_acc = model.train_on_batch(features, labels)
            tr_loss_totally += tr_loss * current_minibatch
            tr_acc_totally += tr_acc * current_minibatch
        print('Epoch {}: Loss {}, Acc {}'.format(epoch, tr_loss_totally/train_reader.size(), 
                                                tr_acc_totally/train_reader.size()))
        

    model.save("multi_lstm_loss.h5".format())
    model.save_weights("multi_lstm_weight.h5")
    
    newmodel = create_new_model(seq_length, feature_dims, hidden_dimensions, class_nums, alpha)
    newmodel.load_weights("multi_lstm_weight.h5", by_name=False)
   
    newmodel.compile(optimizer=tf.train.AdamOptimizer(learning_rate=lr_schedule(0)),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[tf.keras.metrics.categorical_accuracy])

    for epoch in range(epoch_nums):
        train_reader.reset()

        tr_loss_totally = 0.
        tr_acc_totally = 0.
        while train_reader.has_more():
            features, labels, current_minibatch = train_reader.next_minibatch(batch_size)
            tr_loss, tr_acc = newmodel.train_on_batch(features, labels)
            tr_loss_totally += tr_loss * current_minibatch
            tr_acc_totally += tr_acc * current_minibatch
        print('Epoch {}: Loss {}, Acc {}'.format(epoch, tr_loss_totally/train_reader.size(), 
                                                tr_acc_totally/train_reader.size()))






    
        # Test data for trained model
        test_minibatch_size = batch_size

        # process minibatches and evaluate the model
        metric_numer = 0
        metric_denom = 0
        minibatch_index = 0

        test_reader.reset()
        while test_reader.has_more():
            features, labels, current_minibatch = test_reader.next_minibatch(test_minibatch_size)
            tt_loss, tt_acc = newmodel.test_on_batch(features, labels)
            metric_numer += tt_acc * current_minibatch
            metric_denom += current_minibatch
            minibatch_index += 1

        # 保存模型
#        model.save("multi_lstm_loss[{tr_loss_totally/train_reader.size()}].h5".format())
#        model.save("multi_lstm_loss.h5".format())
#    newmodel = load_model('multi_lstm_loss.h5')
#    loaded_model.load_weights("multi_lstm_loss.h5")
    
    
    






import keras
from keras.layers import Dense, MaxPooling1D, Input, Conv1D, GRU, GlobalAveragePooling1D, BatchNormalization, \
    Activation, Lambda, Dropout, Bidirectional,multiply
from keras.activations import relu
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from quaternion_layers.conv import *
from conf_matrix import *
from quaternion_layers.bn import QuaternionBN, QuaternionBatchNormalization
from nn_utils.grud_layers import GRUD
from rwkv import RWKV
import numpy as np
from future_fusion import CSAFF



def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph,
                                           options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
    flops=flops.total_float_ops
    print(f"FLOPS: {flops / 10 ** 6:.03} M")
def add(x):
    return x[0] + x[1]


def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1


def quaterion_layer(x, kenerls):
    nodes = 8
    x1 = QuaternionConv1D(nodes, kenerls, padding="same", strides=2)(x)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Activation(relu)(x1)
    x1 = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x1)

    x1 = QuaternionConv1D(nodes, kenerls, padding="same", strides=1)(x1)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Activation(relu)(x1)
    x1 = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x1)

    x1 = QuaternionConv1D(nodes, kenerls, padding="same", strides=1)(x1)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Activation(relu)(x1)
    x1 = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x1)  # local poooling->获取局部特征
    return x1


def MQCCAF(train_features, train_y, test_features, test_y, classes, save_path):
    orinal_input = Input(shape=(train_features.shape[1], 1))
    nodes = 60
    x = Conv1D(nodes, 64, padding="same", strides=16, activation="relu")(orinal_input)

    x1 = quaterion_layer(x, 2)
    x2 = quaterion_layer(x, 3)
    x3 = quaterion_layer(x, 4)

    att1 = CSAFF(8)([x1, x4, x3])
    features = keras.layers.concatenate([x1, x2, x3, x4,att1])

    features = Bidirectional(GRU(8, return_sequences=True))(features)
    features = GlobalAveragePooling1D()(features)
    features = Dropout(0.5)(features)
    out = Dense(classes, activation="softmax")(features)
    model = keras.models.Model(inputs=[orinal_input], outputs=out)

    path = save_path

    keras_callbacks = [EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=1e-08),
                       ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min',
                                       save_weights_only=True)]

    Adam = keras.optimizers.Adam(lr=0.001, amsgrad=True, epsilon=1e-08)

    model.compile(optimizer=Adam, loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    model.fit([train_features.reshape(len(train_features), train_features.shape[1], 1)],
              to_categorical(train_y),
              batch_size=32, epochs=100,
              verbose=1,
              validation_split=0.2,
              callbacks=keras_callbacks)

    # startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式
    scores = model.evaluate([test_features.reshape(len(test_features), test_features.shape[1], 1)],
                            to_categorical(test_y))
    return model, scores


Xe_train = np.load('./dealed_data/Ottawa/Xe_train.npy', allow_pickle=True)
# Xe_test
Xe_test = np.load('./dealed_data/Ottawa/Xe_test.npy', allow_pickle=True)
# Ye_train
Ye_train = np.load('./dealed_data/Ottawa/Ye_train.npy', allow_pickle=True)
# Ye_test
Ye_test = np.load('./dealed_data/Ottawa/Ye_test.npy', allow_pickle=True)

ACC = []
for i in range(10):
    cnn_gru_parallel, acc = mscnngru(Xe_train.astype(np.float32), Ye_train, Xe_test.astype(np.float32), Ye_test, 100,
                                     "./checkpoint/MQCCAF%d.h5" % (i))
    ACC.append(acc[1])  # acc[0]->loss acc[1]->accuracy
    print("acc: ", acc)
    single_scores = cnn_gru_parallel.predict([Xe_test.astype(np.float32).reshape(len(Xe_test.astype(np.float32)),
                                                                                 Xe_test.astype(np.float32).shape[
                                                                                     1], 1)])
    y_true = to_categorical(Ye_test).argmax(axis=1)
    y_pred = single_scores.argmax(axis=1)
    save_img_path = './multi_scale/MQCCAF' + str(i + 1) + '.pdf'
    acc_score = accuracy_score(y_true, y_pred)
    print("predict class  ACC：", acc_score)
    labels = ['He', 'Or', 'In', 'Ba', 'Co']
    plot_conf(y_pred, y_true, labels, save_img_path)


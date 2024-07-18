import numpy as np

import mne
from mne import io
from mne.datasets import sample

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

import pathlib

# EEGNet模型定义，适用于分类问题
def EEGNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    # 根据传入的dropoutType参数确定使用的Dropout类型
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType必须是SpatialDropout2D或Dropout，以字符串形式传入。')
    
    # 输入层，形状为(通道数, 样本数, 1)
    input1 = Input(shape=(Chans, Samples, 1))

    # 第一个卷积块
    block1 = Conv2D(F1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1), use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)
    
    # 第二个卷积块
    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)
    
    # 展平层
    flatten = Flatten(name='flatten')(block2)
    
    # 全连接层
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    # 创建模型
    return Model(inputs=input1, outputs=softmax)

# 获取数据函数，用于为EEGNet准备数据
def get_data4EEGNet(kernels, chans, samples):
    # 设置图像数据格式为'channels_last'
    K.set_image_data_format('channels_last')

    # 数据路径
    data_path = mne.datasets.sample.data_path()

    # 原始数据和事件文件路径
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
    
    # 定义时间窗口
    tmin, tmax = -0., 1
    # 定义事件ID
    event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

    # 读取原始数据
    raw = io.Raw(raw_fname, preload=True, verbose=False)
    # 应用滤波
    raw.filter(2, None, method='iir')  
    # 读取事件数据
    events = mne.read_events(event_fname)

    # 标记坏的通道
    raw.info['bads'] = ['MEG 2443']  
    # 选择EEG通道
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    # 创建Epochs对象
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks, baseline=None, preload=True, verbose=False)
    # 获取标签
    labels = epochs.events[:, -1]

    # 获取数据并放大
    X = epochs.get_data() * 1000  
    y = labels

    # 分割数据集
    X_train = X[0:144,]
    Y_train = y[0:144]
    X_validate = X[144:216,]
    Y_validate = y[144:216]
    X_test = X[216:,]
    Y_test = y[216:]

    # 将标签转换为one-hot编码
    Y_train = np_utils.to_categorical(Y_train - 1)
    Y_validate = np_utils.to_categorical(Y_validate - 1)
    Y_test = np_utils.to_categorical(Y_test - 1)

    # 调整数据形状以适应模型输入
    X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

    return X_train, X_validate, X_test, Y_train, Y_validate, Y_test

# 定义数据维度
kernels, chans, samples = 1, 60, 151

# 获取数据
X_train, X_validate, X_test, Y_train, Y_validate, Y_test = get_data4EEGNet(kernels, chans, samples)

# 创建EEGNet模型
model = EEGNet(nb_classes=4, Chans=chans, Samples=samples, dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16, dropoutType='Dropout')

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 定义回调函数，用于保存最佳模型
checkpointer = ModelCheckpoint(filepath='./best_model/baseline.h5', verbose=1, save_best_only=True)

# 定义类别权重
class_weights = {0: 1, 1: 1, 2: 1, 3: 1}

# 训练模型
fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=300, verbose=2, validation_data=(X_validate, Y_validate), callbacks=[checkpointer], class_weight=class_weights)

# 加载最佳模型权重
model.load_weights('./best_model/baseline.h5')

# 预测测试集
probs = model.predict(X_test)
preds = probs.argmax(axis=-1)  
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("分类准确率: %f " % (acc))
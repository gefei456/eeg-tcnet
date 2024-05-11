import keras
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv1D, Conv2D, AveragePooling2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Add, Lambda, DepthwiseConv2D, Input, Permute
from tensorflow.keras.constraints import max_norm
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.losses import categorical_crossentropy, binary_crossentropy

def EEGTCNet(nb_classes, Chans=22, Samples=1125, layers=3, kernel_s=10, filt=10, dropout=0, activation='relu', F1=4,
             D=2,
             kernLength=64, dropout_eeg=0.1):
    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = .25
    numFilters = F1
    F2 = numFilters * D

    EEGNet_sep = EEGNet(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    block2 = Lambda(lambda x: x[:, :, -1, :])(EEGNet_sep)
    outs = TCN_block(input_layer=block2, input_dimension=F2, depth=layers, kernel_size=kernel_s, filters=filt,
                     dropout=dropout, activation=activation)
    out = Lambda(lambda x: x[:, -1, :])(outs)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(regRate))(out)
    sigmoid = Activation('softmax', name='sigmoid')(dense)

    return Model(inputs=input1, outputs=sigmoid)


def EEGNet(input_layer, F1=4, kernLength=64, D=2, Chans=22, dropout=0.5):
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D((1, Chans), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = SeparableConv2D(F2, (16, 1),
                             data_format='channels_last',
                             use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3


def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, dropout, activation='relu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out


data = h5py.File('D:/BaiduNetdiskDownload/eeg-tcnet-master/My_data/EA1.mat', 'r')

X_train = np.array(data['X_train'])
X_test = np.array(data['X_test'])
print(X_train.shape)

Chans, eeg_cols, num_classes = 22, 750, 2
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[2], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[2], X_test.shape[1])

y_train = np.array(data['y_train']).reshape(-1, 1) - 1
y_test = np.array(data['y_test']).reshape(-1, 1) - 1
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

F1 = 8
KE = 32
KT = 4
L = 2
FT = 12
pe = 0.2
pt = 0.3
classes = 4
channels = 22
crossValidation = False
batch_size = 256
epochs = 500
lr = 0.01

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1,
                              min_lr=1e-5)  # monitor='val_loss', factor=0.3, patience=5, verbose=1, min_lr=5e-5
model = EEGTCNet(nb_classes=2, Chans=22, Samples=750, layers=L, kernel_s=KT, filt=FT, dropout=pt, activation='elu',
                 F1=F1, D=2, kernLength=KE, dropout_eeg=pe)
opt = Adam(lr=lr)
model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy'])
for j in range(22):
    scaler = StandardScaler()
    scaler.fit(X_train[:, 0, j, :])
    X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
    X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])
model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=[reduce_lr])
y_pred = model.predict(X_test).argmax(axis=-1)
labels = y_test.argmax(axis=-1)
accuracy_of_test = accuracy_score(labels, y_pred)
print('最终测试精度为')
print(accuracy_of_test)

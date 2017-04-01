# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import plot_model, to_categorical
from keras.datasets import mnist
from keras.utils import np_utils


def get_mnist_data():
    (_X_train, _y_train), (_X_test, _y_test) = mnist.load_data()

    _X_train = _X_train.reshape(60000, input_shape[0], input_shape[1], 1) # add channel
    _X_train = _X_train.astype('float32')
    _X_train /= 255

    _X_test = _X_test.reshape(10000, input_shape[0], input_shape[1], 1) # add channel
    _X_test = _X_test.astype('float32')
    _X_test /= 255

    # one-hot encoding
    _Y_train = to_categorical(_y_train, nb_classes)
    _Y_test = to_categorical(_y_test, nb_classes)

    return (_X_train, _Y_train), (_X_test, _Y_test)


def build_cnn(input_shape, nb_filters, filter_size, pool_size):
    model = Sequential()

    model.add(Conv2D(nb_filters[0], kernel_size=filter_size, activation='relu', border_mode='valid', input_shape=input_shape))
    # border_mode が valid の場合出力画像は入力画像より小さくなり，same の場合は同じ大きさになるように padding してくれる．デフォルトは valid

    model.add(Conv2D(nb_filters[1], filter_size, activation='relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax'))

    return model


if __name__ == '__main__':
    nb_classes = 10
    batch_size = 128
    epochs = 12
    input_shape = (28, 28, 1)
    filter_size = (3, 3)
    pool_size = (2, 2)
    nb_filters = (32, 64)

    (X_train, Y_train), (X_test, Y_test) = get_mnist_data()

    model = build_cnn(input_shape, nb_filters, filter_size, pool_size)

    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

    # モデルをコンパイル
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Early-stopping
    # early_stopping = EarlyStopping(patience=0, verbose=1)

    # モデルの訓練
    history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1 # train_data に占める validation_data の比率
                    # callbacks=[early_stopping])
                    )

    score = model.evaluate(X_test, Y_test, verbose=1)

    print('\nTest loss: ', score[0])
    print('Test accuracy: ', score[1])
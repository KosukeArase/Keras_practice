# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.datasets import mnist
from keras.utils import np_utils


n_classes = 10
batch_size = 128
nb_epoch = 1


def build_multilayer_perception():
    model = Sequential()
    
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784).astype('float32') / 255 # reshape (60000, 28 ,28) -> (60000, 784)
X_test = X_test.reshape(10000, 784).astype('float32') / 255 # reshape (10000, 28 ,28) -> (10000, 784)

# one-hot encoding
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)


model = build_multilayer_perception()

# model.summary()
# plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

# モデルをコンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Early-stopping
# early_stopping = EarlyStopping(patience=0, verbose=1)

# モデルの訓練
for i in range(20):
    history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=1,
                    validation_split=0.1, # train_data に占める validation_data の比率
                    # callbacks=[early_stopping])
                    )

    score = model.evaluate(X_test, Y_test, verbose=1)

    print('\nTest loss: ', score[0])
    print('Test accuracy: ', score[1])
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Initializing the CNN
classifier = Sequential()

# Adding 1st convolution layer
classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))

# Max Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding 2nd convolution layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))

# Max Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# adding dropout
classifier.add(Dropout(0.25))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(activation="relu", units=128))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation='softmax', units=10))

# Compiling the CNN
classifier.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=['accuracy'])

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

classifier.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test))
score = classifier.evaluate(x_test, y_test, verbose=0)
print('Results for Convolutional Layer count=2, batch_size=64, epochs=10, optimizer=adadelta dropout=0.25,0.5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
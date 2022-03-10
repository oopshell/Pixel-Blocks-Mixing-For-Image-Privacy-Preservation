import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3  # default input size 299x299
from tensorflow.keras.applications.inception_v3 import preprocess_input
# from tensorflow.keras import backend
# backend.set_image_data_format('channels_first')


def DataSet():
    train_path_0 = 'class0_train/'
    train_path_1 = 'class1_train/'
    test_path_0 = 'class0_test/'
    test_path_1 = 'class1_test/'

    imglist_train_0 = os.listdir(train_path_0)
    imglist_train_1 = os.listdir(train_path_1)
    imglist_test_0 = os.listdir(test_path_0)
    imglist_test_1 = os.listdir(test_path_1)

    X_train = np.empty((len(imglist_train_0) + len(imglist_train_1), 192, 192, 3))
    Y_train = np.empty((len(imglist_train_0) + len(imglist_train_1), 2))
    count = 0
    for img_name in imglist_train_0:
        img_path = train_path_0 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((1, 0))
        count += 1
    for img_name in imglist_train_1:
        img_path = train_path_1 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 1))
        count += 1

    X_test = np.empty((len(imglist_test_0) + len(imglist_test_1), 192, 192, 3))
    Y_test = np.empty((len(imglist_test_0) + len(imglist_test_1), 2))
    count = 0
    for img_name in imglist_test_0:
        img_path = test_path_0 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((1, 0))
        count += 1
    for img_name in imglist_test_1:
        img_path = test_path_1 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 1))
        count += 1

    # shuffle the order of training pictures
    # # 1st time # #
    index = [i for i in range(len(X_train))]
    np.random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    # # 2nd time # #
    index = [i for i in range(len(X_train))]
    np.random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    # # 3rd time # #
    index = [i for i in range(len(X_train))]
    np.random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]

    # shuffle the order of testing pictures
    # # 1st time # #
    index = [i for i in range(len(X_test))]
    np.random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]
    # # 2nd time # #
    index = [i for i in range(len(X_test))]
    np.random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]
    # # 3rd # #
    index = [i for i in range(len(X_test))]
    np.random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = DataSet()

    # # model
    model = InceptionV3(weights=None, classes=2, input_shape=(192, 192, 3))

    # # optimizer
    # sgd = tf.optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    # sgd = tf.optimizers.SGD(lr=0.005, decay=1e-4)
    # sgd = tf.optimizers.SGD(lr=0.01, decay=1e-4)
    # adm = tf.optimizers.Adam(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    # tf.optimizers.Adam(0.001),
    # tf.optimizers.SGD(0.005),
    model.compile(
        optimizer='rmsprop',  # 'sgd'  # 'adam'
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # # train
    model.fit(X_train, y_train, batch_size=8, epochs=30)

    # # evaluate
    model.evaluate(X_test, y_test)

    # # save
    model.save('my_InceptionV3.h5')

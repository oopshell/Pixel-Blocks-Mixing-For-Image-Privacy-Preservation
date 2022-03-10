import os
import numpy as np
# import scipy
# from scipy import ndimage
# import matplotlib.pyplot as plt
# from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50  # default input size 224x224
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import random


def DataSet():
    train_path_0 = '96_48/Asian/'
    train_path_1 = '96_48/Black/'
    train_path_2 = '96_48/Caucasian/'
    train_path_3 = '96_48/Hispanic/'
    test_path_0 = 'CNBC_test/Asian_test/'
    test_path_1 = 'CNBC_test/Black_test/'
    test_path_2 = 'CNBC_test/Caucasian_test/'
    test_path_3 = 'CNBC_test/Hispanic_test/'

    imglist_train_0 = os.listdir(train_path_0)
    imglist_train_1 = os.listdir(train_path_1)
    imglist_train_2 = os.listdir(train_path_2)
    imglist_train_3 = os.listdir(train_path_3)
    imglist_test_0 = os.listdir(test_path_0)
    imglist_test_1 = os.listdir(test_path_1)
    imglist_test_2 = os.listdir(test_path_2)
    imglist_test_3 = os.listdir(test_path_3)

    X_train = np.empty((len(imglist_train_0) + len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3), 192, 192, 3))
    Y_train = np.empty((len(imglist_train_0) + len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3), 4))
    count = 0
    for img_name in imglist_train_0:
        img_path = train_path_0 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((1, 0, 0, 0))
        count += 1
    for img_name in imglist_train_1:
        img_path = train_path_1 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 1, 0, 0))
        count += 1
    for img_name in imglist_train_2:
        img_path = train_path_2 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 1, 0))
        count += 1
    for img_name in imglist_train_3:
        img_path = train_path_3 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 1))
        count += 1

    X_test = np.empty((len(imglist_test_0) + len(imglist_test_1) + len(imglist_train_2) + len(imglist_train_3), 192, 192, 3))
    Y_test = np.empty((len(imglist_test_0) + len(imglist_test_1) + len(imglist_train_2) + len(imglist_train_3), 4))
    count = 0
    for img_name in imglist_test_0:
        img_path = test_path_0 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((1, 0, 0, 0))
        count += 1
    for img_name in imglist_test_1:
        img_path = test_path_1 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 1, 0, 0))
        count += 1
    for img_name in imglist_test_2:
        img_path = test_path_2 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 1, 0))
        count += 1
    for img_name in imglist_test_3:
        img_path = test_path_3 + img_name
        img = image.load_img(img_path, target_size=(192, 192))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 1))
        count += 1
    
    # 打乱训练集中的数据
    # # 打乱索引 # # 1
    index = [i for i in range(len(X_train))]
    # random.shuffle(index)
    np.random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    # # 打乱索引 # # 2
    index = [i for i in range(len(X_train))]
    # random.shuffle(index)
    np.random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    # # 打乱索引 # # 3
    index = [i for i in range(len(X_train))]
    # random.shuffle(index)
    np.random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    # # 利用随机数种子 # #
    # np.random.seed(1367)
    # np.random.shuffle(X_train)
    # np.random.seed(1367)
    # np.random.shuffle(Y_train)

    # 打乱测试集中的数据
    # # 打乱索引 # # 1
    index = [i for i in range(len(X_test))]
    # random.shuffle(index)
    np.random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]
    # # 打乱索引 # # 2
    index = [i for i in range(len(X_test))]
    # random.shuffle(index)
    np.random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]
    # # 打乱索引 # # 3
    index = [i for i in range(len(X_test))]
    # random.shuffle(index)
    np.random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]
    # # 利用随机数种子 # #
    # np.random.seed(1367)
    # np.random.shuffle(X_test)
    # np.random.seed(1367)
    # np.random.shuffle(Y_test)

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = DataSet()

    # # model
    model = ResNet50(weights=None, classes=4, input_shape=(192, 192, 3))

    # # optimizer
    # sgd = tf.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # sgd = tf.optimizers.SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
    # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # sgd = optimizers.SGD(learning_rate=lr_schedule(0), decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer=sgd,
    # adm = tf.optimizers.Adam(lr=0.008, decay=1e-4)
    # optimizer=tf.optimizers.SGD(0.001),
    # optimizer=tf.optimizers.Adam(0.001),
    # optimizer=tf.optimizers.RMSprop(0.001),
    model.compile(optimizer=tf.optimizers.Adam(0.003),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # # train
    model.fit(X_train, Y_train, batch_size=32, epochs=25)

    # # evaluate
    model.evaluate(X_test, Y_test)

    # # save
    model.save('ResNet50_CNBC.h5')

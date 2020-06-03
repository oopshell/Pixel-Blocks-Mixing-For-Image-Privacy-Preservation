import os
import numpy as np
from PIL import Image
import copy
import random


if __name__ == "__main__":
    img_path = '.../'
    imglist = os.listdir(img_path)

    # # # encode # # #
    count = 0
    for img_name in imglist:
        # # load the target img # #
        path = img_path + img_name
        tar_img = Image.open(path)
        tar_img = tar_img.resize((192, 192), Image.ANTIALIAS)
        tar_img = np.array(tar_img)  # <class 'numpy.uint8'>
        if len(tar_img.shape) == 2:  # if the img only has one channel, convert it to 3-channel
            channel3 = np.zeros(shape=(192, 192, 3), dtype=np.uint8)
            channel3[:, :, 0] = tar_img
            channel3[:, :, 1] = tar_img
            channel3[:, :, 2] = tar_img
            tar_img = channel3

        # # select images for trans # #
        img_num = 10
        # random.seed(1011)
        imgs = random.sample(range(0, len(imglist)), img_num)

        # # select img pixel blocks for trans # #
        img_shape = 192
        block_len = 96
        block_wid = 24
        # block_num = int(img_shape / block_size)
        col = int(img_shape / block_len)
        row = int(img_shape / block_wid)

        for c in range(col):
            # np.random.seed(1011)
            tar_block = np.random.randint(low=0, high=2, size=(1, row))  # 0-no-change, 1-change

            # # encode the target img # #
            for r in range(row):
                if tar_block[0][r] == 1:  # 0-no-change, 1-change
                    # random.seed(1011)
                    t = random.randint(0, img_num-1)  # select one img randomly from imgs
                    img_n = imgs[t]
                    # load tar_img_t
                    tar_img_t = np.zeros(shape=tar_img.shape, dtype=np.uint8)
                    count_t = 0
                    for img_name_t in imglist:
                        if count_t == img_n:
                            path_t = img_path + img_name_t
                            img_t = Image.open(path_t)
                            img_t = img_t.resize((img_shape, img_shape), Image.ANTIALIAS)
                            tar_img_t = np.array(img_t)
                            if len(tar_img_t.shape) == 2:  # if the img only has one channel, convert it to 3-channel
                                channel3 = np.zeros(shape=(img_shape, img_shape, 3), dtype=np.uint8)
                                channel3[:, :, 0] = tar_img_t
                                channel3[:, :, 1] = tar_img_t
                                channel3[:, :, 2] = tar_img_t
                                tar_img_t = channel3
                        count_t += 1
                    # copy the (c,r) block of img_n to the block at the same position of tar_img 
                    for rgb in range(3):
                        ti1 = tar_img[:, :, rgb]
                        ti2 = tar_img_t[:, :, rgb]
                        row1 = r * block_wid
                        row2 = r * block_wid + block_wid
                        col1 = c * block_len
                        col2 = c * block_len + block_len
                        ti1[col1:col2, row1:row2] = copy.deepcopy(ti2[col1:col2, row1:row2])
                        tar_img[:, :, rgb] = copy.deepcopy(ti1)

            # # # save the encoded img # # #
            encoded_img = Image.fromarray(np.uint8(tar_img))
            # encoded_img.show()
            # encoded_img.save(img_path+img_name, quality=100)
            encoded_img.save(img_path+img_name)

            count += 1

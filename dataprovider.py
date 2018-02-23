import cv2
import numpy as np
import os
from tqdm import tqdm
class Dataprovider():
    def __init__(self, one_hot=True):
        train_path = 'dataset/train/images/'
        test_path  = 'dataset/test/images/'
        self.one_hot = one_hot
        self.train_list = list(map(lambda x: train_path + x, os.listdir(train_path)))
        self.test_list  = list(map(lambda x: test_path  + x, os.listdir(test_path)))
        self.x_tr,  self.y_tr   = self.read_data(self.train_list)
        self.x_test,self.y_test = self.read_data(self.test_list)

        self.batch_offset_tr = 0
        self.batch_offset_test = 0


    def read_data(self, dir_list):
        images = np.ones(shape=[len(dir_list), 65, 65, 3])
        labels = np.ones(shape=[len(dir_list)], dtype=np.uint8)
        index = 0
        for dir in tqdm(dir_list):
            img = cv2.imread(dir)
            label = int(dir[-5])
            images[index] = img
            labels[index] = label
            index += 1

        if self.one_hot:
            return images.astype(np.float32), np.eye(5)[labels].astype(np.uint8)
        return images.astype(np.float32), labels

    def get_train(self):
        return self.x_tr, self.y_tr

    def get_test(self):
        return self.x_test, self.y_test


    def next_batch_tr(self, batch_size):
        start = self.batch_offset_tr
        self.batch_offset_tr += batch_size
        if self.batch_offset_tr > self.x_tr.shape[0]:
            # Shuffle the data
            perm = np.arange(self.x_tr.shape[0])
            np.random.shuffle(perm)
            self.x_tr = self.x_tr[perm]
            self.y_tr = self.y_tr[perm]
            start = 0
            self.batch_offset_tr = batch_size
        end = self.batch_offset_tr
        x_tr = self.x_tr[start:end]
        y_tr = self.y_tr[start:end]
        return x_tr, y_tr

    def next_batch_test(self, batch_size):
        start = self.batch_offset_test
        self.batch_offset_test += batch_size
        if self.batch_offset_test > self.x_test.shape[0]:
            # Shuffle the data
            perm = np.arange(self.x_test.shape[0])
            np.random.shuffle(perm)
            self.x_test = self.x_test[perm]
            self.y_test = self.y_test[perm]
            start = 0
            self.batch_offset_test = batch_size
        end = self.batch_offset_test
        x_test = self.x_test[start:end]
        y_test = self.y_test[start:end]
        return x_test, y_test

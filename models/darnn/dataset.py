import numpy as np
import math

class Dataset:

    def __init__(self, X_train, y_train, T, split_ratio=0.7, normalized=False):
        self.train_size = int(split_ratio * (y_train.shape[0] - T - 1))
        self.test_size = y_train.shape[0] - T - 1 - self.train_size
        self.time_step = T
        if normalized:
            y_train = y_train - y_train.mean()
        self.X, self.y, self.y_seq = self.time_series_gen(X_train, y_train, T)

    def get_time_step(self):
        return self.time_step

    def get_size(self):
        return self.train_size, self.test_size

    def get_num_features(self):
        return self.X.shape[1]

    def get_train_set(self):
        return self.X[:self.train_size], self.y[:self.train_size], self.y_seq[:self.train_size]

    def get_test_set(self):
        return self.X[self.train_size:], self.y[self.train_size:], self.y_seq[self.train_size:]

    def time_series_gen(self, X, y, T):
        ts_x, ts_y, ts_y_seq = [], [], []
        for i in range(len(X) - T - 1):
            last = i + T
            ts_x.append(X[i: last])
            ts_y.append(y[last])
            ts_y_seq.append(y[i: last])
        return np.array(ts_x), np.array(ts_y), np.array(ts_y_seq)

import time

from models.darnn.trainer import Trainer
from models.darnn.dataset import Dataset
from data_loaders import m3comp, mg
import neurolab as nl
import numpy as np

from models.darnn.trainer import Trainer
from models.modeleval import ModelEval


class DA_RNN(ModelEval):
    def __init__(self, dataset, num_epochs = 100, batch_size=128):
        super(DA_RNN, self).__init__('DA-RNN')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.time_step = 1
        train_size, test_size = dataset.get_size()
        self.trainer = Trainer(num_epochs, batch_size, self.time_step, train_size)
        self.eta = 0.1
        self.dataset = dataset
        self._generator = True


    def train(self):
        X_train, y_train, y_seq_train = self.dataset.get_train_set();
        num_features = X_train.shape[1]
        self.trainer.init(num_features, 64, 64, self.eta)
        self.trainer.train_minibatch(X_train, y_train, y_seq_train)

    def predict(self):
        X_test, y_test, y_seq_test = self.dataset.get_test_set();
        return y_test, self.trainer.test(X_test, y_seq_test)

    def generate_report(self):
        start = time.clock()

        self.train()
        y_test, predictions = self.predict()

        time_taken = time.clock() - start
        return time_taken, y_test, predictions

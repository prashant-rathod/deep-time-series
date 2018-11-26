import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable
from .model import AttnEncoder, AttnDecoder
from .dataset import Dataset
from torch import optim
import os

class Trainer:

    def __init__(self, num_epochs, batch_size, time_step, train_size):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_func = nn.MSELoss()
        self.time_step = time_step
        self.train_size = train_size

    def init(self,input_size, enHidden, decHidden, lr):
        self.encoder = AttnEncoder(input_size=input_size, hidden_size=enHidden, time_step=self.time_step)
        self.decoder = AttnDecoder(code_hidden_size=enHidden, hidden_size=decHidden, time_step=self.time_step)
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr)

    def train_minibatch(self, x_train, y_train, y_seq_train):
        for epoch in range(self.num_epochs):
            i = 0
            loss_sum = 0
            while (i < self.train_size):
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                batch_end = i + self.batch_size
                if (batch_end >= self.train_size):
                    batch_end = self.train_size
                var_x = self.to_variable(x_train[i: batch_end])
                var_y = self.to_variable(y_train[i: batch_end])
                var_y_seq = self.to_variable(y_seq_train[i: batch_end])
                if var_x.dim() == 2:
                    var_x = var_x.unsqueeze(2)
                code = self.encoder(var_x)
                y_res = self.decoder(code, var_y_seq)
                loss = self.loss_func(y_res, var_y)
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                # print('[%d], loss is %f' % (epoch, 10000 * loss.data[0]))
                loss_sum += loss.data.item()
                i = batch_end
            #print('epoch [%d] finished, the average loss is %f' % (epoch, loss_sum))
            #if epoch + 1 == self.num_epochs:
            #    root_fold = os.path.join(os.getcwd(), 'data')
            #    enc_filename = os.path.join(root_fold, 'encoder' + str(self.num_epochs) + '-norm' + '.model')
            #    dec_filename = os.path.join(root_fold, 'decoder' + str(self.num_epochs) + '-norm' + '.model')
            #    print('save',enc_filename)
            #    torch.save(self.encoder.state_dict(), enc_filename)
            #    torch.save(self.decoder.state_dict(), dec_filename)

    def load_model(self):
        pass
        #root_fold = os.path.join(os.getcwd(), 'data')
        #enc_filename = os.path.join(root_fold, 'encoder' + str(self.num_epochs) + '-norm' + '.model')
        #dec_filename = os.path.join(root_fold, 'decoder' + str(self.num_epochs) + '-norm' + '.model')
        #print('load', enc_filename)
        #self.encoder.load_state_dict(torch.load(enc_filename, map_location=lambda storage, loc: storage))
        #self.decoder.load_state_dict(torch.load(dec_filename, map_location=lambda storage, loc: storage))

    def test(self, x_test, y_seq_test):
        y_pred_test = self.predict(x_test, y_seq_test, self.batch_size)
        return y_pred_test

    def predict(self, x, y_seq, batch_size):
        y_pred = np.zeros(x.shape[0])
        i = 0
        while (i < x.shape[0]):
            batch_end = i + batch_size
            if batch_end > x.shape[0]:
                batch_end = x.shape[0]
            var_x_input = self.to_variable(x[i: batch_end])
            var_y_input = self.to_variable(y_seq[i: batch_end])
            if var_x_input.dim() == 2:
                var_x_input = var_x_input.unsqueeze(2)
            code = self.encoder(var_x_input)
            y_res = self.decoder(code, var_y_input)
            for j in range(i, batch_end):
                y_pred[j] = y_res[j - i, -1]
            i = batch_end
        return y_pred

    def to_variable(self, x):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import signalz


def mackey_glass(length, add_noise=False, noise_range=(-0.01, 0.01)):
    initial = .25 + .5 * np.random.rand()
    signal = signalz.mackey_glass(length, a=0.2, b=0.8, c=0.9, d=23, e=10, initial=initial)
    if add_noise:
        signal += np.random.uniform(noise_range[0], noise_range[1], size=signal.shape)
    return signal - 1.

generators = {
    'mg': mackey_glass
}

def _generate_data(length=10000, batch_size=32, add_noise=False, generator='mg'):
    x_data = generators[generator](length + 1, add_noise=add_noise)

    X = x_data[:-1]
    y = x_data[1:]
    return X.reshape(-1,1), y.reshape(-1,1)

def _generate_data_old(length=10000, batch_size=32, add_noise=False, add_val=False, generator='mg', train_size=0.7):
    X = np.empty((length, batch_size, 1))
    y = np.empty((length, batch_size, 1))

    for b in range(batch_size):
        x_data = generators[generator](length + 1, add_noise=add_noise)

        if b == 0:
            plt.figure("Synthetic data", figsize=(15, 10))
            plt.title("Synthetic data")
            plt.plot(range(min(1000, length)), x_data[:min(1000, length)])

        X[:, b, 0] = x_data[:-1]
        y[:, b, 0] = x_data[1:]

    plt.savefig("synthetic_data.png")
    plt.close()

    # 70% training, 10% validation, 20% testing
    train_sep = test_sep = int(length * train_size)

    X_train = Variable(torch.from_numpy(X[:train_sep, :]).float(), requires_grad=False)
    y_train = Variable(torch.from_numpy(y[:train_sep, :]).float(), requires_grad=False)

    if add_val:
        test_sep = train_sep + int(length * 0.1)
        X_val = Variable(torch.from_numpy(X[train_sep:test_sep, :]).float(), requires_grad=False)
        y_val = Variable(torch.from_numpy(y[train_sep:test_sep, :]).float(), requires_grad=False)

    X_test = Variable(torch.from_numpy(X[test_sep:, :]).float(), requires_grad=False)
    y_test = Variable(torch.from_numpy(y[test_sep:, :]).float(), requires_grad=False)

    print(("X_train size = {}, X_val size = {}, X_test size = {}".format(X_train.size(), X_val.size(), X_test.size())))
    if add_val:
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test

def generate_data(**kwargs):
    length = kwargs.get('length', 10000)
    batch_size = kwargs.get('batch_size', 32)
    add_noise = kwargs.get('add_noise', False)
    type = kwargs.get('type', 'mg')

    return _generate_data(length, batch_size, add_noise, type)

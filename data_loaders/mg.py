import numpy as np
import signalz
from sklearn import preprocessing
import sklearn.model_selection as sk

def mackey_glass(length=1000, add_noise=False, noise_range=(-0.01, 0.01)):
    r = np.random.RandomState(42)
    initial = .25 + .5 * r.rand()
    signal = signalz.mackey_glass(length, a=0.2, b=0.8, c=0.9, d=23, e=10, initial=initial)
    if add_noise:
        signal += r.uniform(noise_range[0], noise_range[1], size=signal.shape)

    #signal = preprocessing.scale(signal)
    return signal
def create_test_train_split(signal):
    x = signal[:-1]
    y = signal[1:]
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return sk.train_test_split(x, y, test_size=0.30, random_state=42)

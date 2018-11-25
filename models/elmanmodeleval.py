import neurolab as nl
import numpy as np

from models.modeleval import ModelEval


class ElmanModelEval(ModelEval):
    def __init__(self, min=-1, max=1):
        super(ElmanModelEval, self).__init__('Elman')
        self.model = nl.net.newelm([[min, max]], [20, 10, 1],
                            [nl.trans.TanSig(), nl.trans.TanSig(), nl.trans.PureLin()])

        self.init()

    def init(self):
        self.model.layers[0].initf = nl.init.InitRand([-0.01, 0.01], 'wb')
        self.model.layers[1].initf = nl.init.InitRand([-0.01, 0.01], 'wb')
        self.model.layers[2].initf = nl.init.InitRand([-0.1, 0.1], 'wb')

        self.model.init()
        return self.model

    def train(self, X, y):
        error = self.model.train(X, y, epochs=1000, show=False, goal=0.01)
        return error

    def predict(self, X):
        out = self.model.sim(X)
        return out

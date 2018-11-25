class ModelEval(object):
    def __init__(self, name):
        self.model_name = name

    def get_name(self):
        return self.model_name

    def init(self):
        raise NotImplementedError("Should have implemented this")

    def train(self, x, y):
        raise NotImplementedError("Should have implemented this")

    def predict(self, x):
        raise NotImplementedError("Should have implemented this")

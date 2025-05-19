class BaseEvaluator:
    def __init__(self, model):
        self.model = model
        self.metrics = {}

    def evaluate(self, X_test, y_test):
        raise NotImplementedError

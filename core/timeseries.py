from sklearn.metrics import mean_absolute_error


class TimeSeriesEvaluator:
    def __init__(self, model):
        self.model = model
        self.metrics = {}

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.metrics['MAE'] = mean_absolute_error(y_test, y_pred)
        return self.metrics

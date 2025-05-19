from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegressionEvaluator:
    def __init__(self, model):
        self.model = model
        self.metrics = {}

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.metrics['MAE'] = mean_absolute_error(y_test, y_pred)
        self.metrics['RMSE'] = mean_squared_error(
            y_test, y_pred, squared=False)
        self.metrics['R2'] = r2_score(y_test, y_pred)
        return self.metrics

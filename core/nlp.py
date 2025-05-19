from sklearn.metrics import accuracy_score, classification_report


class NLPEvaluator:
    def __init__(self, model):
        self.model = model
        self.metrics = {}

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.metrics['report'] = classification_report(
            y_test, y_pred, output_dict=True)
        return self.metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


class ClassificationEvaluator:
    def __init__(self, model):
        self.model = model
        self.metrics = {}

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.metrics['report'] = classification_report(
            y_test, y_pred, output_dict=True)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        return self.metrics

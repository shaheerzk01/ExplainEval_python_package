from core.classification import ClassificationEvaluator
from core.regression import RegressionEvaluator
from core.nlp import NLPEvaluator
from core.timeseries import TimeSeriesEvaluator

from explainers.shap_explainer import ShapExplainer
from visualizer.dashboard import generate_html_report


class EvalX:
    def __init__(self, model, task, background_data=None):
        self.model = model
        self.task = task.lower()
        self.background_data = background_data
        self.evaluator = self._get_evaluator()
        self.explainer = ShapExplainer(model, background_data)

    def _get_evaluator(self):
        if self.task == 'classification':
            return ClassificationEvaluator(self.model)
        elif self.task == 'regression':
            return RegressionEvaluator(self.model)
        elif self.task == 'nlp':
            return NLPEvaluator(self.model)
        elif self.task == 'timeseries':
            return TimeSeriesEvaluator(self.model)
        else:
            raise ValueError("Invalid task type")

    def evaluate(self, X_test, y_test):
        return self.evaluator.evaluate(X_test, y_test)

    def explain(self, X_sample):
        return self.explainer.explain(X_sample)

    def generate_report(self, output_file="report.html"):
        context = {
            "task": self.task,
            "metrics": self.evaluator.metrics
        }
        generate_html_report(context, output_file)

    def compare(self, models, X_test, y_test):
        results = {}
        for i, m in enumerate(models):
            temp_eval = EvalX(m, self.task, self.background_data)
            results[f"Model_{i+1}"] = temp_eval.evaluate(X_test, y_test)
        return results

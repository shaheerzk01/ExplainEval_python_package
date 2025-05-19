import shap
import matplotlib.pyplot as plt


class ShapExplainer:
    def __init__(self, model, background_data):
        self.explainer = shap.Explainer(model, background_data)

    def explain(self, X_sample):
        shap_values = self.explainer(X_sample)
        shap.plots.beeswarm(shap_values)
        plt.show()

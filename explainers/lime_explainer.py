from lime.lime_tabular import LimeTabularExplainer


class LimeExplainer:
    def __init__(self, model, X_train):
        self.explainer = LimeTabularExplainer(X_train, mode='classification')

    def explain(self, instance):
        exp = self.explainer.explain_instance(
            instance, self.model.predict_proba)
        exp.show_in_notebook()

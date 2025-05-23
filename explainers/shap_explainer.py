import shap
import matplotlib.pyplot as plt

class ShapExplainer:
    def __init__(self, model, background_data):
        self.model = model
        self.is_pipeline = hasattr(model, "named_steps") and "vec" in model.named_steps and "clf" in model.named_steps

        if self.is_pipeline:
            # Handle NLP pipeline: extract vectorizer and classifier
            self.vectorizer = model.named_steps["vec"]
            self.classifier = model.named_steps["clf"]
            background_matrix = self.vectorizer.transform(background_data)

            # Only support classifiers for now
            self.explainer = shap.Explainer(self.classifier.predict_proba, background_matrix)
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.background_matrix = background_matrix
        else:
            self.explainer = shap.Explainer(model, background_data)
            self.feature_names = getattr(background_data, "columns", None)

    def explain(self, X_sample):
        if self.is_pipeline:
            X_transformed = self.vectorizer.transform(X_sample)
            shap_values = self.explainer(X_transformed)
        else:
            shap_values = self.explainer(X_sample, check_additivity=False)

        # Handle multi-class output
        if hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
            print("⚠️ Multi-class SHAP detected. Showing class 0.")
            shap.plots.beeswarm(shap.Explanation(
                values=shap_values.values[:, :, 0],
                base_values=shap_values.base_values[:, 0],
                data=shap_values.data,
                feature_names=self.feature_names
            ))
        else:
            shap.plots.beeswarm(shap.Explanation(
                values=shap_values.values,
                base_values=shap_values.base_values,
                data=shap_values.data,
                feature_names=self.feature_names
            ))

        plt.show()

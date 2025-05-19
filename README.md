# 📊 ExplainEval

**ExplainEval** is a unified evaluation and explainability framework for machine learning models. It supports classification, regression, NLP, and time series tasks, providing a consistent API to evaluate models, explain predictions, and generate interactive reports.

---

## 🚀 Features

- ✅ Easy evaluation for Classification, Regression, NLP, and Time Series models
- 🔍 Built-in explainability using SHAP and LIME
- 📊 Confusion matrix, ROC curves, MAE, RMSE, and more
- 📋 Auto-generated HTML reports for stakeholders
- 🔁 Model comparison interface
- 📦 Compatible with scikit-learn, XGBoost, LightGBM, Transformers

---

## 📦 Installation

Install via PyPI:

```bash
pip install explaineval
```

Or install from source:

```bash
git clone https://github.com/yourname/explaineval.git
cd explaineval
pip install -e .
```

---

## 🔧 Supported Tasks

| Task Type     | Models Supported                            | Explainability |
|--------------|----------------------------------------------|----------------|
| Classification | scikit-learn, XGBoost, LightGBM, etc.        | SHAP, LIME     |
| Regression     | Linear, Tree-based, Boosting, etc.           | SHAP           |
| NLP            | Sklearn Pipelines, Transformers              | SHAP, Attention|
| Time Series    | ARIMA, LSTM, XGBoost, etc.                   | SHAP           |

---

## 🧠 Usage Example

### 🔍 Classification Example

```python
from explaineval.main import EvalX
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier().fit(X_train, y_train)

evalx = EvalX(model, task="classification", background_data=X_train)
evalx.evaluate(X_test, y_test)
evalx.explain(X_test[:5])
evalx.generate_report("classification_report.html")
```

### 📈 Regression Example

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor

X, y = load_diabetes(return_X_y=True)
model = GradientBoostingRegressor().fit(X, y)
evalx = EvalX(model, task="regression", background_data=X)
evalx.evaluate(X, y)
evalx.explain(X[:5])
evalx.generate_report("regression_report.html")
```

---

## 📋 Auto Reports

Call `generate_report("filename.html")` to save a clean HTML report including:
- Task summary
- Evaluation metrics
- SHAP/LIME explanation visualizations

---

## 🧪 Model Comparison

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

model1 = RandomForestClassifier().fit(X_train, y_train)
model2 = GradientBoostingClassifier().fit(X_train, y_train)

results = evalx.compare([model1, model2], X_test, y_test)
print(results)
```

---

## 📄 License

MIT License © 2025 Shaheer Zaman Khan

---

## 🤝 Contributing

We welcome contributions! To contribute:
1. Fork the repo
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

---

## 📬 Contact

Have questions or feedback? Open an issue or contact us at [shaheerzk01@gmail.com](mailto:shaheerzk01@gmail.com)

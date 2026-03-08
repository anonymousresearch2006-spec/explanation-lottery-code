import numpy as np
from sklearn.svm import LinearSVC
import shap

X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
model = LinearSVC(random_state=42).fit(X, y)

explainer = shap.LinearExplainer(model, X[:10])
sv = explainer.shap_values(X[:5])
print("LinearExplainer works. Shape:", np.array(sv).shape)

explainer2 = shap.KernelExplainer(model.decision_function, X[:10])
sv2 = explainer2.shap_values(X[:5])
print("KernelExplainer works. Shape:", np.array(sv2).shape)

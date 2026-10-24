
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build and train model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)
model.fit(X_train, y_train)

# 4. Predict classes and probabilities
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# 5. Evaluate
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

# 6. Cross-validation on training data
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print("CV accuracy:", scores)
print("Mean CV accuracy:", scores.mean())

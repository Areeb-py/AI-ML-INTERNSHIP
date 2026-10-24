
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for depth in [1, 3, 5, 10, None]:
    model = DecisionTreeClassifier(
        max_depth=depth, random_state=42
    )
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(
        f"Depth={depth}, "
        f"Train={train_score:.3f}, "
        f"Test={test_score:.3f}"
    )

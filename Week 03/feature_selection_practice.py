from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X, y = data.data, data.target

print("Original number of features:", X.shape[1])

selector = SelectKBest(score_func=f_classif, k=5) #CHANGE K=5 TO SELECT DIFFERENT NUMBER OF FEATURES)

X_selected = selector.fit_transform(X, y)

print("Selected number of features:", X_selected.shape[1])

selected_features = data.feature_names[selector.get_support()]

print("Selected features:")
print(selected_features)
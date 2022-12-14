# Source 1: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
# Source 2: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X, y = make_regression(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
regr = MLPRegressor(random_state=1, max_iter=2000).fit(X_train, y_train)
#regr.predict(X_test[:2])
print("Regression Score", regr.score(X_test, y_test))
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
clf = MLPClassifier(random_state=1, max_iter=2000).fit(X_train, y_train)
#clf.predict_proba(X_test[:1])
#clf.predict(X_test[:5, :])
print("Classification Score", clf.score(X_test, y_test))

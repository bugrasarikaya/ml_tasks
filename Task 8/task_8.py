# Source 1: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# Source 2: https://scikit-learn.org/stable/datasets/toy_dataset.html
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
y_pred = SVC().fit(X_train, y_train)
print("Iris Dataset - SVC Score:", y_pred.score(X_test, y_test))
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
y_pred = SVC().fit(X_train, y_train)
print("Wine Dataset - SVC Score:", y_pred.score(X_test, y_test))
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
y_pred = SVC().fit(X_train, y_train)
print("Digits Dataset - SVC Score:", y_pred.score(X_test, y_test))

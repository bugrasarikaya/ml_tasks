# Source 1: https://scikit-learn.org/stable/modules/naive_bayes.html
# Source 2: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
y_pred = GaussianNB().fit(X_train, y_train)
print("Naive Bayes Score:", y_pred.score(X_test, y_test))
from sklearn.linear_model import LinearRegression
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
y_pred = LinearRegression().fit(X_train, y_train)
print("Linear Regression Score:", y_pred.score(X_test, y_test))

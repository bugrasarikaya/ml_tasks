#Source 1: https://stackoverflow.com/questions/32277562/how-to-set-up-id3-algorith-in-scikit-learn
#Source 2: https://pypi.org/project/decision-tree-id3/
#Source 3: https://githubhelp.com/svaante/decision-tree-id3/issues/10
#Source 4: https://scikit-learn.org/stable/datasets/toy_dataset.html
#Source 5: https://github.com/RaczeQ/scikit-learn-C4.5-tree-classifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import six
import sys
sys.modules['sklearn.externals.six'] = six
from id3 import Id3Estimator
from id3 import export_graphviz
iris  = load_iris()
estimator = Id3Estimator()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)
estimator.fit(X_train, y_train)
export_graphviz(estimator.tree_, 'id3_tree.dot', iris.feature_names)
res_pred = estimator.predict(X_test)
print("ID3 accuracy: ", accuracy_score(y_test, res_pred))
from c45 import C45
iris  = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)
clf = C45(attrNames=iris.feature_names)
clf.fit(X_train, y_train)
res_pred_2 = clf.predict(X_test)
print("C4.5 accuracy: ", accuracy_score(y_test, res_pred_2))

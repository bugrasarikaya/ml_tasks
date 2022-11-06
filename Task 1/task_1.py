# Source 1: https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
# Source 2: https://python-course.eu/machine-learning/k-nearest-neighbor-classifier-with-sklearn.php
# Source 3: https://stackoverflow.com/questions/40815008/how-to-do-n-cross-validation-in-knn-python-sklearn
# Source 4: https://stackoverflow.com/questions/49015452/low-k-fold-accuracy-for-first-fold
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
data, labels = iris.data, iris.target
plt.scatter(data[:, 0], data[:, 1], s = 50)
plt.suptitle("Iris Data")
plt.show()
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
    km.fit(data)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.suptitle("The Elbow Method")
plt.show()
km = KMeans(n_clusters = 3, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
labels_km = km.fit_predict(data)
plt.figure(1).suptitle("K-Means")
plt.scatter(
    data[labels_km == 0, 0], data[labels_km == 0, 1],
    s = 50, c = 'lightgreen',
    marker = 's', edgecolor = 'black',
    label = 'cluster 1'
)

plt.scatter(
    data[labels_km == 1, 0], data[labels_km == 1, 1],
    s = 50, c = 'orange',
    marker = 'o', edgecolor = 'black',
    label = 'cluster 2'
)

plt.scatter(
    data[labels_km == 2, 0], data[labels_km == 2, 1],
    s = 50, c = 'lightblue',
    marker = 'v', edgecolor = 'black',
    label = 'cluster 3'
)

plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s = 250, marker = '*',
    c = 'red', edgecolor = 'black',
    label = 'centroids'
)
plt.legend(scatterpoints = 1)
plt.grid()
from sklearn.metrics import accuracy_score
print("K-Means acccuracy: ", accuracy_score(labels_km, labels))
accuracy_array = []
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
kf = KFold(n_splits = 2, shuffle = True)
index = 2
for train_index, test_index in kf.split(data):
    data_train, data_test = data[train_index], data[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    knn = KNeighborsClassifier()
    knn.fit(data_train, labels_train)
    labels_km = knn.predict(data_test)
    plt.figure(index).suptitle("K-NN")
    plt.scatter(
        data_test[labels_km == 0, 0], data_test[labels_km == 0, 1],
        s = 50, c = 'lightgreen',
        marker = 's', edgecolor = 'black',
        label = 'cluster 1'
    )

    plt.scatter(
        data_test[labels_km == 1, 0], data_test[labels_km == 1, 1],
        s = 50, c = 'orange',
        marker = 'o', edgecolor = 'black',
        label = 'cluster 2'
    )

    plt.scatter(
        data_test[labels_km == 2, 0], data_test[labels_km == 2, 1],
        s = 50, c = 'lightblue',
        marker = 'v', edgecolor = 'black',
        label = 'cluster 3'
    )

    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s = 250, marker = '*',
        c = 'red', edgecolor = 'black',
        label = 'centroids'
    )
    plt.legend(scatterpoints = 1)
    plt.grid()
    accuracy_array.append(accuracy_score(labels_km, labels_test))
    index = index + 1
print("K-NN acccuracy: ", np.mean(accuracy_array))
plt.show()


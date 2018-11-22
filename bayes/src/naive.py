import numpy as np
from sklearn.naive_bayes import GaussianNB

features_train = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
labels_train = np.array([1, 1, 1, 2, 2, 2])

# clf - classifier
clf = GaussianNB()
clf.fit(features_train, labels_train)
p = clf.predict([[-0.8, -1]])

print(p)
# outputs the class it belongs to

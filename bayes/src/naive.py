import numpy as np
from sklearn.naive_bayes import GaussianNB

# training data - features
X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# training data - labels
Y = np.array([1, 1, 1, 2, 2, 2])

# clf - classifier

clf = GaussianNB()
clf.fit(X, Y)
p = clf.predict([[-0.8, -1]])

print(p)
# outputs the class it belongs to

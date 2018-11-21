import numpy as np
X = np.array([[-1,1], [-2,-1], [-3,-2], [1,1], [2,1], [3,2]])
Y = np.array([1, 1, 1, 2, 2, 2])

from sklearn.naive_bayes import GaussianNB

# clf - classifier

clf = GaussianNB()
clf.fit(X, Y)
p = clf.predict([[-0.8, -1]])

print(p)
# outputs the class it belongs to


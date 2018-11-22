import numpy as np

def NBAccuracy(features, labels, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features, labels)

    ### use the trained classifier to predict labels for the test features
    # pred =  # TODO

    i = 0
    correct = 0
    for feature in features_test:
        pred = clf.predict([feature])
        if pred == labels_test[i]:
            correct = correct + 1
        i = i + 1

    print ("Correct ", correct)

    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    # accuracy =  # TODO
    accuracy = (correct / labels_test.size) * 100

    return accuracy


def main():
    features = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    labels = np.array([1, 1, 1, 2, 2, 2])

    accuracy = NBAccuracy(features, labels, features, labels)
    print ("Accuracy: ", accuracy)


main()

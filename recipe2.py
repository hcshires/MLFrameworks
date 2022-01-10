# 1. import data set
from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
'''
# print info from set
print(iris.feature_names) # metadata
print(iris.target_names)
print(iris.data[0]) # features
print(iris.target[0]) # labels
'''

testIdx = [0, 50, 100]  # each index where a new set of flowers begin

# training data
trainTarget = np.delete(iris.target, testIdx)
trainData = np.delete(iris.data, testIdx, axis=0)

# testing data
testTarget = iris.target[testIdx]
testData = iris.data[testIdx]

# classifier
clf = tree.DecisionTreeClassifier()
clf.fit(trainData, trainTarget)

print(testTarget)  # what our label is expected to be
print(clf.predict(testData))

print(testData[0], testTarget[0])

from sklearn import tree

# Training data
# Input, 0 = bumpy, 1 = smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# Output, 0 = apple, 1 = orange
labels = [0, 0, 1, 1]

# Defines type of classifier
clf = tree.DecisionTreeClassifier()

# Defines our Learning Algorithm (fit is for "Finding patterns in Data")
clf = clf.fit(features, labels)
print(clf.predict([[140, 1]]))

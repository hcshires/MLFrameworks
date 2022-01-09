from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b) # calculate point distance

class ScrappyKNN(): # Our classifier (Scrappy = bare-bones)
    def fit(self, x_train, y_train):  # takes the features and labels as input
        self.x_train = x_train # features
        self.y_train = y_train # labels

    def predict(self, x_test): # inputs the features from testing data and outputs predictions
        predictions = [] # returning list of predictions
        for row in x_test:
            label = self.closest(row) # finds closest training point from test point
            predictions.append(label)
        return predictions

    def closest(self, row): # loops over all of training points and updates to the closest one so far
        bestDist = euc(row, self.x_train[0]) # records best distance from testing point to training point
        bestIndex = 0 # records the best index
        for i in range(len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < bestDist:
                bestDist = dist
                bestIndex = i
        return self.y_train[bestIndex]

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data # features
y = iris.target # labels

from sklearn.model_selection import train_test_split
# a built in function that allows for automated train/test splitting of the data. 
# features and label values are used, and 'test_size' defines how much of the data is split into train or test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# from sklearn.neighbors import KNeighborsClassifier # train a new type of classifier
clf = ScrappyKNN() # sets classifier var to class

def results(prediction): # converts results to readable strings
    data = prediction
    for i in range(len(data)):
        if predict[i] == 0:
            data[i] = "setosa"
        elif predict[i] == 1:
            data[i] = "versicolor"
        elif predict[i] == 2:
            data[i] = "virginica"
    return data

clf.fit(x_train, y_train)
predict = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predict))

results = results(predict)
print(results) # this prints the type of iris the classifier predicts for each row of testing data

'''
customData = [5.1, 3.5, 1.4, 0.2]
customPredict = clf.predict(customData)
customResults = results(customPredict)
print(customResults)
'''
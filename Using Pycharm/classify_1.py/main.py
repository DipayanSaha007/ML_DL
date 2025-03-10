# Loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading datasets
iris = datasets.load_iris()

# Printing features and labels
features = iris.data
labels = iris.target
# print(features[0],labels[0])

# Training classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[61, 1, 1, 1]])
print(preds)
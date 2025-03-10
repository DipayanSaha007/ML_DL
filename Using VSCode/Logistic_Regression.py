# Train a logistic regression classifier to predict whether a flower is iris verginica or not

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()

# print(iris.keys())
# print(iris['data'])       
# print(iris['data'].shape)  # shape of the data
# print(iris['target'])
# print(iris['DESCR'])       # description of data 

x = iris['data'][:, 3:]      # data of 3rd coloum, "FEATURES"
y = (iris['target'] == 2).astype(np.int64)   # gives [true = 1] if target = 2 (iris verginica) , otherwise gives [false = 0], "LABELS"

# print(x)
# print(y)

## Train a logistic regression classifier

clf = LogisticRegression()
clf.fit(x,y)

example = clf.predict([[2.6]])
print(example)

# Using matplotlib to plot the visualization

x_new = np.linspace(0,3,1000).reshape(-1,1)   # 1000 values of 1 feature btw 0 to 3
# print(x_new)
y_prob = clf.predict_proba(x_new)    # Predicts probability for x_new features
# print(y_prob)
plt.plot(x_new, y_prob[:,1], "g-", label = "Verginica")  # Plotting the graph
plt.show()

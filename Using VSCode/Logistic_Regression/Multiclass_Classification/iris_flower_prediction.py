import pandas as pd
import warnings
import random
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# loading the data
iris = load_iris()
# print(iris)

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# handel warning msgs
warnings.filterwarnings('ignore')

# prediction function
def prediction(pred):
    if (pred == 0):
        print("Prediction: setosa")
    elif (pred == 1):
        print("Prediction: versicolor")
    elif (pred == 2):
        print("Prediction: virginica")
    else:
        print("Prediction: <not 0,1,2>")

# fitting the model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# checking accuricy
ac = model.score(x_test, y_test)
print("Accuricy: ",ac*100)

# taking random data
rand = random.randrange(0, len(iris.target))
print("Size of iris dataset: ",len(iris.target))
print("Random data intex: ",rand)

# predicting data
pred = model.predict([iris.data[rand]])
prediction(pred)
p = iris.target_names[pred]
print("The actual data at that index: ",p)

# one more prediction
pred1 = model.predict([[5.3, 3.04, 5.07, 2.58]])
prediction(pred1)
p1 = iris.target_names[pred1]
print("The actual data at that index: ",p1)
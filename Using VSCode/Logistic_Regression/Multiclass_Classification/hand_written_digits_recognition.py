import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits  # loading sklearn inbuild data set that has 1700 8x8 hand written digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
import random

# Load dataset
digits = load_digits()

# Optionally visualize the first digit
# x = digits.data[0]
# print (x)
# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()


# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=43)
# print(len(x_train), len(x_test))

# Suppress convergence warnings for large datasets
warnings.filterwarnings('ignore')


# Initialize and fit the model
model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# Evaluate the model
ac = model.score(x_test,y_test)
print("Accuricy of the model: ",ac*100)

# taking a random data to predict
rand = random.randrange(0, 1700)
print("This is the random data index: ",rand)

# getting the image of that random digit
plt.matshow(digits.images[rand])
plt.show()

# make predictions
op = model.predict([digits.data[rand]])
print("Prediction: ",op)
p = digits.target[rand]
print("The actual digit in the image: ",p)

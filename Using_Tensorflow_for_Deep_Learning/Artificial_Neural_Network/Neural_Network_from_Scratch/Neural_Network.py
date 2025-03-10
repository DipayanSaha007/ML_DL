import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from Using_Tensorflow_for_Deep_Learning.Gradient_Descent.Gradient_Descent_from_scratch import gradient_descent

# Load dataset
df = pd.read_csv("./insurance_data-1.csv")
x = df.drop(df[['bought_insurance']], axis='columns')
y = df[['bought_insurance']]

# Scale the features
x_scaled = x.copy()
x_scaled['age'] = x_scaled['age'] / 100
# Split the data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary Cross-Entropy
def bce(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = np.clip(y_predicted, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_predicted_new) + (1 - y_true) * np.log(1 - y_predicted_new))

class myNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0
    
    def fit(self, x, y, epochs, loss_threshold):
        self.gradient_descent(x, y, epochs, loss_threshold)
    
    def predict(self, x_test):
        weighted_sum = self.w1 * x_test['age'] + self.w2 * x_test['affordibility'] + self.bias
        return sigmoid(weighted_sum)

    # Gradient Descent
    def gradient_descent(self, x_train, y_train, epochs, loss_threshold):
        # Initialize weights and bias
        w1, w2 = 1.0, 1.0
        bias = 0.0
        rate = 0.5
        n = len(x_train)
        # Convert y_train to numpy array for calculations
        y_train = y_train.values.flatten()
        for i in range(epochs):
            # Linear combination
            y_linear = w1 * x_train['age'] + w2 * x_train['affordibility'] + bias
            # Apply sigmoid function
            y_predicted = sigmoid(y_linear)
            # Compute loss
            loss = bce(y_train, y_predicted)
            # Calculate gradients
            w1d = (1 / n) * np.dot(x_train['age'], (y_predicted - y_train))
            w2d = (1 / n) * np.dot(x_train['affordibility'], (y_predicted - y_train))
            bias_d = np.mean(y_predicted - y_train)
            # Update weights and bias
            w1 -= rate * w1d
            w2 -= rate * w2d
            bias -= rate * bias_d
            # Print progress
            if i % 50 == 0 or i == epochs - 1:
                print(f"Epoch {i}: w1 = {w1:.4f}, w2 = {w2:.4f}, bias = {bias:.4f}, loss = {loss:.4f}")
            if loss<=loss_threshold:
                print(f"Epoch {i}: w1 = {w1:.4f}, w2 = {w2:.4f}, bias = {bias:.4f}, loss = {loss:.4f}")
                break
        return w1, w2, bias
    
customModel = myNN()
customModel.fit(x_train, y_train, epochs=5000, loss_threshold=0.4631)
prediction = customModel.predict(x_test)
print(prediction)
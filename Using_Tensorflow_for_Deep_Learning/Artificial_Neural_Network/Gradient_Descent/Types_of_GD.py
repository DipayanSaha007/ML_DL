from math import cos
from matplotlib.pylab import rand
from matplotlib.pyplot import sca
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random

df = pd.read_csv("./homeprices_banglore.csv")

# Min Max Scaler for scalling the dataset
sx = MinMaxScaler()
sy = MinMaxScaler()
scaled_x = sx.fit_transform(df.drop('price', axis='columns'))
scaled_y = sy.fit_transform(df['price'].values.reshape(df.shape[0], 1))
# print(scaled_y)

# Batch Gradient Descent
def batch_gradient_descent(x, y_true, epochs, learning_rate):
    number_of_features = x.shape[1]
    w = np.ones(number_of_features)
    bias = 0
    total_samples = x.shape[0]
    cost_list = []
    epoch_list = []
    for i in range(epochs):
        y_predicted = np.dot(w, x.T) + bias # w1 * area + w2 * bedrooms + bias
        # Derivatives
        w_grad = -(2/total_samples) * (x.T.dot(y_true - y_predicted))
        bias_grad = -(2/total_samples) * np.sum(y_true - y_predicted)
        w = w - learning_rate * w_grad
        bias = bias - learning_rate * bias_grad
        # Cost
        cost = np.mean(np.square(y_true - y_predicted))

        if i%50 == 0:
            cost_list.append(cost)
            epoch_list.append(i)
    return w, bias, cost, cost_list, epoch_list

w, bias, cost, cost_list, epoch_list = batch_gradient_descent(scaled_x, scaled_y.reshape(scaled_y.shape[0],), 500, 0.01)

# Plotting the Cost vs. Epoch Graph for "Batch Gradient Descent"
# plt.xlabel('epoch')
# plt.ylabel('cost')
# plt.plot(epoch_list, cost_list)
# plt.show()

# Prediction Function
def predict(area, bedrooms, w, b):
    x_scaled = sx.transform([[area, bedrooms]])[0]
    scaled_price = w[0]*x_scaled[0] + w[1]*x_scaled[1] + b
    return sy.inverse_transform([[scaled_price]])[0][0]

print("BGD Price = ", predict(2600, 4, w, bias))

# Stochastic Gradient Descent
def stochastic_gradient_descent(x, y_true, epochs, learning_rate):
    number_of_features = x.shape[1]
    w = np.ones(shape=(number_of_features))
    bias = 0
    total_samples = x.shape[0]
    cost_list = []
    epoch_list = []
    for i in range(epochs):
        random_index = random.randint(0, total_samples-1)
        sample_x = x[random_index]
        sample_y = y_true[random_index]
        y_predicted = np.dot(w, sample_x.T) + bias # w1 * area + w2 * bedrooms + bias
        # Derivatives
        w_grad = -(2/total_samples) * (sample_x.T.dot(sample_y - y_predicted))
        bias_grad = -(2/total_samples) * np.sum(sample_y - y_predicted)
        w = w - learning_rate * w_grad
        bias = bias - learning_rate * bias_grad
        # Cost
        cost = np.mean(np.square(sample_y - y_predicted))

        if i%50 == 0:
            cost_list.append(cost)
            epoch_list.append(i)
    return w, bias, cost, cost_list, epoch_list

w_sgd, bias_sgd, cost_sgd, cost_list_sgd, epoch_list_sgd = stochastic_gradient_descent(scaled_x, scaled_y.reshape(scaled_y.shape[0],), 10000, 0.01)

# # Plotting the Cost vs. Epoch Graph for "Stochastic Gradient Descent"
# plt.xlabel('epoch')
# plt.ylabel('cost')
# plt.plot(epoch_list_sgd, cost_list_sgd)
# plt.show()

print("SGD Price = ", predict(2600, 4, w_sgd, bias_sgd))

# Mini-batch Gradient Descent
def mini_batch_gradient_descent(x, y_true, epochs, learning_rate):
    number_of_features = x.shape[1]
    w = np.ones(shape=number_of_features)
    bias = 0
    total_samples = x.shape[0]
    cost_list = []
    epoch_list = []
    for i in range(epochs):
        random_index1 = random.randint(0, total_samples-1)
        random_index2 = random.randint(random_index1, total_samples-1)
        sample_x = x[random_index1:random_index2]
        sample_y = y_true[random_index1:random_index2]
        y_predicted = np.dot(w, sample_x.T) + bias  # w1 * area + w2 * bedrooms + bias
        # Derivatives
        w_grad = -(2/total_samples) * sample_x.T.dot(sample_y - y_predicted)
        bias_grad = -(2/total_samples) * np.sum(sample_y - y_predicted)
        # New weights & bias
        w = w - learning_rate * w_grad
        bias = learning_rate * bias_grad
        # Cost
        cost = np.mean(np.square(sample_y - y_predicted))

        if i%50 == 0:
            cost_list.append(cost)
            epoch_list.append(i)
    return w, bias, cost, cost_list, epoch_list

w_mbgd, bias_mbgd, cost_mbgd, cost_list_mbgd, epoch_list_mbgd = mini_batch_gradient_descent(scaled_x, scaled_y.reshape(scaled_y.shape[0],), 2000, 0.01)

print("MBGD Price = ", predict(2600, 4, w_mbgd, bias_mbgd))
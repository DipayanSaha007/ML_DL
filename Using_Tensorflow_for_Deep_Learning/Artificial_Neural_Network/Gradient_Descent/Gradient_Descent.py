import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

df = pd.read_csv("./insurance_data-1.csv")   # C:\Users\User\OneDrive\Desktop\ML Projects\Using_Tensorflow_for_Deep_Learning\Gradient_Descent\insurance_data-1.csv
# print(df.head())

x = df[['age', 'affordibility']]
y = df['bought_insurance']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(x_train)

# Scalling data
x_train_scaled = x_train.copy()
x_train_scaled['age'] = x_train_scaled['age'] / 100
x_test_scaled = x_test.copy()
x_test_scaled['age'] = x_test_scaled['age'] / 100
# print(x_train_scaled)

model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_scaled, y_train, epochs=5000)
model.predict(x_test)
loss, accuricy = model.evaluate(x_test_scaled, y_test)
# print("Accuricy = ", accuricy*100)
# print("Loss = ", loss*100)

## For getting the prediction
coef, intercept = model.get_weights()
print("Coef = ", coef, ", Intercept = ", intercept)
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))
# def prediction_function(age, affordability):
#     weighted_sum = coef[0]*age + coef[1]*affordability + intercept
#     return sigmoid(weighted_sum)

# # From this code
# Coef = [[5.1156354], [1.8614864]]
# Intercept = [-3.2415366]
# loss = 0.4044
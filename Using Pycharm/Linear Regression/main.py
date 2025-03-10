import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

#print(diabetes.keys()) --> [dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])]
#print(diabetes.DESCR)

diabetes_x = diabetes.data
#print(diabetes_x)

diabetes_x_train = diabetes_x[:-30]                               # data for training the model <features>
diabetes_x_test = diabetes_x[-30:]                                # data for testing  the model <features>

diabetes_y_train = diabetes.target[:-30]                          # <label corresponding features>
diabetes_y_test = diabetes.target[-30:]                           # <label corresponding features>

model = linear_model.LinearRegression()                           # making the model
model.fit(diabetes_x_train,diabetes_y_train)                      # fedding data to model

diabetes_y_predict = model.predict(diabetes_x_test)               # testing the model
# accuiricy determining
print("Mean Squared Error: ", mean_squared_error(diabetes_y_test,diabetes_y_predict))  # testing the mean square error over the testing data

print("Weights: ", model.coef_)                                  # checking weight -> w1, w2 ,w3,....
print("Intercept: ", model.intercept_)                           # cheking intercept -> w0

#plt.scatter(diabetes_x_test, diabetes_y_test)   plotting the scatter data in graph
#plt.plot(diabetes_x_test, diabetes_y_predict)   plotting line in graph
#plt.show()      showing the data in graph

# after using > diabetes_x = diabetes.data[:, np.newaxis, 2] <
#Mean Squared Error:  3035.060115291269
#Weights:  [941.43097333]
#Intercept:  153.39713623331644

# after using > diabetes_x = diabetes.data <
#Mean Squared Error:  1826.4841712795046
#Weights:  [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067  458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
#Intercept:  153.05824267739402
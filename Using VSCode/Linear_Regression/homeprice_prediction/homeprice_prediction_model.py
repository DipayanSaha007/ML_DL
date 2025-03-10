import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#loading the xlsx data
df = pd.read_excel(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Linear_Regression\homeprice_prediction\houseprice.xlsx")

#linear regression model creation
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df[['price']])

# if i want a user input
x = float(input("Enter the area of which you want to predict the price: "))

# Predicting data with a DataFrame to match feature names
prediction_input = pd.DataFrame({'area': [x]})
prediction = reg.predict(prediction_input)
print("The house price corresponding to the area: ",prediction)

# as price = m * area + b, where m = coefficient, b = intercept
m = reg.coef_
b = reg.intercept_
print("Coef(m), intercept(b): ",m,b)

# plotting the graph to get a visual representation
plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.price,color='red')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()

#if i have a xlsx file with areas whose prices i hv to predict & store in another xl file
d = pd.read_excel(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Linear_Regression\homeprice_prediction\price_predict.xlsx")

p = reg.predict(d)
d['price'] = p

d.to_excel(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Linear_Regression\homeprice_prediction\predictions.xlsx")

# saving model using pickle

import pickle
with open('model_pickle','wb') as f:
    pickle.dump(reg,f)
with open('model_pickle','rb') as f:
    mp = pickle.load(f)
x1 = mp.predict(np.array([[5000]]))
print("x1: ",x1)

# saving model using joblib

import joblib
joblib.dump(reg,'model_joblib')
mj = joblib.load('model_joblib')
x2 = mj.predict(np.array([[5000]]))
print("x2: ",x2)
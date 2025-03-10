import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# reading data
df = pd.read_excel(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Multivariable_Regression\houseprice_prediction_using_multiplevariable\house_prices_multivariable.xlsx")

# filling null values with median value
median_bedrooms = int(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

# training the model & getting the inputs
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df[['price']])
x1 = float(input("Enter the area: "))
x2 = float(input("Enter the no. of bedrooms: "))
x3 = float(input("Enter the age: "))

# predicting the price
prediction_input = pd.DataFrame({'area' : [x1], 'bedrooms' : [x2], 'age' : [x3]})
prediction = reg.predict(prediction_input)
print("Predicted Price: ",prediction)

# as price = m1*x1 + m2*x2 + m3*x3 + b
mi = reg.coef_
b = reg.intercept_
print("Coefs & Intercept:  ",mi,b)


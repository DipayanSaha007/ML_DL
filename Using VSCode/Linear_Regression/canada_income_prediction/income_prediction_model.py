import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# reading data 7 creating model
df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\canada_income_prediction\canada_per_capita_income.csv")
reg = linear_model.LinearRegression()
reg.fit(df[['year']],df[['per capita income (US$)']])

# predicting data
x = float(input("Enter year to predict Canadas per capita income: "))
predict_input = pd.DataFrame({'year' : [x]})
prediction = reg.predict(predict_input)
print("The predicted per capita income: ",prediction)

# plotting graph
plt.xlabel('Year')
plt.ylabel('per capita income (US$)')
plt.scatter(df[['year']],df[['per capita income (US$)']],color='red')
plt.plot(df[['year']],reg.predict(df[['year']]),color='blue')
plt.show()

# reading a whole excel file with years to predict the per capita income
d = pd.read_excel(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\canada_income_prediction\income_predict.xlsx")
p = reg.predict(d)
d['per capita income (US$)'] = p
d.to_excel(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\canada_income_prediction\predicted_incomes.xlsx")
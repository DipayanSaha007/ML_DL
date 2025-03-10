from word2number import w2n
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Multivariable_Regression\salary_prediction_multivariable\hiring.csv")

df.experience = df.experience.fillna('zero')
df.experience = df.experience.apply(lambda x: w2n.word_to_num(x) if isinstance(x, str) else x)
test_score_median = int(df.test_score_outof10.median())
df.test_score_outof10 = df.test_score_outof10.fillna(test_score_median)
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score_outof10','interview_score_outof10']],df[['salary($)']])
x1 = float(input("Enter the experience: "))
x2 = float(input("Enter the test_score_outof10: "))
x3 = float(input("Enter the interview_score_outof10: "))

prediction_input = pd.DataFrame({'experience' : [x1], 'test_score_outof10' : [x2], 'interview_score_outof10' : [x3]})
prediction = reg.predict(prediction_input)
print("The salary: ",prediction)

m = reg.coef_
b = reg.intercept_
print("The Coefs & Intercepts: ",m, b)

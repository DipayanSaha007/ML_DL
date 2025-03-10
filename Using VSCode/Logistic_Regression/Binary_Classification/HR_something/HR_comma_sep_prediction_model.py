import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Logistic_Regression\Binary_Classification\HR_something\HR_comma_sep.csv")

# plt.scatter(df.last_evaluation,df.left, marker= '+', color= 'red')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(df[['last_evaluation']],df.left, test_size= 0.01)

model = LogisticRegression()
model.fit(x_train,y_train)
print("Prediction values: \n",x_test)
op = model.predict(x_test)
print("Prediction: \n",op)

ac = model.score(x_test,y_test)
print("Accuricy: ",ac)
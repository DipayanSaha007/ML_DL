import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Logistic_Regression\Binary_Classification\insurence_prediction\insurance_data.csv")

# plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
# plt.show()

# Splitting the data set into train & test
x_train, x_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance, test_size=0.2)
print("The testing age: \n",x_test)

# Creating model
reg = LogisticRegression()
reg.fit(x_train,y_train)

# Predicting the value
op = reg.predict(x_test)
print("Prediction: ",op)

# Checking Accuricy
ac = reg.score(x_test,y_test)
print("Accuricy: ",ac)
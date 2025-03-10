import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\OneHotEncoding\Car_Price_Prediction\carprices_onehot.csv")

dummies = pd.get_dummies(df['Car Model'])
dummies = dummies.astype(int)
merged = pd.concat([df,dummies],axis='columns')
final = merged.drop(['Car Model','Audi A5'],axis='columns')
print(final)

model = LinearRegression()
x = final.drop('Sell Price($)',axis='columns')
y = final['Sell Price($)']
model.fit(x,y)
input1 = pd.DataFrame({'Mileage': [45000], 'Age(yrs)': [4], 'BMW X5': [0], 'Mercedez Benz C class': [1]})
input2 = pd.DataFrame({'Mileage': [86000], 'Age(yrs)': [7], 'BMW X5': [1], 'Mercedez Benz C class': [0]})
y1 = model.predict(input1)
y2 = model.predict(input2)
print("Predicted Price: ",y1,y2)

ac = model.score(x,y)
print("Accuricy: ",ac)
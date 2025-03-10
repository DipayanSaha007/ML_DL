import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Splitting_of_Dataset_in_Training&Testing\carprice_spliting.csv")

x = df[['Mileage','Age(yrs)']]
y = df[['Sell Price($)']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

model = LinearRegression()
model.fit(x_train,y_train)
op = model.predict(x_test)
print("The Predicted Value: ",op)

ac = model.score(x_test,y_test)
print("Accuricy: ",ac)
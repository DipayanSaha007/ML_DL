import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\OneHotEncoding\homeprices_onehot.csv")

# implementing ONE_HOT_ENCODING_USING_PANDAS 
dummies = pd.get_dummies(df.town)
dummies = dummies.astype(int)
merged = pd.concat([df,dummies],axis='columns')

# dropping the original variable(town) & a dummy variable to avoid "Dummy_Variable_Trap"
# sklearn linear_model auto drops the dummy variable cause it is aware of the "Dummy_Variable_Trap"
# but it is a good practice to drop it 
final = merged.drop(['town','west windsor'],axis='columns')
print(final)

# creating the model
model = LinearRegression()
x = final.drop('price',axis='columns')
y = final.price
model.fit(x,y)
prediction_input = pd.DataFrame({'area': [2800], 'monroe township': [0], 'robinsville': [1]})
y1 = model.predict(prediction_input)
print(y1)

# cheking the accuricy of the model
ac = model.score(x,y)
print("Accuricy: ",ac)

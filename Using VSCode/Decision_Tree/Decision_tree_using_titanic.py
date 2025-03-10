import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split

# loading data
df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Decision_Tree\titanic.csv")
inputs = df.drop(['Survived','PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis='columns')
target = df['Survived']

# creating new 'inputs_n'
lb_Sex = LabelEncoder()
inputs['Sex_b'] = lb_Sex.fit_transform(inputs['Sex'])
inputs_n = inputs.drop(['Sex'], axis='columns')

# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.1, random_state=42)

# fitting the model
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
ac = model.score(x_test, y_test)
print("Accuricy: ",ac*100)

# predicting the data
pred = model.predict(x_test)
print("Prediction: ",pred)
print("Actual data: \n",y_test)
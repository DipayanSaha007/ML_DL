import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Naive-Bayes-Classification\titanic.csv")
df.drop(df[['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']], axis='columns', inplace=True)
# print(df.head())
target = df.Survived
inputs = df.drop('Survived', axis='columns')
dummies = pd.get_dummies(inputs.Sex)
dummies = dummies.astype(int)
inputs = pd.concat([inputs, dummies], axis='columns')
inputs.drop('Sex', axis='columns', inplace=True)
# print(inputs.head())
# print(inputs.columns[inputs.isna().any()])      # It shows if there is some 'NaN' values in the dataframe; here 'Age' has some NaN
inputs.Age = inputs.Age.fillna(inputs.Age.mean())   # As 'Age' had some NaN values we do this to fill those values with the mean of 'Age'

x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

# Gaussian Naive Bayes Model
model = GaussianNB()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
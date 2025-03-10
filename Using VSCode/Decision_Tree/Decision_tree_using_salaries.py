import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split

# loading dataset
df = pd.read_csv("./salaries.csv")     # "C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Decision_Tree\salaries.csv"
inputs = df.drop('salary_more_then_100k',axis='columns')
target = df['salary_more_then_100k']

# creating new datas in 'inputs' & 'inputs_n'
lb_company = LabelEncoder()
lb_job = LabelEncoder()
lb_degree = LabelEncoder()
inputs['company_n'] = lb_company.fit_transform(inputs['company'])
inputs['job_n'] = lb_job.fit_transform(inputs['job'])
inputs['degree_n'] = lb_degree.fit_transform(inputs['degree'])
inputs_n = inputs.drop(['company','job','degree'],axis='columns')

# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.2)

# creating model using Decision Tree
model = tree.DecisionTreeClassifier(splitter='random')
model.fit(x_train, y_train)
ac = model.score(x_test, y_test)
print("Accuricy: ",ac*100)

# predicting data
pred = model.predict(x_test)
print("Prediction: ",pred)
print("Actual data: \n",y_test)
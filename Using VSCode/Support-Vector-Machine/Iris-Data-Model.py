import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC     # Support Vetor Machine (svm) classifier "SVC"
from matplotlib import pyplot as plt   # for data visualization

iris = load_iris()
# print(dir(iris))
# print(iris.feature_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)    # making a dataframe
df['target'] = iris.target      # making a target column with the target value [0, 1 or 2]
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])     # target flower names will be added, run print(df) each step for more info
# print(df)

# for data visualization
# df0 = df[df.target == 0]
# df1 = df[df.target == 1]
# df2 = df[df.target == 2]
# plt.xlabel('sapel length(cm)')
# plt.ylabel('sapel width(cm)')
# plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='blue', marker='*')
# plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='green', marker='+')
# plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color='red', marker='.')
# plt.show()

# Dropping the extra added colomns
x = df.drop(['target', 'flower_name'], axis='columns')
# print(x.head())
y = df.target
# print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Creating model
# model = SVC()   # The parameter 'C' is by default 1.0
# model = SVC(C=10)   # The parameter 'C' is now 10
# model = SVC(gamma = 100)
model = SVC(kernel='linear')    # Another parameter 'kernel'
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
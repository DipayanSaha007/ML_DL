import pandas as pd
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

iris = load_iris()
# print(dir(iris))
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
# print(df.head())

x = df.drop('target', axis='columns')
y = df.target
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# model = SVC(C=30, gamma='auto', kernel='rbf')
# model.fit(x_train, y_train)
# print(model.score(x_test, y_test))

clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False)

clf.fit(x, y)
# print(clf.cv_results_)
df1 = pd.DataFrame(clf.cv_results_)
# print(df1.head())
print(clf.best_params_, " ", clf.best_score_)
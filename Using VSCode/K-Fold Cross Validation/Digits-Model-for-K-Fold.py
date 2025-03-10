import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold    # Basic K-Fold
from sklearn.model_selection import StratifiedKFold    # Similer to K-Fold but it also devides each classification categories in a uniform way while separating Folds
from sklearn.model_selection import cross_val_score     # Does the same thing as lines 41 to 49 and lines 16 to 20

digits = load_digits()
# x = digits.data
# y = digits.target
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# # Method for the models
# def get_score(model, x_train, x_test, y_train, y_test):
#     model.fit(x_train, y_train)
#     # print(model, " Score = ", 100*model.score(x_test, y_test), "%")
#     return 100*model.score(x_test, y_test)

# # Logistic Regression Model
# lr = LogisticRegression()
# lr.fit(x_train, y_train)
# print("Logistic Regression Score = ", 100*lr.score(x_test, y_test), "%")

# # Support Vector Machine
# svm = SVC()
# svm.fit(x_train, y_train)
# print("SVM Score = ", 100*svm.score(x_test, y_test), "%")

# # Random Forest Classifier
# rf = RandomForestClassifier()
# rf.fit(x_train, y_train)
# print("Random Forest Score = ", 100*rf.score(x_test, y_test), "%")

# get_score(LogisticRegression(max_iter=1000), x_train, x_test, y_train, y_test)
# get_score(SVC(), x_train, x_test, y_train, y_test)
# get_score(RandomForestClassifier(), x_train, x_test, y_train, y_test)

# # Sratified K-Fold for UNDRSTANDING PURPOSE ONLY->
# folds = StratifiedKFold(n_splits=3)      # 'n_splits' represents number of folds
# scores_lr, scores_svm, scores_rf = [], [], []
# # 'train_index' is for 'x' & 'test_index' is for 'y'; and 'digits.data' is for 'x' & 'digits.target' is for 'y'
# for train_index, test_index in folds.split(digits.data, digits.target):
#     x_train, x_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]
#     scores_lr.append(get_score(LogisticRegression(max_iter=1000), x_train, x_test, y_train, y_test))
#     scores_svm.append(get_score(SVC(kernel='linear'), x_train, x_test, y_train, y_test))
#     scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), x_train, x_test, y_train, y_test))

scores_lr, scores_svm, scores_rf = [], [], []

scores_lr = cross_val_score(LogisticRegression(max_iter=1000), digits.data, digits.target, cv=3)    # 'cv' mentions the no. of folds, default is 5
scores_svm = cross_val_score(SVC(kernel='linear'), digits.data, digits.target, cv=3)
scores_rf = cross_val_score(RandomForestClassifier(n_estimators=50), digits.data, digits.target, cv=3)

print("Logistic Regression Score = ",scores_lr)
print("SVM Score = ",scores_svm)
print("Random Forest Score = ",scores_rf)
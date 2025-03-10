from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

iris = load_iris()

scores_lr, scores_svm, scores_rf, scores_dt = [], [], [], []

scores_lr = cross_val_score(LogisticRegression(max_iter=1000), iris.data, iris.target, cv=3)
scores_svm = cross_val_score(SVC(gamma=4, kernel='linear'), iris.data, iris.target, cv=3)
scores_rf = cross_val_score(RandomForestClassifier(n_estimators=10), iris.data, iris.target, cv=3)
scores_dt = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target, cv=3)

print("Logistic Regression Score = ",scores_lr)
print("SVM Score = ",scores_svm)
print("Random Forest Score = ",scores_rf)
print("Decision Tree Score = ",scores_dt)
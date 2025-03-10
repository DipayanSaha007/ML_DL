# We can use this code to select the proper model selection based on 'best_score' & 'best_param'

# import pandas as pd
from sklearn import datasets
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
from my_package import Best_Model_and_Parameters

iris = datasets.load_iris()

# These codes are in the 'Best-Model-and-Parameters.py'
# model_params = {
#     'svm': {
#         'model': svm.SVC(gamma='auto'),
#         'params' : {
#             'C': [1,10,20],
#             'kernel': ['rbf','linear']
#         }  
#     },
#     'random_forest': {
#         'model': RandomForestClassifier(),
#         'params' : {
#             'n_estimators': [1,5,10]
#         }
#     },
#     'logistic_regression' : {
#         'model': LogisticRegression(solver='liblinear'),
#         'params': {
#             'C': [1,5,10]
#         }
#     }
# }

# scores = []
# for model_name, mp in model_params.items():
#     clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
#     clf.fit(iris.data, iris.target)
#     scores.append({
#         'model': model_name,
#         'best_score': clf.best_score_,
#         'best_params': clf.best_params_
#     })
    
# df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# print(df)

Best_Model_and_Parameters.get_best(iris.data, iris.target)
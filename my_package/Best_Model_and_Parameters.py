import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

model_params = {
    'svm': {
        'model': svm.SVC(probability=True),
        'params': {
            'model__gamma': ['auto', 'scale'],  # Corrected: Add 'model__' to target the model step
            'model__C': [1, 10, 20],            # Corrected: Add 'model__' to target the model step
            'model__kernel': ['rbf', 'linear']  # Corrected: Add 'model__' to target the model step
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [1, 5, 10]  # Corrected: Add 'model__' to target the model step
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [1, 5, 10],           # Corrected: Add 'model__' to target the model step
            'model__solver': ['liblinear']    # Corrected: Add 'model__' to target the model step
        }
    },
    'decision_tree': {
        'model': tree.DecisionTreeClassifier(),
        'params': {
            'model__splitter': ['random', 'best']  # Corrected: Add 'model__' to target the model step
        }
    },
    'gaussian_NB': {
        'model': GaussianNB(),
        'params': {}
    }
    # 'multinomial_NB': {
    #     'model': MultinomialNB(),
    #     'params': {}
    # }
}
best_estimator = {}
def get_best(x, y):
    scores = []
    for model_name, mp in model_params.items():
        pipe = Pipeline([
            ('scaling', StandardScaler()),  # Standardization step
            ('model', mp['model'])          # Model step
        ])
        
        # Corrected: Use 'param_grid' with model-specific parameters
        clf = GridSearchCV(pipe, param_grid=mp['params'], cv=5, return_train_score=False)
        
        clf.fit(x, y)
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })
        best_estimator[model_name] = clf.best_estimator_
    df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    return df

def get_best_estimator():
    return best_estimator
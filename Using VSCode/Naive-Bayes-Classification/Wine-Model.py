import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
# print(df.head())

x = df.drop('target', axis='columns')
y = df.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

def get_score(model):
    clf = Pipeline([
        ('nb', model)
    ])
    clf.fit(x_train, y_train)
    print(model, " Score = ", 100*clf.score(x_test, y_test))

get_score(GaussianNB())
get_score(MultinomialNB())
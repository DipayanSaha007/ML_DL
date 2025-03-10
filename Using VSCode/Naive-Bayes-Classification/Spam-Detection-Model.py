import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer     # Used to represent the 'Message' part of the csv to numbers
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline   # Instead of lines 14 to 22, we can do lines 24 to 29

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\Naive-Bayes-Classification\spam.csv")
# print(df.groupby('Category').describe())
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
# print(df.head())

x_train, x_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)

# v = CountVectorizer()
# x_train_count = v.fit_transform(x_train.values)
# x_train_count.toarray()[:3]

# model = MultinomialNB()
# model.fit(x_train_count, y_train)

# x_test_count = v.transform(x_test)
# print(100*model.score(x_test_count, y_test))

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(x_train, y_train)
print(100*clf.score(x_test, y_test))
import pandas as pd
from sklearn.datasets import load_digits    # Digits datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()
# print(dir(digits))
df = pd.DataFrame(digits.data, columns=digits.feature_names)
df['target'] = digits.target
# print(df.head())
x = df.drop(['target'], axis='columns')
# print(x)
y = df.target
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# model = SVC()   # Score = 98.33%
model = SVC(C=10)   # Score = 99.166%
# model = SVC(kernel='linear')    # # Score = 98.055%
# model = SVC(kernel='rbf')   # Score = 98.88%
# model = SVC(gamma=10)   # Score = 7.5%
model.fit(x_train, y_train)
print(100*model.score(x_test, y_test), "%")
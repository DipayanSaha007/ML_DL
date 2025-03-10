import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier     # 'ensemble' is used when we use multiple algorithms to predict the outcome
from sklearn.metrics import confusion_matrix       # Used to plot truth vs prediction
from matplotlib import pyplot as plt
import seaborn as sn

digits = load_digits()

df = pd.DataFrame(digits.data)
df['target'] = digits.target

x = df.drop(['target'], axis='columns')
y = df.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier()    # By default 10random trees
# model = RandomForestClassifier(n_estimators=20)     # 20 random trees
# model = RandomForestClassifier(n_estimators=30)     # 30 random trees
model.fit(x_train, y_train)
print(100*model.score(x_test, y_test), "%")

# For visualization of confusion matrix
y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
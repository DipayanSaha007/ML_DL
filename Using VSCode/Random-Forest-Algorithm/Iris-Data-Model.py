import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sn

iris = load_iris()
df = pd.DataFrame(iris.data)
df['target'] = iris.target

x = df.drop(['target'], axis='columns')
y = df.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=10)     # Default 'n_estimators' value
model.fit(x_train, y_train)
print(100 * model.score(x_test, y_test), "%")

y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
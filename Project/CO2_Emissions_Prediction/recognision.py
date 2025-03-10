import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load Dataset
data = pd.read_csv("./co2.csv")

# Check for null values
data.isnull().sum()  # If null values are found, use `data.fillna()` to handle them.

# Exploratory Data Analysis
data.describe()
data.hist(figsize=(12, 7))
# plt.show()

# Initialize and Fit Encoders
encoders = {
    "Make": LabelEncoder(),
    "Model": LabelEncoder(),
    "Vehicle Class": LabelEncoder(),
    "Transmission": LabelEncoder(),
    "Fuel Type": LabelEncoder(),
}
for column, encoder in encoders.items():
    data[column] = encoder.fit_transform(data[column])

# Save Encoders for Future Use
with open("Encoders.pkl", "wb") as file:
    pkl.dump(encoders, file)

# Standard Scaling
scaler = StandardScaler()
columnsToScale = [
    'Make', 'Model', 'Vehicle Class', 'Engine Size(L)', 'Cylinders',
    'Transmission', 'Fuel Type', 'Fuel Consumption City (L/100 km)',
    'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
    'Fuel Consumption Comb (mpg)', 'CO2 Emissions(g/km)'
]
data_sc = scaler.fit_transform(data[columnsToScale])
data_sc_df = pd.DataFrame(data_sc, columns=columnsToScale)

# Save Scaler
with open("Scaler.pkl", "wb") as file:
    pkl.dump(scaler, file)

# Correlation Heatmap
sns.heatmap(data_sc_df.corr(), annot=True)
# plt.show()

# Prepare Data for Training
x = data_sc_df.drop("CO2 Emissions(g/km)", axis="columns")
y = data["CO2 Emissions(g/km)"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train Linear Regression Model
model = DecisionTreeClassifier(splitter='random')
model.fit(x_train, y_train)

# Save Model
with open("Model.pkl", "wb") as file:
    pkl.dump(model, file)

# Evaluate Model
y_pred = model.predict(x_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("r2_score for Linear Regression = ", r2)
print("mean_absolute_error for Linear Regression = ", mae)
print("mean_squared_error for Linear Regression = ", mse)
# import my_package.Best_Model_and_Parameters
# best_model = my_package.Best_Model_and_Parameters.get_best(x_train, y_train)
# print(best_model)
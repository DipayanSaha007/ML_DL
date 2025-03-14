import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\K-Means-Clustering-Algo\income.csv")

# For Checking the value of 'k'
# plt.scatter(df['Age'], df['Income($)'])
# plt.show()

scaler = MinMaxScaler()     # Used to scale the data
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
# print(df.head())

# km = KMeans(n_clusters=3)   # 3 cluster values we get from the plot
# y_predicted = km.fit_predict(df[['Age', 'Income($)']])
# # print(y_predicted)
# df['cluster'] = y_predicted

# df1 = df[df.cluster==0]
# df2 = df[df.cluster==1]
# df3 = df[df.cluster==2]
# plt.scatter(df1.Age, df1['Income($)'], color='green')
# plt.scatter(df2.Age, df2['Income($)'], color='blue')
# plt.scatter(df3.Age, df3['Income($)'], color='red')
# plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='black', marker='*')      # Used to plot the centroids
# plt.xlabel('Age')
# plt.ylabel('Income($)')
# plt.show()

# Instead of lines 19 to 33 we do 35 to 48
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)   # it will give the SSE
# print(sse)

# For Elbow Technique
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_rng, sse)
plt.show()
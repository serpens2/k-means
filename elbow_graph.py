from kmeans import kmeans
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Iris.csv')
X = df.drop(columns=['Species', 'Id']).values

WCSS_arr = []
k_arr = [k for k in range(2,6)]
for k in k_arr:
    _, WCSS = kmeans(X, K=k)
    WCSS_arr.append(WCSS)
plt.plot(k_arr, WCSS_arr, 'o-', color='black', markersize=5)
plt.grid()
plt.xlabel('K')
plt.ylabel('WCSS')
plt.show()
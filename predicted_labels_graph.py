from kmeans import kmeans
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Iris.csv')
X = df.drop(columns=['Species', 'Id']).values

y_pred, WCSS = kmeans(X,K=3)
dict_ = {0:'green', 1:'red', 2:'blue'}

fig,axs = plt.subplots(2,3)
for i in range(len(X)):
    axs[0,0].plot( X[i,0],X[i,1], 'o', markersize = 3, color = dict_[y_pred[i]])
    axs[0,1].plot( X[i,0],X[i,2], 'o', markersize = 3, color = dict_[y_pred[i]])
    axs[0,2].plot( X[i,0],X[i,3], 'o', markersize = 3, color = dict_[y_pred[i]])
    axs[1,0].plot( X[i,1],X[i,2], 'o', markersize = 3, color = dict_[y_pred[i]])
    axs[1,1].plot( X[i,1],X[i,3], 'o', markersize = 3, color = dict_[y_pred[i]])
    axs[1,2].plot( X[i,2],X[i,3], 'o', markersize = 3, color = dict_[y_pred[i]])
axs[0,0].set(xlabel = 'SepalLengthCm',ylabel='SepalWidthCm')
axs[0,1].set(xlabel = 'SepalLengthCm',ylabel='PetalLengthCm')
axs[0,2].set(xlabel = 'SepalLengthCm',ylabel='PetalWidthCm')
axs[1,0].set(xlabel = 'SepalWidthCm',ylabel='PetalLengthCm')
axs[1,1].set(xlabel = 'SepalWidthCm',ylabel='PetalWidthCm')
axs[1,2].set(xlabel = 'PetalLengthCm',ylabel='PetalWidthCm')
plt.show()

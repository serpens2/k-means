import pandas as pd
import matplotlib.pyplot as plt

dict_ = {'Iris-setosa':'red', 'Iris-virginica':'blue','Iris-versicolor':'green'}
df = pd.read_csv('Iris.csv')
X = df.drop(columns=['Species','Id']).values
y = df['Species'].values
X_y = [ [X[i],y[i]] for i in range(len(X)) ]

fig,axs = plt.subplots(2,3)
for i in range(len(X)):
    axs[0,0].plot( X[i,0],X[i,1], 'o', markersize = 3, color = dict_[y[i]])
    axs[0,1].plot( X[i,0],X[i,2], 'o', markersize = 3, color = dict_[y[i]])
    axs[0,2].plot( X[i,0],X[i,3], 'o', markersize = 3, color = dict_[y[i]])
    axs[1,0].plot( X[i,1],X[i,2], 'o', markersize = 3, color = dict_[y[i]])
    axs[1,1].plot( X[i,1],X[i,3], 'o', markersize = 3, color = dict_[y[i]])
    axs[1,2].plot( X[i,2],X[i,3], 'o', markersize = 3, color = dict_[y[i]])
axs[0,0].set(xlabel = 'SepalLengthCm',ylabel='SepalWidthCm')
axs[0,1].set(xlabel = 'SepalLengthCm',ylabel='PetalLengthCm')
axs[0,2].set(xlabel = 'SepalLengthCm',ylabel='PetalWidthCm')
axs[1,0].set(xlabel = 'SepalWidthCm',ylabel='PetalLengthCm')
axs[1,1].set(xlabel = 'SepalWidthCm',ylabel='PetalWidthCm')
axs[1,2].set(xlabel = 'PetalLengthCm',ylabel='PetalWidthCm')
plt.show()
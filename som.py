# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
dataset = dataset.dropna()
X = dataset.iloc[:, 1:]
'''X = X.drop(["Name"], axis = 1)
ohe = pd.get_dummies(dataset["Sector"])
X = pd.concat([ohe,X],axis = 1)
'''
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 12291, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 'x']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[1],
         markeredgecolor = colors[1],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
grp1 = np.concatenate((mappings[(6,2)], mappings[(3,5)], mappings[(1,6)], mappings[(2,7)]), axis = 0)
grp1 = sc.inverse_transform(grp1)
grp2 = np.concatenate((mappings[(0,0)], mappings[(0,9)], mappings[(9,0)], mappings[(8,9)], mappings[(9,8)], mappings[(9,4)]), axis = 0)
grp2 = sc.inverse_transform(grp2)
grp3 = np.concatenate((mappings[(1,0)], mappings[(5,5)], mappings[(8,3)], mappings[(5,9)]), axis = 0)
grp3 = sc.inverse_transform(grp3)
grp4 = np.concatenate((mappings[(0,4)], mappings[(0,2)], mappings[(8,2)], mappings[(3,9)], mappings[(0,7)]), axis = 0)
grp4 = sc.inverse_transform(grp4)
grp5 = np.concatenate((mappings[(0,1)], mappings[(2,9)], mappings[(7,9)]), axis = 0)
grp5 = sc.inverse_transform(grp5)
grp6 = np.concatenate((mappings[(5,9)], mappings[(8,6)], mappings[(8,4)]), axis = 0)
grp6 = sc.inverse_transform(grp6)
grp7 = np.concatenate((mappings[(1,8)], mappings[(3,3)], mappings[(4,2)], mappings[(2,1)], mappings[(5,0)], mappings[(1,2)], mappings[(4,3)]), axis = 0)
grp7 = sc.inverse_transform(grp7)
grp8 = np.concatenate((mappings[(1,1)], mappings[(3,2)], mappings[(3,8)], mappings[(6,7)]), axis = 0)
grp8 = sc.inverse_transform(grp8)
grp9 = np.concatenate((mappings[(2,3)], mappings[(2,5)], mappings[(4,7)]), axis = 0)
grp9 = sc.inverse_transform(grp9)
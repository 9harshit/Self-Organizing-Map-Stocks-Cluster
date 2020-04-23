# Self-Organizing-Map-Stocks-Cluster

This project uses Self Organizing Map to create cluster of stocks with similar movement in prices or similarity in features.

The dataset contains 10 years of historic data of 50 different stocks.

You can tweak the SOM by making changes in minisom.py

To run the file keep som.py in the same folder as minisom.py and __pycache__.

The stocks are grouped based on the similar coloured position in the map

figure.png file shows the generated SOM along with markings.

grps.npy file contains the cluster that were obtained from the SOM.

Cluster.txt contains the list of stocks in respective cluster obtained.

Note :- Since this a unsupervised learning algorithm, running the file each time may generate different outputs and maps. So your generated output may not be same as output in this repository


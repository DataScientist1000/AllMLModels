# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans

path =  os.path.abspath("Clustering/KMeans/Data/Wholesale_customers.csv")
data = pd.read_csv(path)
#----------EDA------------
data.info()
data.head()
data.Channel.value_counts()
data.Region.value_counts()
data.describe()
#-----------------------------------
X = data.iloc[:,2:]


Clusters = KMeans(n_clusters=3)

Clusters.fit(X)

Clusters.cluster_centers_
Clusters.get_params

Clusters.inertia_

Clusters.n_clusters
#-----------------------------------
wcss = []
for i in range(1,11):
    kmeans = KMeans( init='k-means++', n_clusters=i , n_init=10,random_state=0,max_iter=300)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The Elbow metod")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
#-----------------------------------

kmeans = KMeans( init='k-means++', n_clusters=5 , n_init=10,random_state=0,max_iter=300)
y_means = kmeans.fit_predict(X)
       
#-----------------------------------
#visulaize the clusters
plt.scatter    


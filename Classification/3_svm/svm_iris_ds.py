# -*- coding: utf-8 -*-
#import the Libraries
#-------------------------------------------------------------------------
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

#iris = datasets.load_iris()
# Loading built-in Datasets:
iris = sns.load_dataset("iris")
iris = sns.lo
iris_data = iris.data
iris_y = iris.target
print(iris_data.head())


sns.pairplot(iris_data,hue="species", palette = "bright")

tips = sns.load_dataset("tips")
""" Importing relevant libraries """

import os 
from pathlib import Path # For fetching the required files
import pandas as pd # For data handling
import numpy as np

import matplotlib.pyplot as plt # For building plots
import seaborn as sns # For visualization

from sklearn.ensemble import RandomForestClassifier # For building RandomForest models
from sklearn.metrics import accuracy_score # For evaluating performance of predictions
from sklearn.model_selection import train_test_split # For creating train-test split

from utils.Analysis import tfIdf, wordcloudTfIdf, plotCluster, binaryClass, DiscriWordsRF

""" Loading the Clustering results """

# Load the Indexed data
DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path

ResultsPath  = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\Clusters.xlsx"
ResultsDf = pd.read_excel(ResultsPath)  

print(ResultsDf.head()) # Controlling the data loaded

""" WordClouds analysis """

# Generating wordclouds for each of the clusters based on the tf-idf "frequencies"
dfTfidfVect = tfIdf(ResultsDf)
dfTfidfVect.groupby('Cluster').apply(lambda x: wordcloudTfIdf(x, x["Cluster"].unique()))

""" Composition of the clusters """

# Building barplots showing the topic and area distribution in each cluster 
ResultsDf.groupby('Cluster').apply(lambda x: plotCluster(x, x["Cluster"].unique())) # Applying the function on each cluster

""" Identifying the most discriminant words """

for i in range(7): # Looping over the clusters and plotting the 20 most discriminant words for each

    DiscriWordsRF(dfTfidfVect, binaryClass(dfTfidfVect, i))

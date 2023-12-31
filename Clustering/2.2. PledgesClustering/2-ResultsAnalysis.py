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
from utils.Preprocessing import PreProcessing

""" Loading the Clustering results """

# Load the Indexed data
DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path

ResultsPath  = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\Clusters.xlsx"
#ResultsPath  = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\Clusters2.xlsx"
#ResultsPath  = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\Clusters3.xlsx"
#ResultsPath  = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\Clusters4.xlsx"
#ResultsPath  = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\Clusters5.xlsx"
ResultsDf = pd.read_excel(ResultsPath)  

print(ResultsDf.head()) # Controlling the data loaded

ResultsDf["PreProcessedText"] = ResultsDf["PreProcessedText"].apply(lambda x: PreProcessing(x, n=0))

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

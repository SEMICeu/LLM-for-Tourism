""" Importing relevant libraries """

import os 
from pathlib import Path # For fetching the required files
import pandas as pd # For data handling
import numpy as np

from utils.ClusterFunction import OptiCluster
from utils.Visuals import Tsne, ClusterPlot, tSNEPlot

""" Loading the Indexed Pledges and Topic """

# Load the Indexed data
DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path

IndexedPath  = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\IndexedDataV1.csv"
IndexedData = pd.read_csv(IndexedPath, index_col=0)  

# Load the Topic column from the preprocessed data
PreprocessedPath  = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\PreProcessedData.csv"  
Topics = pd.read_csv(PreprocessedPath, index_col=0)

IndexedData["Topic"] = Topics["Topic"].values # Adding a Topic column to the IndexedData dataframe

print(IndexedData.head()) # Controlling the data loaded


""" Finding optimal clusters """

# Applying the function to our data for 6 clusters
x = np.array(IndexedData.loc[:, IndexedData.columns != "Topic"]) 
yKm = OptiCluster(x, 6)

""" Exporting the results in excel """

ResultsDf = pd.DataFrame() # Dataframe to store all the results we are interested in

# Load the original pledge text column from the cleaned data
CleanedPath  = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\CleanedData.csv"  
PledgeDf = pd.read_csv(CleanedPath, index_col=0)

# Populate the ResultsDf columns with the different results
ResultsDf["Pledge"] = PledgeDf["Pledge"]
ResultsDf["PreProcessedText"] = Topics["PreProcessedText"]
ResultsDf["Cluster"] = yKm + 1 # Ensure that clusters are listed from 1 to 6
ResultsDf["Topics"] = Topics["Topic"]

# Control the results data
print(ResultsDf["Cluster"] ) 

# Adding an extra column with the area of focus
area1 = [1, 2, 3, 4, 5]
area2 = [6,7,8,12,13]
area3 = [9,10,14,15,16]
area4 = [11, 19,20,23,27]
area5 = [17,18,21,22,24,25,26]

areaList = []

for topic in ResultsDf["Topics"]:

    if topic in area1:
        areaList.append("Policy & regulation")
    elif topic in area2: 
        areaList.append("Green Transition")
    elif topic in area3: 
        areaList.append("Digital Transition")
    elif topic in area4:
        areaList.append("Stakeholder support")
    elif topic in area5:
        areaList.append("Skills & resilience")
    else:
        areaList.append("Other")

ResultsDf["Area"] = areaList

# Exporting to excel
ResultsDf.to_excel(str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\Clusters.xlsx")


""" Dimensionality reduction for visuals """

df = Tsne(IndexedData)

""" Visualizing results """
# Scatter plot of all clusters
ClusterPlot(df, yKm+1) # +1 to have clusters name going from 1 to 6

# Scatter plots cluster by cluster
df["Cluster"] = yKm + 1
df.groupby('Cluster').apply(lambda x: tSNEPlot(x, x["Cluster"].unique(), "Cluster "))
""" Importing Relevant Libraries """

import pandas as pd # For data handling
import numpy as np
import math # For basic mathematical operations

import matplotlib.pyplot as plt # For building plots
import seaborn as sns # For visualization

from sklearn.manifold import TSNE # For applying a dimensionality reduction (t-SNE)
from sklearn.cluster import KMeans # For K-mean clustering algorithm
import scipy.cluster.hierarchy as shc # For building hierachichal clustering algorithms

""" Optimal Number of clusters - Visuals """

# To find the optimal number of cluster in the data, we proceed with a visual exploration using:
# 1° The Elbow method:

## Function to compute the Total Sum of Square in the data
def TSS(data):
    """
    TSS computes the total sum of square of a given dataset
    
    :param data: pd Dataframe containing the indexed pledges
    :return: TSS (float)
    """
    x = np.array(data)
    size = x.shape[0]
    tss = []

    for i in range(0,size):

        tss.append((math.dist(x[i,:], np.mean(x, axis = 0)))**2)

    return sum(tss)  

## Function to build an elbow graph
def ElbowGraph(x):
    """
    Elbowgraph construct an elbowgraph using WSS/TSS criterion for K-means clustering for  1 < c < 26

    :param x: pd Dataframe of data to cluster (indexed pledges)
    :return: Elbowgraph
    """

    print("Elbow Graph: \n")
    inertias = []
    size = np.array(x).shape[0]

    if size > 26:
        size = 26

    for i in range(1, size):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(x)
        inertias.append(kmeans.inertia_ / TSS(x))

    plt.plot(range(1,size), inertias, marker='o')
    plt.title('WSS/TSS method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WSS/TSS')
    plt.show()

# 2° Dendrogram
def Dendro(x):
    """
    Dendro builds a dendrogram for a given set of data

    :param x: pd Dataframe of data to cluster (indexed pledges)
    :return: Dendrogram
    """

    print("Dendrogram: \n")
    plt.figure(figsize=(30, 7))
    plt.title("Topics Dendrogram")

    # Selecting Annual Income and Spending Scores by index
    SelectedData = x
    clusters = shc.linkage(SelectedData, 
                method='ward', 
                metric="euclidean")
    shc.dendrogram(Z=clusters, color_threshold=1.75)
    plt.show()

# Combining the two functions:
def GraphAnalysis(x, topic):
    """
    GraphAnalysis combines both ElbowGraph and Dendro
    
    :param x: pd Dataframe of data to cluster (indexed pledges)
    :param topic: topic of interest
    :return: An elbowgraph and a dendrogram
    """
    print("Cluster Analysis of Topic:" + str(topic[0]) + "\n")

    ElbowGraph(x)
    Dendro(x)


""" Dimensionality reduction for visuals """

def Tsne(IndexedData):
    """
    Tsne reduces the dimensionality of a pandas dataframe to 2

    :param IndexedData: Dataframe containing the indexed vectors of all pledges + 1 column containing the topics of the pledges
    :return: Dataframe containing the reduced representation of pledges + a topic column
    """

    # Reducing the dimension of the Indexed vectors from 300 to 2 using t-SNE
    results = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(IndexedData.loc[:, IndexedData.columns != "Topic"])

    # Building a new df for better visualization of t-SNE results
    df = pd.DataFrame()
    df["Topic"] = IndexedData["Topic"].values
    df["Y1"] = results[:,0]
    df["Y2"] = results[:,1]

    return df

""" Visualizing results """

# Plotting the global results using t-SNE: tsne = dataframe containing the results of a t-SNE dimensionality reduction ; labels = predicted clusters
def ClusterPlot(tsne, labels):
    """
    ClusterPlot generates scatter plots based on a tsne dataframe

    :param tsne: Dataframe of dimension 2 containing the tsne reduction of the original indexed data
    :param labels: Array containing a list of predicted clusters
    :return: A scatter plot
    """

    tsne["cluster"] = labels

    csfont = {'fontname':'Arial'} # setting font for axis labels
    hfont = {'fontname':'Georgia'} # setting font for title

    #plt.figure(figsize=(8.75,3)) # setting size of the graph
    sns.scatterplot(

        x="Y1", y="Y2", # Data to plot
        hue="cluster",  # Group by cluster
        palette=sns.color_palette(), # Apply a given color palette
        data=tsne, # Data source file

        
    ).set_title("2D t-SNE plot", fontdict = hfont) # Adding a title with the font hfont

    plt.show()


# Creating a function for building tSNE plots by cluster
def tSNEPlot(x, topic, title):
    """
    tSNEPlot generates scatter plots based on a tsne dataframe

    :param x: Dataframe of dimension 2 containing the tsne reduction of the original indexed data
    :param topic: Tuple containing the topic number at index 0
    :param title: String indicating the title of the plot (either Topic or Cluster)
    :return: A scatter plot
    """
    csfont = {'fontname':'Arial'} # Font name for labels
    hfont = {'fontname':'Georgia'} # Font name for title

    ax = plt.axes()
    ax.scatter(x["Y1"], x["Y2"], c="#d04a02", marker= "v") # Setting data to plot, color, and markers type
    ax.set_title("2D t-SNE plot of " + str(title) + str(topic[0]), **hfont) # Defining a title and its format
    ax.set_ylabel("Y2", **csfont) # Defining y-label
    ax.set_ylim(-35,35) # Standardizing the scale of y axis
    ax.set_xlabel("Y1", **csfont) # Idem for x axis
    ax.set_xlim(-35,35)

    ax.set_facecolor((218 / 255, 222 / 255, 224 / 256)) # Formatting the background's color
    plt.grid(which='major', color='w', linestyle='-') # Adding a grid to the background
    ax.set_axisbelow(True)  # Formatting the axis style
    for spine in ax.spines:
        ax.spines[spine].set_color('white')

    plt.show()

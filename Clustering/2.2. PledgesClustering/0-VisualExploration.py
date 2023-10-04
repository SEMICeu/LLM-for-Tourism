""" Importing Relevant Packages """

import os # For finding pre-processed data
from pathlib import Path

import pandas as pd # For data handling
import numpy as np

import matplotlib.pyplot as plt # For building plots
import seaborn as sns # For visualization

from sklearn.manifold import TSNE # For applying a dimensionality reduction (t-SNE)

from utils.Visuals import GraphAnalysis

""" Loading the Indexed Pledges and Topic """

# Load the Indexed data
DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path - Specific for ipynb file - For .py: Path(os.path.dirname(os.path.realpath(__file__)).replace("\\", "/"))
IndexedPath  = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\IndexedDataV1.csv"  

print("The current location of IndexedDataV1.csv is: ", IndexedPath)

IndexedData = pd.read_csv(IndexedPath, index_col=0)  # Loading the indexed pledges into a dataframe

print(IndexedData.head()) # Controlling the data loaded


# Load the Topic column from the preprocessed data
DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path - Specific for ipynb file - For .py: Path(os.path.dirname(os.path.realpath(__file__)).replace("\\", "/"))
PreprocessedPath  = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\PreProcessedData.csv"  
Topics = pd.read_csv(PreprocessedPath, index_col=0)

IndexedData["Topic"] = Topics["Topic"].values # Adding a Topic column to the IndexedData dataframe


""" Dimensionality reduction """

# Reducing the dimension of the Indexed vectors from 300 to 2 using t-SNE
results = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(IndexedData.loc[:, IndexedData.columns != "Topic"])

# Building a new df for better visualization of t-SNE results
df = pd.DataFrame()
df["Topic"] = IndexedData["Topic"].values
df["Y1"] = results[:,0]
df["Y2"] = results[:,1]

""" Global Visualization """

# Building a scatter plot for the t-SNE results
csfont = {'fontname':'Arial'} # setting font for axis labels
hfont = {'fontname':'Georgia'} # setting font for title

plt.figure(figsize=(8.75,3)) # setting size of the graph
sns.scatterplot(

    x="Y1", y="Y2", # Data to plot
    #hue="Topic",  # Group by Topic
    #palette=sns.color_palette(), # Apply a given color palette
    #style = "Topic", # Ensure that each topic has a different style
    data=df, # Data source file
    legend="full", # Adding a legend
    
).set_title("2D t-SNE plot", fontdict = hfont) # Adding a title with the font hfont

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0, ncol = 2) # Placing the legend on the graph
plt.show()


""" Topic by Topic visualization """

# Creating a function for building tSNE plots
def tSNEPlot(x, topic):
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
    ax.set_title("Topic " + str(topic[0]) + ": " + str(mapping[topic[0]]), **hfont) # Defining a title and its format
    ax.set_ylabel("Y2", **csfont) # Defining y-label
    ax.set_ylim(-30,30) # Standardizing the scale of y axis
    ax.set_xlabel("Y1", **csfont) # Idem for x axis
    ax.set_xlim(-30,30)

    ax.set_facecolor((218 / 255, 222 / 255, 224 / 256)) # Formatting the background's color
    plt.grid(which='major', color='w', linestyle='-') # Adding a grid to the background
    ax.set_axisbelow(True)  # Formatting the axis style
    for spine in ax.spines:
        ax.spines[spine].set_color('white')

    plt.show()

# Applying the function on all Topic
mapping = {1: "Short-term rentals", 2: "Multimodal travelling", 3: "Expanding tourism indicators", 4: "Comprehensive tourism strategies", 5: "Collaborative and smart destination governance", 6: "Sunstainable mobility", 7: "Circular tourism services", 8: "Companies reducing environmental impacts", 9: "Data-driven tourism services", 10: "Clear online information offer", 11: "Networking, Best practice sharing", 12: "R&I projects and pilots on sustainable tourism", 13: "Experimenting environmental footprint methods for tourism", 14: "Interoperable data space for tourism", 15: "R&I for digital tools and services", 16: "Digitalisation of SMEs and destinations", 17: "Facilitating travelling (cross-border, coodinated rules sharing)", 18:"Facilitating travelling (cross-border, coodinated rules sharing)", 19: "Awareness raising (skills need, transition benefits)", 20:  "Awareness raising (skills need, transition benefits)", 21: "Skills and education development", 22: "Skills and education development", 23: "One-stop-shop to resources (skills, funding)", 27: "One-stop-shop to resources (skills, funding)", 24: "Fair and good quality jobs", 25: "Accessible tourism services", 26: "Diversification of tourism services, including resident perspective", 28: "Other"}

df.groupby('Topic').apply(lambda x: tSNEPlot(x, x["Topic"].unique()))


""" Optimal Number of clusters - Global """

# To find the optimal number of cluster in the data, we proceed with a visual exploration using:
# 1° The Elbow method:
# 2° Dendrogram

GraphAnalysis(IndexedData.loc[:, IndexedData.columns != "Topic"], "Global")




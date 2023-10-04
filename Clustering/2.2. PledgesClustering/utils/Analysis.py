""" Importing relevant libraries """

import os 
from pathlib import Path # For fetching the required files
import pandas as pd # For data handling
import numpy as np

import matplotlib.pyplot as plt # For building plots
import seaborn as sns # For visualization
from wordcloud import WordCloud # For generating wordclouds

from sklearn.feature_extraction.text import TfidfVectorizer # For tf-idf representation

from sklearn.ensemble import RandomForestClassifier # For building RandomForest models
from sklearn.metrics import accuracy_score # For evaluating performance of predictions
from sklearn.model_selection import train_test_split # For creating train-test split

""" WordClouds analysis """

# Generating wordclouds for each of the clusters based on the tf-idf "frequencies"

def tfIdf(ResultsDf):
    """
    tfIdf returns the tf-idf matrix for a provided corpus of text

    :param ResultsDf: pd Dataframe containing a column "PreProcessedText" where each row is a string
    :return: pd Dataframe containing the tf-idf matrix
    """
    # For tf-idf representation
    tfidfvectorizer = TfidfVectorizer(analyzer='word') # Initializing the tf-idf vectorizer
    tfidfWm = tfidfvectorizer.fit_transform(ResultsDf["PreProcessedText"]) # Generating tf-idf word matrix on pre-processed text
    tfidfTokens = tfidfvectorizer.get_feature_names_out() # List of words present in the pledges

    dfTfidfVect = pd.DataFrame(data = tfidfWm.toarray(),columns = tfidfTokens) # Matrix tf-idf

    dfTfidfVect["Cluster"] = ResultsDf["Cluster"] # Adding column containing the predicted cluster for each pledge

    print("\nTD-IDF Vectorizer\n")
    print(dfTfidfVect)

    return dfTfidfVect


# Building word clouds by cluster: data = tf-idf matrix of the cluster ; cluster = name of the cluster
def wordcloudTfIdf(data, cluster):
    """
    wordcloudTfIdf builds word clouds based on the tf-idf matrix of pledges from a given cluster

    :param data: pd Dataframe containing the tf-idf matrix of the cluster
    :param cluster: int indicating the number of the cluster
    :return: Wordcloud
    """

    data = data.loc[:, data.columns != "Cluster"]
    data = data.T.mean(axis = 1)

    print("Cluster" + str(cluster))
    cloud=WordCloud(colormap="ocean",width=600,height=400, background_color="white", max_words=20).generate_from_frequencies(data) # Setting color for the map, background + defining dimensions
    fig=plt.figure(figsize=(13,18)) # Size of the figure
    plt.axis("off") # Removing the axis
    plt.imshow(cloud,interpolation='bilinear')
    plt.show()

""" Composition of the clusters """
# Color palette for the different areas
SelfPalette = {"Policy & regulation": "blue", "Green Transition": "green", "Digital Transition": "orange", "Stakeholder support": "red", "Skills & resilience": "purple", "Other": "brown"}

# Function to plot topic distribution across clusters:
def plotCluster(x, c):
    """
    plotCluster plots the topic distribution across clusters

    :param x: dataframe containing the topics and areas for each pledge
    :param c: name of the cluster to plot
    return: Catplot
    """ 

    sns.set_style("dark") # Building a catplot of the number of pledges per topic
    graph = sns.catplot(data=x, x="Topics", kind="count", hue ="Area", height = 3, aspect = 2, dodge = False, palette= SelfPalette)
    plt.title("Cluster {}".format(c[0]))
    plt.show()

""" Identifying the most discriminant words """

# 1° Create 1 binary vector for each cluster:
def binaryClass(x, group):
    """
    binaryClass create a binary vector for each cluster

    :param x: tf-idf dataframe
    :param group: cluster of interest
    :return: Binary vector
    """ 
    binaryVec = []

    for cluster in x["Cluster"]:

        if cluster == group:
            binaryVec.append(1)
        else:
            binaryVec.append(0)

    return binaryVec

# 2° Build a randomforest classification model to extract the most discriminant words: data = tf-idf matrix ; cluster = binary vector for cluster of interest
def DiscriWordsRF(data, cluster):
    """
    DiscriWordsRF builds a randomforest classifier to extract the most discriminant words and plots the results

    :param data: pd Dataframe containing a tf-idf matrix
    :param cluster: binary vector for the cluster of interes
    :return: Barplot of the 20 most discriminant words
    """

    data = data.copy()
    data["Cluster"] = cluster

    # Building training and test sets
    X, xTest, Y, yTest = train_test_split(data.loc[:, data.columns != "Cluster"], data["Cluster"], test_size=0.2)
    
    rf = RandomForestClassifier(random_state=0) # Initializing RF model on training set
    rf.fit(X, Y)

    yPred = rf.predict(xTest) # Prediction on the test set
    accuracy = accuracy_score(yTest, yPred) # Evaluating accuracy
    print("Accuracy:", accuracy)

    FeatureImportances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Plot a simple bar chart of the 20 most discriminant words
    FeatureImportances[:20].plot.bar()
    plt.show()
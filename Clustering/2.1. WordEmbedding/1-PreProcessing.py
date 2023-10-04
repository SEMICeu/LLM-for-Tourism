""" Importing Relevant Libraries """

import os # For finding the location of Cleaned Data file
from pathlib import Path

import pandas as pd # For data handling
import matplotlib.pyplot as plt # For plotting data
from wordcloud import WordCloud # For plotting wordclouds

from utils.PreprocessingFunction import PreProcessing


""" Loading the cleaned data file: CleanedData.csv """

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path
PledgesCsvPath = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\CleanedData.csv" 

print("The current location of CleanedData.csv is: ", PledgesCsvPath)

PledgesDf = pd.read_csv(PledgesCsvPath, index_col=0) # Reading the csv file using a DataFrame

""" PreProcessing the Pledges """
n=0
print("begin pre-processing")
PledgesDf['PreProcessedText'] = PledgesDf['Pledge'].apply(lambda x: PreProcessing(x, n))
print("end pre-processing\n")

print(PledgesDf.head())

""" Adding extra stopwords (most frequent words)"""
# No coherent results at the moment => no reason to keep it
# Uncomment if you want to consider the most frequent words from the pledges as stopwords

# StopWords2 = pd.Series(" ".join(PledgesDf['PreProcessedText']).split()).value_counts()[:30].index.tolist()
# print(StopWords2)

# def RemoveFrequentWords(string, FrequentWords):
#     a = [i for i in string.split() if i not in FrequentWords] # Removing usual english stopwords from the string
#     return ' '.join(a) #Output - Same string after all the transformations

# PledgesDf['PreProcessedText'] = PledgesDf['PreProcessedText'].apply(lambda x: RemoveFrequentWords(x, StopWords2))

# print(PledgesDf.head())


""" Outputing the pre-process data """

PreProcessedDataPath = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\PreProcessedData.csv"
PledgesDf.to_csv(PreProcessedDataPath)


""" Visualisation """

# Building a wordcloud for visualizing most frequent words
cloud=WordCloud(colormap="ocean_r",width=600,height=400, background_color="#e5e5e5").generate(PledgesDf['PreProcessedText'].str.cat(sep=' ')) # Setting color for the map, background + defining dimensions
fig=plt.figure(figsize=(13,18)) # Size of the figure
plt.axis("off") # Removing the axis
plt.imshow(cloud,interpolation='bilinear')
plt.show() 

# Most frequent words in the pledges after pre-processing
plt.style.use('ggplot')  # Setting up the style
plt.figure(figsize=(14,6)) # Setting up the size of the figure
freq=pd.Series(" ".join(PledgesDf['PreProcessedText']).split()).value_counts()[:30] # Defining the series to plot, i.e., the 30 most frequent words
freq.plot(kind="bar", color = "orangered") # Choosing the plot type (bar) and colore (orangered)
plt.title("30 most frequent words",size=20) # Title
plt.show()





""" Library to import """

import os # For finding source files
from pathlib import Path

import pandas as pd # For data handling
import numpy as np
import string # For handling of textual data

import matplotlib.pyplot as plt # For data visualization
import seaborn as sns # For statistical and data visualization


""" Uploading the cleaned file: CleanedData.csv """

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path's parent
CSVFilePath = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\CleanedData.csv"  

print("Current Location of the Data file is :", CSVFilePath)

PledgeDf = pd.read_csv(CSVFilePath, index_col=0) # Creating a dataframe from the content of the csv file # Args: index_col = 0, first column of the csv contains the indexes

# Inspecting the first rows of the dataframe
print(PledgeDf.head())


""" Plotting some insights about the different topics and pledges """

sns.set_style("dark") # Building a catplot of the number of pledges per topic
graph = sns.catplot(data=PledgeDf, x="Topic", kind="count", height = 3, aspect = 2.9, palette = "ocean")
#plt.title("Frequency showing number of pledges assigned in different topics", size = 20, y = 0.8)
plt.axhline(y=np.mean(np.array(PledgeDf.groupby("Topic").size())), color='blue', ls='--', lw=2.5) # Adding a mean line
plt.draw()

# Defining a new column containing the number of words per pledge
LengthList = [] # List that will contain the number of words of each pledge
for pledge in PledgeDf["Pledge"]: # Looping over the different pledges from the dataframe
    LengthList.append(len(pledge.split())) # Appending the number of words to the pledge
PledgeDf["Length"] = LengthList


plt.figure(figsize=(8,6))
sns.set_style("dark") 
graph = sns.boxplot(data=PledgeDf["Length"], palette = "ocean") # Plotting the dispersion of word count across the different pledges (looking for outliers)
plt.title("Boxplot of Pledges' Length", size = 20)
plt.draw()

plt.figure(figsize=(8,6))
sns.boxplot(x = PledgeDf["Topic"], y = PledgeDf["Length"]) # Boxplots of length by topic
plt.title("Boxplot of Pledges' Length by Topic", size = 20)
plt.draw()

plt.show()



""" Importing relevant libraries """
import os 
from pathlib import Path # For fetching the required files
import pandas as pd # For data handling

import matplotlib.pyplot as plt # For building plots
import seaborn as sns # For visualization

from datetime import datetime # For dealing with Date format

# Loading the Pledges with their Topic and Clusters

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path's parent
XlsxFilePath = str(DirPpath.absolute()) + r"\semic_pledges\OutputFiles\NerResults.xlsx"  

print("Current Location of the Data file is :", XlsxFilePath)

DatesResults = pd.read_excel(XlsxFilePath) # Creating a dataframe from the content of an excel file 

# Inspecting the first rows of the dataframe
DatesResults.head()

# Defining the initial size of the dataset
CsvFilePath = str(DirPpath.absolute()) + r"\semic_pledges\OutputFiles\CleanedData.csv"
InitialData = pd.read_csv(CsvFilePath, index_col=0)  

size = len(InitialData["Topic"])
print(size)


""" First Analysis """

# Number of pledges containing a date
firstAnalysis = DatesResults
n = firstAnalysis["Pledge"].value_counts().size

print(n)

pledgeplot = pd.Series([1]).repeat(n).append(pd.Series([0]).repeat(size-n))
pledgeplot.value_counts().plot(kind="bar")
plt.show()


# Number of pledges containing a result
firstAnalysis = DatesResults[DatesResults["Results?"] == 1]
n = firstAnalysis["Pledge"].value_counts().size

print(n)

pledgeplot = pd.Series([1]).repeat(n).append(pd.Series([0]).repeat(size-n))
pledgeplot.value_counts().plot(kind="bar")
plt.show()

# Number of sentences containing results

data = DatesResults.copy()
data["Results?"].value_counts().plot(kind="bar")
plt.show()
print(data["Results?"].value_counts())


""" Timeline of Results """

# Timeline for a specific time frame, only results sentences
from utils.GraphAnalysis import GlobalTimeline
intervals = [[2022, 2027], [2028, 2050]]

for interval in intervals:
    GlobalTimeline(DatesResults, interval[0], interval[1])

# Timeline for each cluster
from utils.GraphAnalysis import TimelineBy

dateMin = datetime.strptime("January 2022", '%B %Y')
dateMax = datetime.strptime("December 2031", '%B %Y')

data = DatesResults[(DatesResults["Dates"] >= dateMin) & (DatesResults["Dates"] <= dateMax) & (DatesResults["Results?"] == 1)]

data.groupby('Topics').apply(lambda x: TimelineBy(x, x["Topics"].unique(), dateMin, dateMax, "Topic"))
data.groupby('Clusters').apply(lambda x: TimelineBy(x, x["Clusters"].unique(), dateMin, dateMax, "Cluster"))

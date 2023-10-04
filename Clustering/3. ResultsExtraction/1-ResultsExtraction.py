""" Importing relevant libraries """

import os 
from pathlib import Path # For fetching the required files
import pandas as pd # For data handling

import re # For dealing with regular expressions

from utils.DatesExtraction import DatesExtraction
from utils.ResultsExtraction import GetResults

""" Main function """

# Loading the Pledges with their Topic and Clusters

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path's parent
XlsxFilePath = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\Clusters.xlsx"  

print("Current Location of the Data file is :", XlsxFilePath)

PledgeDf = pd.read_excel(XlsxFilePath) # Creating a dataframe from the content of an excel file 

# Inspecting the first rows of the dataframe
PledgeDf.head()


# Preprocessing Data

def FirstClean(text):
    return " ".join(text.split())

def RemoveURL(text):
    """Remove URLs from a sample string"""
    text = re.sub(r"https:\s?\S+", "", text)
    text = re.sub(r"http\S+", "", text) 
    return re.sub(r'[^\x00-\x7f]',r'',  text) # Remove non-ASCII

PledgeDf["PreProcessedText"] = PledgeDf["Pledge"].apply(lambda x: RemoveURL(FirstClean(x)))

# Extracting Results

#1° Extracting Dates = pandas df containing dated sentences, dates, original pledge of the sentence, topic of the pledge, cluster of the pledge 
DatesResults = DatesExtraction(PledgeDf)
#2° Extracting Dated Results = Adding a column indicating for the presence of results in the dated sentence + 6 columns indicating for the categories of results presented in the sentence (based on keywords)
output = GetResults(DatesResults["Sentences"])
names = ["Results?", "Reports?", "Events?", "Trainings?", "Project results", "Best practices?", "Awards?"]
i = 0

for name in names:

    DatesResults[name] =  output[i]
    i +=1

print(DatesResults.head())

# Outputting results excel
XlsxFilePath = str(DirPpath.absolute()) + r"\semic_pledges\OutputFiles\NerResults.xlsx"
DatesResults.to_excel(XlsxFilePath)
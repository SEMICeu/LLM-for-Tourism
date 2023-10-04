""" Library to import """

import os # For finding source files
from pathlib import Path

import pandas as pd # For data handling
import string # For handling of textual data

import matplotlib.pyplot as plt # For data visualization
import seaborn as sns # For statistical and data visualization

from langdetect import detect # Language detection models


""" Uploading the source file: PledgeList.xlsx """

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path's parent
PledgeFilePath = str(DirPpath.absolute()) + "\semic_pledges\InputFile\PledgeList.xlsx"  

print("Current Location of the Source file is :", PledgeFilePath)

PledgeSheet = "Pledges text" # Define the excel sheet where the pledges are to be found
PledgeDf = pd.read_excel(PledgeFilePath, sheet_name = PledgeSheet) # Creating a dataframe from the content of the excel file

# Inspecting the first rows of the dataframe
print(PledgeDf.head())


""" Data Cleansing """

PledgeDf = PledgeDf.fillna("") # Dealing with the potential presence of nan in the data

ToDrop =  ["Organisation name", "Country", "Type", "Pledge status"] # The 4 last Variables are not used for the purpose of this project
PledgeDf.drop(ToDrop, inplace = True, axis = 1) # Using the .drop function to get rid of unnecessary columns (as listed in ToDrop) # Args: inplace = True to apply the changes directly into the dataframe ; axis = 1 to drop the values in the columns of df

print("\n")
print(PledgeDf.head()) #Inspecting the first rows of the modified df

if not all(PledgeDf['Pledge'].map(type) == str):  # Controlling the presence of Data types errors in the two remaining columbs
    print("\nData type issue in Pledges")
elif not all(PledgeDf['Topic'].map(type) == int):
    print("\nData Type issue in Topic")
else:
    print("\nNo apparent Data Type issues")


""" Splitting by languages (Fr and En) """

liste = [] # List to contain the rows of French pledges

for i in PledgeDf.index:
    text = PledgeDf.iloc[i,1] # Looping over all Pledges row by row

    #print(text)
    #print(i)

    if detect(text) == "fr": # If the majority of the text is french
        print(i)
        liste.append(i) # Then add the row index to the list

liste.append(50) # Exception to handle (too small pledge to detect the french)

PledgeFr = PledgeDf[PledgeDf.index.isin(liste)]
PledgeEn = PledgeDf[~PledgeDf.index.isin(liste)]


""" Exporting Cleaned Data """

CSVFilePath = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\CleanedData.csv" 
PledgeEn.to_csv(CSVFilePath)

CSVFilePathFr = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\CleanedDataFr.csv" 
PledgeFr.to_csv(CSVFilePathFr)
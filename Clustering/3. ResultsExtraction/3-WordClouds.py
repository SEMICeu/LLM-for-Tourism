""" Importing relevant libraries """
import os 
from pathlib import Path # For fetching the required files
import pandas as pd # For data handling

import matplotlib.pyplot as plt # For building plots
import seaborn as sns # For visualization
from wordcloud import WordCloud # For generating wordclouds
from nltk.corpus import stopwords

from datetime import datetime # For dealing with Date format

# Loading the Pledges with their Topic and Clusters

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path's parent
XlsxFilePath = str(DirPpath.absolute()) + r"\semic_pledges\OutputFiles\NerResults.xlsx"  

print("Current Location of the Data file is :", XlsxFilePath)

DatesResults = pd.read_excel(XlsxFilePath) # Creating a dataframe from the content of an excel file 

# Inspecting the first rows of the dataframe
print(DatesResults.head())

""" Building wordclouds for year 2023, 2025, and 2030 """

# Selecting subsets of Datesresults for 2023, 2025, and 2030
Data2023 = DatesResults[(DatesResults["Dates"] >= datetime.strptime("January 2023", '%B %Y')) & (DatesResults["Dates"] < datetime.strptime("January 2024", '%B %Y')) & (DatesResults["Results?"] == 1)]
Data2025 = DatesResults[(DatesResults["Dates"] >= datetime.strptime("January 2025", '%B %Y')) & (DatesResults["Dates"] < datetime.strptime("January 2026", '%B %Y')) & (DatesResults["Results?"] == 1)]
Data2030 = DatesResults[(DatesResults["Dates"] >= datetime.strptime("January 2030", '%B %Y')) & (DatesResults["Dates"] < datetime.strptime("January 2031", '%B %Y')) & (DatesResults["Results?"] == 1)]

# Build three wordclouds
for data in [Data2023, Data2025, Data2030]:

    print(len(data["Sentences"]))

    cloud=WordCloud(colormap="ocean_r",width=600,height=400, background_color="white", stopwords=stopwords.words('english'), max_words=20).generate(data["Sentences"].str.cat(sep=' '))
    fig=plt.figure(figsize=(13,18))
    plt.axis("off")
    plt.imshow(cloud,interpolation='bilinear')
    plt.show()

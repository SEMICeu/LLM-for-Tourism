""" Importing relevant packages """
import os # For finding pre-processed data
from pathlib import Path

import pandas as pd # For data handling
from sklearn.metrics import silhouette_score # intrinsic validation metric


results = {}
models = {"BERT": "", "TourBERT": "2", "FBERT": "3", "RoBERTa": "4", "FRoBERTa": "5", "Word2Vec": "6"}

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path

for model in models.keys():
    
    """ Loading the pre-processed data """

    if model == "Word2Vec":
        PledgesCsvPath = str(DirPpath.absolute()) + "\\semic_pledges\\OutputFiles\\Clusters.xlsx"
        IndexedCsvPath = str(DirPpath.absolute()) + "\\semic_pledges\\OutputFiles\\IndexedDataV1.csv"
    elif model == "BERT":
        PledgesCsvPath = str(DirPpath.absolute()) + f"\LLM-for-Tourism\Clustering\OutputFiles\Clusters{models[model]}.xlsx" 
        IndexedCsvPath = str(DirPpath.absolute()) + f"\LLM-for-Tourism\Clustering\OutputFiles\IndexedDataV1.csv"
    else: 
        PledgesCsvPath = str(DirPpath.absolute()) + f"\LLM-for-Tourism\Clustering\OutputFiles\Clusters{models[model]}.xlsx" 
        IndexedCsvPath = str(DirPpath.absolute()) + f"\LLM-for-Tourism\Clustering\OutputFiles\IndexedDataV{models[model]}.csv"

    print("The current location of PreprocessedData.csv is: ", PledgesCsvPath)
    PledgesDf = pd.read_excel(PledgesCsvPath, index_col=0) # Loading the preprocessed pledges into a dataframe

    print("The current location of IndexedData.csv is: ", IndexedCsvPath)
    data = pd.read_csv(IndexedCsvPath, index_col=0)

    data["Prediction"] = PledgesDf["Cluster"].values

    """ Computing the silouhette score """
    results[model] = silhouette_score(data.loc[:, data.columns != 'Prediction'], data["Prediction"])
    print(results[model])
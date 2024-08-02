""" Importing relevant packages """
import os # Environment variable
from pathlib import Path # For fetching file path
from utils.Summarisation import GPTSummarisation
from utils.Predictions import GPTEval
import pandas as pd


models = {"BERT": "", "TourBERT": "2", "FBERT": "3", "RoBERTa": "4", "FRoBERTa": "5"}

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path

APIType = "azure"
APIBase = "XXX"
APIVersion = "2023-06-01-preview"
os.environ["OPENAI_API_KEY"] = "XXX"

resultsDict = {}


for model in models.keys():

    ResultsPath  = str(DirPpath.absolute()) + f"\LLM-for-Tourism\Clustering\OutputFiles\Clusters{models[model]}.xlsx"
    summary = GPTSummarisation(ResultsPath, model)

    summary.CreateClusterText()
    summary.BuildingPrompts()
    summary.ClusterSummaries(str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\\3. Evaluation\Summaries.csv")

    Eval = GPTEval(summary.Summaries, summary.Df)
    Eval.GPTPredictions(APIType, APIBase, APIVersion)
    Eval.Evaluation()

    resultsDict[model] = Eval.prediction

    #exec(f"{model} = {Eval}")

    print(f"Accuracy: {Eval.Accuracy} , F1-Score: {Eval.F1}")


predictionsDf = pd.DataFrame.from_dict(resultsDict)
predictionsDf.to_excel("Results.xlsx")


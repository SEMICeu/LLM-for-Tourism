""" Importing relevant packages """
import os # Environment variable
from pathlib import Path # For fetching file path
from utils.Summarisation import GPTSummarisation
from utils.Predictions import GPTEval


models = {"BERT": "", "TourBERT": "2", "FBERT": "3", "RoBERTa": "4", "FRoBERTa": "5"}

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path

APIType = "azure"
APIBase = "https://eastusazuredemo.openai.azure.com/"
APIVersion = "2023-07-01-preview"
os.environ["OPENAI"] = "XXXX"


for model in models.keys():

    ResultsPath  = str(DirPpath.absolute()) + f"\OutputFiles\Clusters{models[model]}.xlsx"
    summary = GPTSummarisation(ResultsPath, model)

    summary.CreateClusterText()
    summary.BuildingPrompts()
    summary.ClusterSummaries("Summaries.csv")

    Eval = GPTEval(summary.Summaries, summary.Df)
    Eval.GPTPredictions(APIType, APIBase, APIVersion)
    Eval.Evaluation()

    exec(f"{model} = {Eval}")

    print(f"Accuracy: {Eval.Accuracy} , F1-Score: {Eval.F1}")


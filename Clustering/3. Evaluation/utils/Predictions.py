""" Importing relevant packages """
import openai # For querying GPT
import re # For cleaning strings
import time # For managing token rate limits
from sklearn.metrics import f1_score, accuracy_score # For evaluating the model
import os # For accessing environment variables
import pandas as pd # For handling data

class GPTEval:
    """
    Class for classifying pledges into predefined clusters using GPT. This can also be used to evaluate the quality of clusters
    """
    def __init__(self, summaries, pledges):
        """
        __init__ creates the GPTEval object with two object variables based on the input of the user

        :list summaries: List of the summaries of clusters (lenght = 6)
        :DataFrame pledges: Pandas dataframe containing at least a column with the pledges' texts and a column with the clusters
        """
        self = self
        self.summaries = summaries
        self.pledges = pledges
        

    def GPTPredictions(self, APIType, APIBase, APIVersion):
        """
        GPTPredictions creates a list with the GPT predictions for each pledge and stores it as an object variable

        :string APIType: Corresponds to the openai.api_type variable
        :string APIBase: Corresponds to the openai.api_base variable
        :string APIVersion: Corresponds to the openai.api_version variable
        """        
        # Initialise all necessary variables to make API calls to openai
        openai.api_type = APIType
        openai.api_base = APIBase
        openai.api_version = APIVersion
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self.prediction = [] # Create the object variable to store predictions
        i = 0

        while len(self.prediction) < len(self.pledges["Pledge"]): # Handling the token rate limit issue
        
            texts = self.pledges.iloc[i, :]["Pledge"] # Loop over the pledges
            
            # Prompting GPT for classification
            try: 
                MessageText = [{"role":"system","content":f"You are an AI assistant that helps people find information.You will be provided with different textual pledges. Classify each pledge into Cluster 1, Cluster 2, Cluster 3, Cluster 4, Cluster 5, or Cluster 6. {self.summaries[0]} {self.summaries[1]} {self.summaries[2]} {self.summaries[3]} {self.summaries[4]} {self.summaries[5]}"}
                                ,{"role":"user","content":f"Here is the pledge to classify: {texts}. Only provide the Cluster number as output"}]

                completion = openai.ChatCompletion.create(
                    engine="basetest",
                    messages = MessageText,
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )

                self.prediction.append(re.sub(r'Cluster\s([0-9])', r'\1' ,completion.choices[0]["message"]["content"]))
                i+=1

            except: # Handling the token rate limit error
                time.sleep(30)

        return "Predictions obtained"

    def Evaluation(self):
        """
        Evaluation computes the accuracy and f1_score for the current model.

        Both variables are stored as object variables
        """        
            
        for i in range(len(self.prediction)): # Loop over the predictions
    
            try: # Clean the string to only keep the cluster number
                self.prediction[i] = re.sub(r'[^\d]+', '', self.prediction[i])
                self.prediction[i] = int(self.prediction[i])
            except:
                next

        df = pd.DataFrame.from_dict({"Prediction": self.prediction, "Cluster": self.pledges["Cluster"]})

        # Compute the chosen metrics
        self.F1 = f1_score(df["Prediction"], df["Cluster"], average="weighted")
        self.Accuracy = accuracy_score(df["Prediction"], df["Cluster"])
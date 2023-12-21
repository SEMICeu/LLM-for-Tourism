""" Importing relevant packages """
import pandas as pd # For data handling

class GPTSummarisation:
    """
    Class for handling the cluster summarisation process. Allows to perform all the necessary steps to go from
    a pandas DF with one pledge per row to a dictionary containing the summaries of each cluster

    The intermediate results of the process are stored as variables of the object
    """
    def __init__(self, file, model):
        """
        __init__ creates the GPTSummarisation object. The function also initialise three variables of the object.

        :string file: Path to an excel file containing pledges and their respective clusters
        :string model: Name of the embedding model 
        """
        self = self
        self.file = file
        self.model = model
        self.Df = pd.read_excel(self.file)

    def CreateClusterText(self):
        """
        CreateClusterText creates a dictionary of list. Each list contains strings with the texts of pledges from a common cluster. 
        (strings are splitted to fit within the limit of a GPT prompt).    
        
        The dictionary is stored as a object variable: ClusterTxt   
        """
        self.ClusterTxt = {} # Create the object variable

        for i in range(1,7): # Loop over the 6 clusters
            
            cluster = self.Df[self.Df["Cluster"] == i] # Create a subset of the pd dataframe containing only pledges from the current cluster
            text = ""
            
            j = 0
            self.ClusterTxt[i] = []

            for pledge in cluster["Pledge"]: # Loop over the pledges
                
                j +=1 

                # Add pledges to the string as long as the limit of tokens is not reached
                new_text = text + " Text " + str(j) + ": " + pledge + "\n"

                if len(new_text) > 3900*4:
                
                    self.ClusterTxt[i].append(text) # Add the text to the list
                    text = " Text " + str(j) + ": " + pledge + "\n"

                else:
                    text = new_text
                
                
                
            self.ClusterTxt[i].append(text)

        return "Text ready"
    
    def BuildingPrompts(self):
        """
        BuildingPrompts builds a dictionary with a list of prompts for obtaining the summary of each cluster (one list per cluster). 
        
        The dictionary is stored as an object variable: Prompt
        """
        
        self.Prompt = {} # Create the object variable
        
        for keys in self.ClusterTxt.keys(): # Loop over the clusters

            self.Prompt[keys] = [] # For each cluster create a list to store the different prompts

            if len(self.ClusterTxt[keys]) == 1: # If the cluster text is sufficiently small (only one item in the list) use the following prompt
                   self.Prompt[keys].append(f"""Below you can find a set of texts belonging to a common cluster. Based on their content, I want you to tell me what is the topic of this cluster:                        
                                            {self.ClusterTxt[keys][0]}
                                            """)
            
            else: 
                for i in range(len(self.ClusterTxt[keys])): # Loop over the different strings inside that cluster to split the prompt in subpart

                                
                    if i == 0: # For the first prompt
                        self.Prompt[keys].append(f"""Do not answer yet. This is just another part of the text I want to send you. Just receive and acknowledge as "Part 1/{len(self.ClusterTxt[keys])} received" and wait for the next part.             
                        [START PART 1/{len(self.ClusterTxt[keys])}]            
                        Below you can find a set of texts belonging to a common cluster. Based on their content, I want you to tell me what is the topic of this cluster:            
                        {self.ClusterTxt[keys][i]}
                        [END PART 1/{len(self.ClusterTxt[keys])}]  
                        Remember not answering yet. Just acknowledge you received this part with the message "Part 1/{len(self.ClusterTxt[keys])} received" and wait for the next part.
                        """)

                    elif i == (len(self.ClusterTxt[keys])-1): # For the last prompt
                        self.Prompt[keys].append(f"""[START PART {len(self.ClusterTxt[keys])}/{len(self.ClusterTxt[keys])}]    
                        {self.ClusterTxt[keys][i]}
                        [END PART {len(self.ClusterTxt[keys])}/{len(self.ClusterTxt[keys])}]
                        ALL PARTS SENT. Now you can continue processing the request.
                        """)

                    else: # For the rest
                        self.Prompt[keys].append(f"""Do not answer yet. This is just another part of the text I want to send you. Just receive and acknowledge as "Part {i+1}/{len(self.ClusterTxt[keys])} received" and wait for the next part.  
                        
                        [START PART {i+1}/{len(self.ClusterTxt[keys])}]
                        {self.ClusterTxt[keys][i]}
                        [END PART {i+1}/{len(self.ClusterTxt[keys])}]
                        Remember not answering yet. Just acknowledge you received this part with the message "Part {i+1}/{len(self.ClusterTxt[keys])} received" and wait for the next part.
                        """)

                # Add the instruction as an initial prompt
                content = f"""The total length of the content that I want to send you is too large to send in only one piece. For sending you that content, I will follow this rule:              
                [START PART 1/{len(self.ClusterTxt[keys])}]  
                this is the content of the part 1 out of {len(self.ClusterTxt[keys])} in total  
                [END PART 1/{len(self.ClusterTxt[keys])}]  
                Then you just answer: "Received part 1/{len(self.ClusterTxt[keys])}"  
                And when I tell you "ALL PARTS SENT", then you can continue processing the data and answering my requests.

                """

                self.Prompt[keys].insert(0, content)

        return "Prompts ready"
        
    def ClusterSummaries(self, file):
        """
        ClusterSummaries retrieves the cluster summaries stored in a csv file and stores it as a list in an object variable.    
        
        :string file: Path to the summary csv 
        """
        SummaryDf = pd.read_csv(file, index_col=0)
        self.Summaries = SummaryDf.to_dict()[self.model]

        return "Cluster summaries ready"
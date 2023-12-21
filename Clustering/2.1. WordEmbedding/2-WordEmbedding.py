""" Importing relevant packages """
import os # For finding pre-processed data
from pathlib import Path

import pandas as pd # For data handling
import numpy as np

import nltk #  For nlp processing
from sklearn.feature_extraction.text import TfidfVectorizer # For obtaining Tf-Idf tokenization

from utils.DocumentEmbedding import pledgeEmbedding
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, BertTokenizerFast, BertConfig, RobertaTokenizer, RobertaModel, RobertaTokenizerFast, RobertaConfig

""" Loading the pre-processed data """

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path
PledgesCsvPath = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\PreprocessedData.csv" 

print("The current location of PreprocessedData.csv is: ", PledgesCsvPath)

PledgesDf = pd.read_csv(PledgesCsvPath, index_col=0) # Loading the preprocessed pledges into a dataframe

print(PledgesDf.head()) # Controlling the data loaded


""" Tokenize the pledges on words """

documents = [i for i in PledgesDf["PreProcessedText"]]
length = max([len(nltk.word_tokenize(i)) for i in documents])

tokens = [nltk.word_tokenize(i) for i in PledgesDf["PreProcessedText"]] 

""" Analysis on the tokens """

def BuildWordFreq(tokens):
    WordFreq = {}
    for sent in tokens:
        for i in sent:

            if i not in WordFreq.keys():
                WordFreq[i] = 1
            else:
                WordFreq[i] += 1
    return WordFreq

WordFreq = BuildWordFreq(tokens)

# Size of the vocabulary used in those pledges
print(len(WordFreq))
# 10 Most frequent words used in the pledges
sorted(WordFreq, key=WordFreq.get, reverse=True)[:10]


""" LLM model """

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )


""" Indexing Pledges with Mean Embedding """

# converting text to numerical data using LLM model
vectorsLLM = pledgeEmbedding(documents, tokenizer, model)

DocIndexV1 = pd.DataFrame(vectorsLLM)# Outputting the indexed pledges file

IndexedPath = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\IndexedDataV1.csv"
DocIndexV1.to_csv(IndexedPath)


""" LLM model """

tokenizer = AutoTokenizer.from_pretrained("veroman/TourBERT",do_lowercase=True, truncation=True, padding="max_length",model_max_length=128)

# Load pre-trained model (weights)
model = AutoModel.from_pretrained("veroman/TourBERT",
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )


""" Indexing Pledges with Mean Embedding """

# converting text to numerical data using LLM model
vectorsLLM = pledgeEmbedding(documents, tokenizer, model)

DocIndexV1 = pd.DataFrame(vectorsLLM)# Outputting the indexed pledges file

IndexedPath = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\IndexedDataV2.csv"
DocIndexV1.to_csv(IndexedPath)


""" LLM model """

tokenizer = BertTokenizerFast.from_pretrained(str(DirPpath.absolute()) + '\\LLM-for-Tourism\\Clustering\\2.1. WordEmbedding\\model\\BERT\\', config=BertConfig())
tokenizer.model_max_length = 512
tokenizer.init_kwargs['model_max_length'] = 512

# Load pre-trained model (weights)
model = AutoModel.from_pretrained(str(DirPpath.absolute()) + '\\LLM-for-Tourism\\Clustering\\2.1. WordEmbedding\\model\\BERT\\',
                                   output_hidden_states = True, # Whether the model returns all hidden-states.
                                   )


""" Indexing Pledges with Mean Embedding """

# converting text to numerical data using LLM model
vectorsLLM = pledgeEmbedding(documents, tokenizer, model)

DocIndexV1 = pd.DataFrame(vectorsLLM)# Outputting the indexed pledges file

IndexedPath = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\IndexedDataV3.csv"
DocIndexV1.to_csv(IndexedPath)


""" ROBERTA model """

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


# Load pre-trained model (weights)
model = RobertaModel.from_pretrained('roberta-base', output_hidden_states = True) # Whether the model returns all hidden-states.


""" Indexing Pledges with Mean Embedding """

# converting text to numerical data using LLM model
vectorsLLM = pledgeEmbedding(documents, tokenizer, model, Bert=False)

DocIndexV1 = pd.DataFrame(vectorsLLM)# Outputting the indexed pledges file

IndexedPath = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\IndexedDataV4.csv"
DocIndexV1.to_csv(IndexedPath)


""" Fine-tuned LLM model """

tokenizer = RobertaTokenizerFast.from_pretrained(str(DirPpath.absolute()) + '\\LLM-for-Tourism\\Clustering\\2.1. WordEmbedding\\model\\RoBERTa\\', config=RobertaConfig())
tokenizer.model_max_length = 512
tokenizer.init_kwargs['model_max_length'] = 512

# Load pre-trained model (weights)
model = AutoModel.from_pretrained(str(DirPpath.absolute()) + '\\LLM-for-Tourism\\Clustering\\2.1. WordEmbedding\\model\\RoBERTa\\',
                                   output_hidden_states = True, # Whether the model returns all hidden-states.
                                   )


""" Indexing Pledges with Mean Embedding """

# converting text to numerical data using LLM model
vectorsLLM = pledgeEmbedding(documents, tokenizer, model, Bert=False)

DocIndexV1 = pd.DataFrame(vectorsLLM)# Outputting the indexed pledges file

IndexedPath = str(DirPpath.absolute()) + "\LLM-for-Tourism\Clustering\OutputFiles\IndexedDataV5.csv"
DocIndexV1.to_csv(IndexedPath)
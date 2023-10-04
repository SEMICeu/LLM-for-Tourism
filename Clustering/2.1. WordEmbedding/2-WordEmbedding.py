""" Importing relevant packages """
import os # For finding pre-processed data
from pathlib import Path

import pandas as pd # For data handling
import numpy as np

import nltk #  For nlp processing
from sklearn.feature_extraction.text import TfidfVectorizer # For obtaining Tf-Idf tokenization

import gensim # For building and fine-tuning Word2Vec model
from gensim.models import Word2Vec
import gensim.downloader as api # Helpful for downloading pre-trained models

""" Loading the pre-processed data """

DirPpath = Path(os.path.abspath('')).parent # Fetching the current directory path
PledgesCsvPath = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\PreprocessedData.csv" 

print("The current location of PreprocessedData.csv is: ", PledgesCsvPath)

PledgesDf = pd.read_csv(PledgesCsvPath, index_col=0) # Loading the preprocessed pledges into a dataframe

print(PledgesDf.head()) # Controlling the data loaded


""" Tokenize the pledges on words """

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


""" Word2Vec model """

wv2 = api.load('word2vec-google-news-300') # Downloading a word2vec model pre-trained on Google News dataset, i.e., trained on more than 100 bn words
# Model contains a corpora of 3bn words represented by vectors of dimension 300

model1 = wv2 # First approach: Rely on an unmodified version of the pre-trained word2vec

# Fine tuning to add here
# https://czarrar.github.io/Gensim-Word2Vec/

""" Indexing Pledges with Mean Embedding """

#building Word2Vec representation of each pledge using an averaging approach
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
    def fit(self, X, y):
            return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    

w2v = dict(zip(model1.index_to_key, model1.vectors))
modelw = MeanEmbeddingVectorizer(w2v)

# converting text to numerical data using Word2Vec
vectorsW2v = modelw.transform(tokens)

DocIndexV1 = pd.DataFrame(vectorsW2v)# Outputting the indexed pledges file

IndexedPath = str(DirPpath.absolute()) + "\semic_pledges\OutputFiles\IndexedDataV1.csv"
DocIndexV1.to_csv(IndexedPath)


# """ Indexing Pledges with Tf-Idf Embedding """

# TfIdfVectorizer = TfidfVectorizer(analyzer='word',stop_words='english')

# TfIdfWm = TfIdfVectorizer.fit_transform(PledgesDf["PreProcessedText"])

# TfIdfTokens = TfIdfVectorizer.get_feature_names_out()
# DfTfIdfVect = pd.DataFrame(data = TfIdfWm.toarray(),columns = TfIdfTokens)

# print("\nTD-IDF Vectorizer\n")
# print(DfTfIdfVect)

# #building Word2Vec representation of each pledge using a Tf-Idf approach
# class TfIdfEmbeddingVectorizer(object):
#     def __init__(self, word2vec):
#         self.word2vec = word2vec
#         # if a text is empty we should return a vector of zeros
#         # with the same dimensionality as all the other vectors
#         self.dim = len(next(iter(word2vec.values())))
#     def fit(self, X, y):
#             return self

#     def transform(self, X, tfidf):

#         DocList = []
#         i = 0
        
#         for words in X:

#             WordList = []

#             for w in words:
                 
#                 try:
#                     if w in self.word2vec:
#                         weight = tfidf[w].iloc[i]
#                         WordList.append(self.word2vec[w] * weight)
#                     else:
#                         WordList.append(np.zeros(self.dim))
#                 except:
#                     WordList.append(np.zeros(self.dim))

#             i+=1
#             DocList.append(np.sum(np.array(WordList), axis = 0))
        
#         return np.array(DocList)
    

# w2v2 = dict(zip(wv2.index_to_key, wv2.vectors))
# ModelW2 = TfIdfEmbeddingVectorizer(w2v2)

# # converting text to numerical data using Word2Vec
# VectorsW2v2 = ModelW2.transform(tokens, DfTfIdfVect)

# print(VectorsW2v2)

# DocIndexV1 = pd.DataFrame(VectorsW2v2)# Outputting the indexed pledges file

# IndexedPath = str(DirPpath.absolute()) + "\semic_pledges\IndexedDataV1Tf.csv"
# DocIndexV1.to_csv(IndexedPath)


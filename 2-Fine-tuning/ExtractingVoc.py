""" Importing Relevant Libraries """
from pathlib import Path    # for accessing files
import os 

import pandas as pd  # for handling data

from tokenizers import BertWordPieceTokenizer # for tokenising the data
import transformers
import tokenizers


""" Loading the corpus data: corpus.csv """

DirPpath = Path(os.path.abspath('')).parent
file = str(DirPpath) + "\DataCollection\corpus.csv"

CorpusDF = pd.read_csv(file)

""" Preparation of data for fine-tuning """

# Final cleaning (drop NaN and unnecessary information)
CorpusDF = CorpusDF.dropna()
CorpusDF['0'] = CorpusDF['0'].apply(lambda x: x.replace('skip to main content this site uses cookies to offer you a better browsing experience. find out more on how we use cookies. accept all cookies accept only essential cookies an official website of the european union', ''))

# Save the cleaned file (for storing in S3)
file = str(DirPpath) + "\Fine-tuning\Files\corpus.csv"
CorpusDF.to_csv(file)

# Create a .txt file (format needed for the next steps) where each line is one document
file = str(DirPpath) + "\Fine-tuning\Files\corpus.txt"
with open(file, 'w', encoding='utf-8') as f:
    for ID, content in zip(CorpusDF['Unnamed: 0'].values, CorpusDF['0'].values):
        f.write('\n'.join([str(ID), content]))

""" Define global variables """

# LOCAL_INPUT_PATH is the location of corpus.txt 
LOCAL_INPUT_PATH = file 
# LOCAL_OUTPUT_PATH is the location where the tokenizer will save the new vocabulary
LOCAL_OUTPUT_PATH =  str(DirPpath) + "vocab"
VOCAB_SIZE = 30522


""" Extracting  and exporting the voacbulary """

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=file, vocab_size=VOCAB_SIZE)

DirPpath = Path(os.path.abspath('')).parent
OutputPath = str(DirPpath) + "\Fine-tuning\Files"

tokenizer.save_model(OutputPath)
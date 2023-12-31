""" Importing Relevant Libraries """
from pathlib import Path    # for accessing files
import os 

import pandas as pd  # for handling data

from tokenizers import BertWordPieceTokenizer # for tokenising the data
import transformers
import tokenizers
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


""" Loading the corpus data: corpus.csv """

DirPpath = Path(os.path.abspath('')).parent
file = str(DirPpath) + "\\1-DataCollection\Files\corpus.csv"

CorpusDF = pd.read_csv(file)

""" Preparation of data for fine-tuning """

# Final cleaning (drop NaN and unnecessary information)
CorpusDF = CorpusDF.dropna()
CorpusDF['0'] = CorpusDF['0'].apply(lambda x: x.replace('skip to main content this site uses cookies to offer you a better browsing experience. find out more on how we use cookies. accept all cookies accept only essential cookies an official website of the european union', ''))

# Save the cleaned file (for storing in S3)
file = str(DirPpath) + "\\2-Fine-tuning\Extracting Voc\Files\corpus.csv"
CorpusDF.to_csv(file)

# Create a .txt file (format needed for the next steps) where each line is one document
file = str(DirPpath) + "\\2-Fine-tuning\Extracting Voc\Files\corpus.txt"
with open(file, 'w', encoding='utf-8') as f:
    for ID, content in zip(CorpusDF['Unnamed: 0'].values, CorpusDF['0'].values):
        f.write('\n'.join([str(ID), content]))

""" Extracting  and exporting the voacbulary """
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

# Customize training
tokenizer.train(files=file, vocab_size=8192, min_frequency=2,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>",
])
#Save the Tokenizer to disk
output = str(DirPpath) + "\\2-Fine-tuning\Extracting Voc\Files\RoBERTa"
tokenizer.save_model(output)

#Saving as csv
df = pd.read_csv(output + "\merge.txt", delimiter="/t", header=None)
df.to_csv(output + "\merges.csv")
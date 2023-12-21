""" Importing the relevant libraries """

import string # For handling textual data
import re # For preprocessing

import nltk # NLP libraries and packages for preprocessing
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import contractions # For dealing with contractions, e.g., I'm --> I am

# To use only once to download all the libraries from nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

""" Defining common pre-processing functions """

def FirstClean(text): 
    """
    FirstClean takes a Text as input and replace tabulations (\n and _x000D_) by whitespaces

    :param text: A piece of text under string format
    :return: Original string without \n and _x000D_
    """

    return " ".join(text.split()).replace("_x000D_",". ")

def ReplaceContractions(text):
    """
    ReplaceContractions replaces contractions in string of text, e.g., I'm --> I am

    :param text: A piece of text in string format
    :return: Original string with the contractions replaced by their full form
    """
    
    return contractions.fix(text)

def PreProcess(text): 
    """
    PreProcess applies a set of common preprocessing transformations on a text

    :param text: Piece of text in string format
    :return: Preprocessed string
    """

    # Dealing with errors created by the scraping    
    text = re.sub(r'\s-([a-zA-Z0-9])', r'-\1', text)  # Correcting "low -carbon" in "low-carbon"      
    text = re.sub(r'\sth\s', " th", text) # Correcting "th is" by "this"
    text = re.sub(r'\sTh\s', " Th", text) # Correcting "Th is" by "This"
    text = re.sub(r'\spro\s', " pro", text) # Correcting "pro duction" by "production"
    text = re.sub(r'\scon\s', " con", text) # Correcting "con duct" by "conduct"
    text = re.sub(r'\sCon\s', " Con", text) # Correcting "Con duct" by "Conduct"
    text = re.sub(r'\ssu\s', " su", text) # ...
    text = re.sub(r'\sst\s', " st", text)
    text = re.sub(r'\sSt\s', " St", text)
    text = re.sub(r'\sAc\s', " Ac", text)
    text = re.sub(r'\sac\s', " ac", text)
    text = re.sub(r'\sex\s', " ex", text)
    text = re.sub(r'\sres\s', " res", text)

    # General cleaning functions
    text = text.lower() # Lowercase all the characters from the string
    text = re.sub(r'\n', ' ', text) # remove tabulation
    text = re.compile('<.*?>').sub(' ', text) # Remove links

    text = text.encode('ascii', 'ignore').decode()  # remove unicode characters
    text = re.sub(r'https*\S+', ' ', text) # remove https links
    text = re.sub(r'http*\S+', ' ', text) # remove http links
    text = re.sub(r'\s[^\w\s]\s', ' ', text)
    
    text = contractions.fix(text) # Manages contractions
    text = text.strip() # Remove trailing and leading whitespaces
    text = re.sub(r'\s{2,}', ' ', text) # Remove double and triple whitespaces

    return text #Output - Same string after all the transformations

nltk.download('stopwords')

def StopWord(string): 
    """
    StopWord removes stopwords from a string using NLTK stopwords corpus

    :param string: A string
    :return: Original string without stopwords
    """

    a = [i for i in string.split() if i not in stopwords.words('english')] # Removing usual english stopwords from the string
    return ' '.join(a) #Output - Same string after all the transformations

def PreProcessing(text, n):
    """
    Preprocessing combines all the previous functions to preprocess a text

    :param text: Piece of text in a string format to be preprocessed
    :param n: integer to count the number of pledge
    :return: Preprocessed text
    """
    n = n +1
    
    print("**************")
    print("n is : ")
    print(n)
    print("length of the text is : ")
    print(len(FirstClean(text)))
    return PreProcess(ReplaceContractions(FirstClean(text)))
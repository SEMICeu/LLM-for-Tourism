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

    text = re.sub(r'https*\S+', ' ', text) # remove links
    text = re.sub(r'http*\S+', ' ', text)
    text = text.lower() # Lowercase all the characters from the string
    text = text.strip() # Remove the leading and trailing whitespaces
    text = re.compile('<.*?>').sub(' ', text)
    # #text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text) # Removing Punctuation
    # #text = re.sub(r'\[[0-9]*\]', ' ', text)
    # #text = re.sub(r'\d', '', text) # Removing digits
    # #text = re.sub(r"\b[a-zA-Z]\b", "", text) # Removing single characters
    # #text = re.sub(r'[^\w\s]', '', str(text)) # Remove non alphanumeric characters

    text = text.encode('ascii', 'ignore').decode()  # remove unicode characters
    # # cleaning up text
    #text = re.sub(r'\'\w+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s[^\w\s]\s', ' ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'\(\d+\)', ' ', text)
    text = re.sub(r'\w+\d+\w+', ' ', text)
    text = text.strip()
    text = contractions.fix(text)
    text = re.compile(r'\s\.+').sub(' ', text)
    text = re.sub(r'\s+', ' ', str(text).strip()) # Replacing "double, triple, etc" whitespaces by one
    text = text.strip() # Remove the leading and trailing whitespaces

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
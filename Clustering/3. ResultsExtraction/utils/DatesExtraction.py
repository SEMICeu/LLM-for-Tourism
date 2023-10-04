""" Importing relevant libraries """
import pandas as pd # For data handling

import re # For dealing with regular expressions
import spacy # For tokenization and nlp analysis

# For date extraction and formatting
from word2number import w2n 
from calendar import month_name
from datetime import datetime

""" Sentence Tokenization of Pledges """

def SentTokenization(PledgeDf, nlp):

    """
    SentTokenization applies spacy's tokenization algorithm to tokenize a pledge on sentences

    :param PledgeDf: pandas DataFrame containing a column of preprocessed pledge
    :param nlp: nlp model
    :return: List of list of sentence tokens (one list per pledge)
    """

    SentTokens = [[sent.text for sent in nlp(doc).sents] for doc in PledgeDf["PreProcessedText"]]

    return SentTokens


""" Dates Extraction """

def NERDates(doc, datelim):

    """
    NERDates looks for DATES > than a given limit into a given sentence using NER from spacy

    :param doc: spacy object containing the sentence to be analysed --> nlp(sentence)
    :param datelim: date limit (int)
    
    :return: set of DATE entities > datelim
    """
    
    # Add an NER entity, ent from doc.ents, only if its a DATE + the year it contains > datelim or it is one of the two exceptions   
    dates = set((ent.text) for ent in doc.ents if ent.label_ == "DATE" and ((re.search(r"[12][0-9]{3}", ent.text) and int(re.findall(r"[12][0-9]{3}", ent.text)[0]) > datelim) or (re.search(r"^(-within|up to)", ent.text)))) #Limit ourself to expression with years > 2022
    
    datesPattern = re.compile(r'(\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3})')
    filteredDates = set(date for date in dates if not datesPattern.match(date))

    return filteredDates

def ExceptionDates(doc, filteredDates, datelim):

    """
    ExceptionDates looks for DATES 1) not identified by NER and 2) > than a given limit. This is done by scanning sentences using regex

    :param doc: spacy object containing the sentences to be analysed --> nlp(sentence)
    :param filteredDates: set of Dates identified by NER in doc
    :param datelim: date limit (int)
    
    :return: update filteredDates with new Dates
    """

# Exceptions 1 --> Dates intervals using regex
    
    # Finding all intervals from a sentence
    exceptions = re.findall(r"[12][0-9]{3}\-[12][0-9]{3}|[12][0-9]{3}\/[12][0-9]{3}", doc.text)
    
    # Loop over the different intervals
    for exception in exceptions:

        # Verify the interval respects the datelim condition
        date = re.findall(r"[12][0-9]{3}$", exception)[0]
        if int(date) > datelim:
            filteredDates.add(str(date)) # Use the end of interval for this sentence

# Exceptions 2 --> Specific structures (based on a human analysis of the sentences) using regex

    # Finding specific structures
    exceptions = re.findall(r"(by 2025 data|up to 2027 are:|Target: 2023|the project - 2023|end of 2023 NBTC|a framework Q2 2023)", doc.text)
    
    # Loop over the different elements
    for exception in exceptions:

        # Verify it respects the datelim condition
        if int(re.findall(r"[12][0-9]{3}", exception)[0]) > datelim:
            filteredDates.add(exception)


def ExtractYears(filteredDates):

    """
    ExtractYears transform DATES into a uniform month/year format by using different mapping rules

    :param filteredDates: Set of DATES identified in a sentence

    :return: A set of years 
    """
    
    years = set() # New set to contain the transformed DATES

    # Map to change season, quarter, etc. to month expressions
    mapDict = {"Summer": "August", "summer": "August", "Fall": "November", "fall": "November", "Winter": "February", "winter": "February", "Spring": "June", "spring": "June", "the Summer of": "August", "summer of": "August", "Fall of": "November", "fall of": "November", "Winter of": "February", "winter of": "February", "Spring of": "June", "spring of": "June", "Q1": "March", "Q2": "June", "Q3": "September", "Q4": "December", "early": "January", "the end of": "December", "First half of": "June", "first half of": "June", "Second half of": "December", "second half of": "December", "1st semester of": "June", "2nd semester of": "December", "1st quarter of": "March", "2nd quarter": "June", "3rd quarter": "September", "4th quarter": "December"}

    # Looping over the different DATES from the set filteredDates
    for date in filteredDates:

        # Step 1: Transforming (when needed) every time expression by their respective month using mapDict rules 
        try:
            pattern = '|'.join(list(mapDict.keys()))
            date = re.sub(pattern, mapDict[re.search(pattern, date).group(0)], date)
        except:
            print("")
        
        # Step 2: Keeping only Month XXXX from the date text
        try: # General Case, there is a month + a year in date
            pattern = 'January [12][0-9]{3}|February [12][0-9]{3}|March [12][0-9]{3}|April [12][0-9]{3}|May [12][0-9]{3}|June [12][0-9]{3}|July [12][0-9]{3}|August [12][0-9]{3}|September [12][0-9]{3}|October [12][0-9]{3}|November [12][0-9]{3}|December [12][0-9]{3}'
            year = re.findall(pattern, date, re.IGNORECASE)[0]

        except:
            try: # Exception 1, there is only a year in the date
                year = "January " + str(re.findall(r"[12][0-9]{3}", date)[0])
            except:
                try: # Exception 2, there is no year but a reference to a time difference (within one year, etc.)
                    gap = w2n.word_to_num(date)
                    year = "January " + str(2023 + gap)
                except:
                    gap = int(re.findall(r'\d+', date)[0])
                    year = "January " + str(2023 + gap)

        # Step 3: Transform the date from string to Datetime format
        year = datetime.strptime(year, '%B %Y')

        # Add the transformed year to the set
        years.add(year)
    
    return years


def DatesExtraction(PledgeDf):

    """
    DatesExtraction takes a dataframe containing the text from pledges, extract sentences from it and looks for dated sentences

    :param PledgeDf: pandas Dataframe containing at least a column named "Preprocessed Text", "Topics" and "Cluster"

    :return: A pandas dataframe with 5 columns --> Pledge text, Sentence, Date, Topic, Cluster
    """

    nlp = spacy.load("en_core_web_sm")

    DatesPledge = [] # List of 0-1 where each item indicates the presence or absence of dates in the pledge having the same index
    # MultDatesSent = [] # Uncomment if want insights on multdate sentences   
    DatesSent = [] # List of sentences containing a date
    topics = [] # List indicating the topic of the sentence
    clusters = [] # List indicating the cluster of the sentence
    Dates = [] # List of dates contained in the sentence
    Pledges = [] # List of pledges from which the sentence


    # Loop over Pledges = List of sentence tokens (+ their assigned topic and cluster)
    for Pledge, topic, cluster in zip(SentTokenization(PledgeDf, nlp), PledgeDf["Topics"], PledgeDf["Cluster"]):

        datesPresent = 0 # Variable to indicate the presence (1) or absence (0) of Dates in a Pledge
        
        # Loop over the sentences in the pledge
        for sent in Pledge:

            doc = nlp(sent) # Create a spacy object from the sentence

            filteredDates = NERDates(doc, 2022) # Find Dates (> 2022) using NER
            ExceptionDates(doc, filteredDates, 2022) # Add exceptions to the set
            
            # Set of actions if a date was found in the pledge
            if filteredDates != set():

                datesPresent = 1 # There is a date in the pledge

                # Format the DATE into month/year format
                print(filteredDates)
                years = ExtractYears(filteredDates)
                print(years)
                print("------------------------------------------------")

                # Dealing with multidates sentences
                if len(years)> 1:
                    
                    #MultDatesSent.append(sent) # Uncomment if you want an analysis on the mutltiple dates sentences
                    # + ADD A FUNCTION TO SPLIT SENTENCES INTO CLAUSES CONTAINING THE different DATES
                    
                    # Loop over the different dates in the set
                    for date in years:
                        
                        # Append the relevant element to their respective lists (future columns of a dataframe)
                        Dates.append(date) 
                        DatesSent.append(sent)
                        topics.append(topic)
                        clusters.append(cluster)
                        Pledges.append(" ".join(Pledge))
                
                # Sentences with one date
                else:
                    Dates.append(list(years)[0])
                    DatesSent.append(sent)
                    topics.append(topic)
                    clusters.append(cluster) 
                    Pledges.append(" ".join(Pledge))
                
        DatesPledge.append(datesPresent) # For each pledge indicate if a date was found

    DatesResults = pd.DataFrame({"Pledge": Pledges, "Sentences": DatesSent, "Dates": Dates, "Topics": topics, "Clusters": clusters})
    print(DatesResults.head())

    return DatesResults
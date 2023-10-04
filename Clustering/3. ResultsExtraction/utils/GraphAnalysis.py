""" Importing relevant library """

import pandas as pd # For data handling

import matplotlib.pyplot as plt # For building plots
import seaborn as sns # For visualization

from datetime import datetime # For dealing with Date format

""" Timeline functions """

def GlobalTimeline(DatesResults, MinYear, MaxYear): 
    """
    GlobalTimeline create an histogram of pledges over a given time period

    :param DatesResults: pd df containing a set of years
    :param MinYear: Lower bound of the time interval
    :param MaxYear: Upper bound of the time interval

    :return: Histogram of the number of dates for each year
    """

    dateMin = datetime.strptime("January {year}".format(year = MinYear), '%B %Y')
    dateMax = datetime.strptime("December {year}".format(year = MaxYear), '%B %Y')

    data = DatesResults[(DatesResults["Dates"] >= dateMin) & (DatesResults["Dates"] <= dateMax) & (DatesResults["Results?"] == 1)]

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=data["Dates"], bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.title('Timeline of Results ({}-{})'.format(MinYear, MaxYear))
    #plt.xticks(rotation = 45, ha = 'right')
    plt.xlim([dateMin, dateMax])
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=150)

    plt.show()


def TimelineBy(data, name, dateMin, dateMax, type):
    """
    TimelineBy draw an histogram of results sentences by date between dateMin and Max for a given cluster (or topic) name

    :param data: pd df of dates for the chosen cluster (or topic)
    :param name: tupple containing the number of the cluster (or topic) at index 0
    :param dateMin: Lower bound of the Date interval
    :param dateMax: Upper bound of the Date interval
    :param type: string indicating whether we are plotting timeline by "Cluster" or "Topic"
    """

    # !!! Adapt the clusters mapping in function of the analysis from the Clustering (the numbering can vary from one iteration to another)
    mapping = {"Topic": {"1": "Fair measures for Short-Term Rentals (STR)", "2": "Regulatory support for improved multimodal travelling", "3": "Improving statistics and indicators for tourism", "4": "Comprehensive tourism strategies development or update", "5": "Collaborative local destination governance", "6": "Sustainable mobility", "7": "Circularity of hospitality industry", "8": "Green transition of tourism companies and SMEs", "9": "Data-driven tourism services", "10": "Improve the availability of information \n on tourism offer online", "11": "Easily accessible best practices, \n peer learning and networking for SMEs", "12": "R&I and pilots on circular and climate friendly tourism", "13": "Appropriation of PEF and OEF methodology \n and development of support tools for tourism ecosystem", "14": "Technical implementation for tourism data space", "15": "R&I for digital tools and services in tourism", "16": "Support for digitalisation of tourism SMEs and destinations", "17": "Seamless cross-border travelling", "18": "Coordinated management and updated information on travelling", "19": "Awareness raising on skills needs for twin transition in tourism", "20": " Awareness raising on changes in tourism demand \n and the opportunities of twin transition for tourism", "21": "Educational organisations to engage \n in developing and renewing tourism education", "22": "Pact for skills in tourism", "23": "One-stop-shop for learning opportunities for tourism SMEs", "24":"Fairness and equality in tourism jobs", "25": "Enhancing accessible tourism services", "26": "Tourism services for visitors and residents alike", "27": "Support visibility of funding opportunities for tourism actors", "28": "Other topic"}, "Cluster": {"1": "Data", "2": "Digital", "3": "Sustainable Strategy", "4": "Education & Cooperation", "5": "Sustainable Mobility", "6": "Investment & Start-ups"}}
    mapping = mapping[type]

    max = {"Topic": 30, "Cluster": 50}

    n = data.groupby("Dates").size()

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=data["Dates"], bins=n.size, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.title(type + ' ' + str(name[0]) + ': ' + mapping[str(name[0])])
    #maxfreq = n.max()
    plt.xticks(rotation = 45, ha = 'right')
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=max[type])
    plt.xlim([dateMin, dateMax])
    plt.show()
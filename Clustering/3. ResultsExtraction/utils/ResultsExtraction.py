""" Importing relevant libraries """

import re # For dealing with regular expressions


""" Extracting results """


def ResultsCategories(sent, Report, Event, Training, Project, Practice, Award):
    
    # List of results categories and their typical phrases !!! To improve based on a deeper analysis of the pledges + add extra categories
    ReportPattern = "report|Report|reports|Reports|study|Study|studies|Studies|reporting|Reporting"
    EventPattern =   "event|events|Event|Events|conference|conferences|Conference|Conferences|webinar|webinars|Webinar|Webinars"
    TrainingPattern = "training|trainings|Training|Trainings|course|courses|Course|Courses"
    ProjectPattern = "project result|project results|Project result|Project results|Fact sheet|Fact sheets|fact sheet|fact sheets"
    PracticePattern = "Best practice|Best practices|best practice|best practices|best practice collecion|Best practice collection|best practices collection|Best practices collection|newsletters|newsletter|Newsletter|Newsletters"
    AwardPattern = "award|Award|awards|Awards|certification|Certification|Certifications|certifications|label|Label|labels|Labels"
    
    # For each results category check if one of its respective pattern can be found in the pledge, and update the provided lists in consequence
    if re.search(ReportPattern, sent, re.IGNORECASE):
            Report.append(1)
    else: 
        Report.append(0)
    
    if re.search(EventPattern, sent, re.IGNORECASE):
        Event.append(1)
    else:
        Event.append(0)

    if re.search(TrainingPattern, sent, re.IGNORECASE):
        Training.append(1)
    else:
        Training.append(0)

    if re.search(ProjectPattern, sent, re.IGNORECASE):
        Project.append(1)
    else:
        Project.append(0)

    if re.search(PracticePattern, sent, re.IGNORECASE):
        Practice.append(1)
    else:
        Practice.append(0)

    if re.search(AwardPattern, sent, re.IGNORECASE):
        Award.append(1)
    else:
        Award.append(0)



def GetResults(DatesSent):
    """
    GetResults analyses Dated sentences and identifies the presence of results

    :param DatesSent: List of sentences containing dates

    :return: List of 0 and 1 indicating the presence of results into the sentences ; List of sentences containing results
    """

    # List of typical phrases announcing a results
    ResultsPatterns = "goal|goals|target|targets|objective|objectives|result|results|aim|aims to have by|increases by|decreases by|we expect to|expected for|are expected to|for [12][0-9]{3}|expect that by|as of [12][0-9]{3}|in [12][0-9]{3}|for [12][0-9]{3}|by [12][0-9]{3}|expected by|aimed at|our target for [12][0-9]{3} is to have|should be online|should be ready|from [12][0-9]{3}|measurable|milestones|report|Report|reports|Reports|event|events|Event|Events|conference|conferences|Conference|Conferences|webinar|webinars|Webinar|Webinars|training|trainings|Training|Trainings|course|courses|Course|Courses|project result|project results|Project result|Project results|Fact sheet|Fact sheets|fact sheet|fact sheets|Best practice|Best practices|best practice|best practices|best practice collecion|Best practice collection|best practices collection|Best practices collection|newsletters|newsletter|Newsletter|Newsletters"

    # Variables
    ResultsSent = []
    Results = []

    Report = []
    Event = []
    Training = []
    Project = []
    Practice = []
    Award = []

    # Loop over the dated sentences
    for sent in DatesSent:
        try: 
            # Identify results sentences based on patterns
            re.findall(ResultsPatterns, sent, re.IGNORECASE)[0]
            ResultsSent.append(1) 
            Results.append(sent)
            ResultsCategories(sent, Report, Event, Training, Project, Practice, Award)

        except:
            ResultsSent.append(0)
            Report.append(0)
            Event.append(0)
            Training.append(0)
            Project.append(0)
            Practice.append(0)
            Award.append(0)

    return ResultsSent, Report, Event, Training, Project, Practice, Award, Results

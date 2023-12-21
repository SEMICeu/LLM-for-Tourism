import pandas as pd # for data handling
import time


import pikepdf # for extracting text from pdfs
from PyPDF2 import PdfReader # for reading online pdfs

import requests # for accessing online resources
import io

import os
from pathlib import Path # For fetching the required files

from selenium import webdriver
from selenium.webdriver.common.by import By # web scraping libraries

file = "TransitionPathwayForTourism.pdf"


def scrapingURLs(file):
    """
    scrapingURLs takes a pdf file as input and extract all the references to urls from it

    :param file: A pdf file
    :return: List of urls scraped from the file
    """
    
    pdfFile = pikepdf.Pdf.open(file) # Opening the pdf file and storing the content in a variable

    urls = [] # List to store the scraped urls

    for page in pdfFile.pages: # Loop over the pages of the pdf file
        try: #If there is any annotation on the page then
            for annots in page.get("/Annots"):
                uri = annots.get("/A").get("/URI")
                if uri is not None and uri not in urls:
                    print("[+] URL Found:", uri)
                    urls.append(uri)
        except: #Else print an error message
            print("No annotation in this page")

    print("[*] Total URLs extracted:", len(urls)) # output the number of urls scraped

    return urls

def addingURLs(urls, badURLS, newURLS):
    """
    addingURLS takes a list of urls, removes unproper, and add new urls based on two provided lists

    :param urls: list of urls to modify
    :param badURLS: list of urls to remove
    :param newURLS: list of urls to add
    :return: List of urls after modification
    """
    for url in badURLS:
        if url in urls:
            urls.remove(url)


    for url in newURLS:
        if url not in urls:
            urls.append(url)

    return urls

    
def extractPDF(url, types):
    """
    extractPDF takes link to an online pdf or local file (in function of chosen types) and returns the text contained in it

    :param url: string indicating the url link to an online pdf or the name of a local file
    :param types: string indicating whether an "url" or "pdf" was entered
    :return: Text of the pdf as one string
    """
    DirPpath = Path(os.path.abspath('')).parent

    if types == "url":
        response = requests.get(url)
        with io.BytesIO(response.content) as f:
                    pdf = PdfReader(f)
                    number_of_pages = len(pdf.pages)

                    return ' '.join([pdf.pages[i].extract_text() for i in range(number_of_pages)])
    
    else: 
        pdf = PdfReader(open(str(DirPpath) + "\LLM-for-Tourism\DataCollection\PDF resources\\" + url, 'rb'))
        number_of_pages = len(pdf.pages)

        return ' '.join([pdf.pages[i].extract_text() for i in range(number_of_pages)])
    


def webScraping(urls):
    """
    webscraping takes a list of urls and scrapes the content from it

    :param urls: list of urls to scrape
    :return: List of the content scraped from the urls
    """
    content = [] # List to store the content of the pages
    count = 0 # Counter to record the number of issues
    issue = [] # Liste to record potential errors

    # Initiating the webdriver
    driver = webdriver.Chrome()

    # Looping over the urls
    for url in urls:
        driver.get('{}'.format(url))
        # To load entire webpage
        time.sleep(5)

        if "select-language" in driver.current_url: # Dealing with the select language page from EU website           
            # Block for selecting the language
            print("Select language page")
            button = driver.find_element(By.XPATH, "/html/body/div[3]/div/div/div/div/div[2]/ul/li[8]/a")
            link = button.get_attribute("href")
            driver.get(link)
            time.sleep(1)

            if driver.current_url == "https://commission.europa.eu/documents_en": # Specific issure from eu websites
                #Block for searching report
                issue.append(url)
                count += 1            
            else: # If no issue, scrape the content
                text = (driver.find_element(By.XPATH, "/html/body").text)        
        else: 
            # Printing the whole body text
            text = (driver.find_element(By.XPATH, "/html/body").text)

            if text == '':   # If the page is a pdf             
                try:
                    text = extractPDF('{}'.format(url), "url")               
                except:
                    issue.append(url)
                    count += 1

        content.append(text) # Add the scraped text to the list of content

    driver.close()

    return content

def PDFscraping(folder, content):
    """
    PDFscraping takes a folder name in which pdfs are stored and a list of text and adds the content of the pdfs to this list

    :param folder: string with the name of the folder in which the relevant pdfs have been stored
    :param content: list containing strings of already scraped content
    :return: List of the content scraped from the urls
    """   
    DirPpath = Path(os.path.abspath('')).parent
    pdfs = os.listdir(str(DirPpath) + "\LLM-for-Tourism\DataCollection\\" + folder)
    
    for pdf in pdfs:

        content.append(extractPDF(pdf, "file"))

    return content
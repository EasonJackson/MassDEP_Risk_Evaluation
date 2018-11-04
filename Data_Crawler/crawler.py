# This script is used for automatically download and save pdfs from 
# MassDEP website with specified RTN-submit time pairs
# Author @Ethan 2018

# To run the script
# In Shell:
# python3 crawler.py <full_name_of_TSAUD_file.xlsx>
# If run in notebook:
# Change TSAUD_file to <full_name_of_TSAUD_file.xlsx> instead

import requests
import re
import os
import sys
import stat
from lxml import html
import xlrd


## Constants
TSAUD_file = str(sys.argv[1]) # TSAUD file for rtn and time pairs, to run in notebook, use TSAUD_file = 'actual file name of TSAUD.xlsx'
CURR_DIR_NAME = os.getcwd() # current directory name
URL_REQUEST_PREFIX = "http://eeaonline.eea.state.ma.us/EEA/fileviewer/" # url prefix
TABLE_ROW_RECORDS_TEXT = '''//tr[@id='UltraWebTab1xxctl0xGrid_r_{0}']//text()''' # table row regex
TABLE_ROW_RECORDS_HREF = '''//tr[@id='UltraWebTab1xxctl0xGrid_r_{0}']//@href'''
FORM_NAME_PREFIX = "BWSC105 Immediate Response Action"
SUBMIT_RAW_TIME_FORMAT = re.compile(r'[0-9]+/[0-9]+/[0-9]{4}')
SUBMIT_TIME_FORMAT = re.compile(r'[0-9]+-[0-9]+-[0-9]{4}')

## Test query
QUERY_TEST = '2-020220'
TIME_TEST = '8-10-2017'


def get_html(query):
    '''
    Get HTTP response with query
    
    Argument
    query: rtn no.

    Return
    reponse: html page object
    '''
    response = ""
    try:
        response = requests.get(URL_REQUEST_PREFIX + "Rtn.aspx?rtn=" + query)
    except Exception:
        print("Cannot get response for query: {0}.".format(query))

    return response


def parse_html_response(response):
    '''
    Parsing html response into DOM tree structures

    Argument
    response: html format response

    Return
    tree: DOM tree object
    '''
    try:
        tree = html.fromstring(response.content)
    except Exception:
        return {}
    return tree


def create_folder(name):
    '''
    Create folder with provide name in current directory
    Change previlage of current folder with read, write and execute

    Argument
    name: name of folder in format of <rtn no.>_mm-dd-yyyy
    '''
    if name is None:
        return

    if os.path.isdir(CURR_DIR_NAME + "/" + name):
        return

    try:
        os.mkdir(name)
    except Exception:
        print("Cannot create folder {0}".format(name))


def get_records(tree):
    '''
    Search DOM tree with table row information
    Filter out row records when the form is BWSC105
    Return the corresponding text and links for pdfs

    Argument
    tree: DOM tree parsed from html response

    Return
    rows_text: list of <list of unicodes> text parsed from rows
    rows_href: list of <list of unicodes> links parsed from rows
    '''
    index = 0
    rows_text = []
    rows_href = []

    while True:
        text = []
        href = []
        try:
            text = tree.xpath(TABLE_ROW_RECORDS_TEXT.format(index))
            href = tree.xpath(TABLE_ROW_RECORDS_HREF.format(index))
        except Exception:
            print("Not able to get rows")

        if len(text) == 0 or text is None:
            break

        if text[1].startswith(FORM_NAME_PREFIX):
            rows_text.append(text)
            rows_href.append(href)

        index += 1

    return rows_text, rows_href


def check_time_match(text, time):
    '''
    Check if query rtn result has the matched submit time
    Two conditions:
        1. Time exactly matches
        2. Only year matches

    Argument
    text: text record in a row
    time: required time stamp

    Return
    flg: indicator on whether time matches
    time_result: real submit time from website
    '''

    submit_time_raw_text = text[2]
    time_result = re.match(SUBMIT_RAW_TIME_FORMAT, submit_time_raw_text).group(0)
    time_result = time_result.replace('/', '-')

    record_date = time_result.split('-')
    required_date = time.split('-')

    if time_result == time: # Entire time matches
        return True, time_result
    elif record_date[2] == required_date[2]: # Year matches
        return True, time_result

    return False, None


def download_save_pdf(text, href, folder_name, submit_real):
    '''
    Match IRA with its link
    It downloads pdfs whose link must contain word 'IRA'.

    Argument
    text: list of text in the row record
    href: all the url in the row record
    folder_name: folder to save pdf
    '''
    tmp = []
    count = 1
    for s in text[3:]:
        if s.endswith('.pdf') and ('IRA' in s):
            tmp.append(count)
            count += 1

    for index in tmp:
        if index < len(href):
            pdf_link = href[index]
            pdf = requests.get(URL_REQUEST_PREFIX + pdf_link)
            with open(CURR_DIR_NAME + "/" + folder_name + "/" + submit_real + "_report.pdf", "wb") as f:
                f.write(pdf.content)


def open_xlsx_rtn_time_list():
    '''
    Open excel file that contains required rtn_time list

    Return 
    rtns_submitTime: a dict key = rtn and value = submit time
    '''
    workbook = xlrd.open_workbook(TSAUD_file)
    sheet = workbook.sheet_by_index(0)
    rtns_submitTime = {} # index 0 and 1

    for rowx in range(1, sheet.nrows):
        col = sheet.row_values(rowx)
        rtn = str(col[0])
        submit_time = re.match(SUBMIT_TIME_FORMAT, col[1].split()[-1]).group(0)
        rtns_submitTime[rtn] = str(submit_time)

    return rtns_submitTime


## Main
if __name__ == "__main__":
    # Get rtn list needs to query from MassDEP
    rtns_submitTime = open_xlsx_rtn_time_list()


    for rtn, time in rtns_submitTime.items():
        # Create folders under the same directory
        name = rtn + "_" + time
        create_folder(name)

        # Send HTTP
        response = get_html(rtn)
        tree = parse_html_response(response)
        text_list, href_list = get_records(tree) # texts and hrefs match BWSC105

        for text, href in zip(text_list, href_list):
            flg, submit_real = check_time_match(text, time)
            if flg:
                download_save_pdf(text, href, name, submit_real)










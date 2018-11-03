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
import pyPDF2

## Constants
TSAUD_file = sys.argv[0] # TSAUD file for rtn and time pairs
CURR_DIR_NAME = os.getcwd() # current directory name
URL_REQUEST_PREFIX = "http://eeaonline.eea.state.ma.us/EEA/fileviewer/" # url prefix
TABLE_ROW_RECORDS_TEXT = '''//tr[@id='UltraWebTab1xxctl0xGrid_r_{0}']//text()''' # table row regex
TABLE_ROW_RECORDS_HREF = '''//tr[@id='UltraWebTab1xxctl0xGrid_r_{0}']//@href'''
FORM_NAME_PREFIX = "BWSC105 Immediate Response Action"
SUBMIT_TIME_FORMAT = re.compile(r'[0-9]+-[0-9]+-[0-9]{4}')

## Test query
QUERY_TEST = '2-020220'
TIME_TEST = '8-10-2017'


def get_html(query):
    '''
    Get HTTP response with query

    query: rtn no.
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

    response: html format response
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

    tree: DOM tree parsed from html response
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

        if len(row) == 0 or row is None:
            break

        if text[1].startswith(FORM_NAME_PREFIX):
            rows_text.append(text)
            rows_href.append(href)

        index += 1

    return rows_text, rows_href


def check_time_match(text, time):
    '''
    Check if query rtn result has the matched submit time

    text: text record in a row
    time: required time stamp
    '''

    # TODO: test time format matching 
    time_result = re.match(SUBMIT_TIME_FORMAT, text).group(0)

    if time_result == time:
        return True

    return False


def download_save_pdf(text, href_list, folder_name):
    '''
    Match IRA with its link

    text: list of text in the row record
    href_list: all the url in the row record
    folder_name: folder to save pdf
    '''
    tmp = []
    for s in text[3:]:
        if s.endswith('.pdf')
        tmp.add(s)

    for index in enumerate(tmp):
        if tmp[index].startswith('IRA')
            break

    href = href_list[1 + index]
    pdf = requests.get(URL_REQUEST_PREFIX + href)
    with open(CURR_DIR_NAME + "/" + folder_name + "report.pdf", "wb") as f:
        f.write(pdf.content)


def open_xlsx_rtn_time_list():
    '''
    Open excel file that contains required rtn_time list
    Return dict
    '''
    workbook = xlrd.open_workbook(TSAUD_file)
    sheet = workbook.sheet_by_index(0)
    rtns_submitTime = {} # index 0 and 1

    for rowx in range(sheet.nrows):
        col = sheet.row_values(rowx)
        rtns_submitTime[str(col[0])]
        submit_time.append(re.match(TIME_PATTERN, str(col[1])).group(1))

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

        for text in text_list:
            if check_time_match(text, time):
                download_save_pdf(text, href_list, name)










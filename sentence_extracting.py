import pandas as pd
import os


def get_raw_text(indi):
    files_n = os.listdir('Data/{0}/N'.format(indi))
    files_y = os.listdir('Data/{0}/Y'.format(indi))
    files_N = []
    files_Y = []
    for file in files_n:
        files_N.append('Data/{0}/N/'.format(indi) + file)

    for file in files_y:
       files_Y.append('Data/{0}/Y/'.format(indi) + file)

    final_text = []
    for file in files_N:
        with open(file, 'r', encoding='utf-8') as f:
            print('Read file: {0}'.format(file))
            doc = f.read()
            sentences = doc.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 0:
                    final_text.append(sentence)

    output = open('Data/{0}_Negtive.txt'.format(indi), 'w')
    for sentence in final_text:
        output.write(sentence + '\r')

    output.close()

    final_text = []
    for file in files_Y:
        with open(file, 'r', encoding='utf-8') as f:
            print('Read file: {0}'.format(file))
            doc = f.read()
            sentences = doc.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 0:
                    final_text.append(sentence)

    output = open('Data/{0}_Positive.txt'.format(indi), 'w')
    for sentence in final_text:
        output.write(sentence + '\r')

    output.close()
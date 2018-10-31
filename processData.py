import os
import pandas as pd
from fnmatch import fnmatch
import os, shutil, tempdir, tempfile
import PyPDF2
import textract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfFileWriter, PdfFileReader
from subprocess import check_call
from nltk.stem.porter import *
from nltk.corpus import stopwords
import string
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import pikepdf

stemmer = PorterStemmer()
stop_words = stopwords.words('english')
pd.set_option('display.max_columns', None)

class processData():
    def __init__(self, run_config):
        self.run_config = run_config
        self.report_name = []


    # append *.pdf to self.report_name
    def load_pdf_filename(self):
        for path, subdirs, files in os.walk(self.run_config['input_dir']):
            for name in files:
                if fnmatch(name, self.run_config['pattern']):
                    if not fnmatch(name, self.run_config['ignore']):
                        self.report_name.append(os.path.join(path, name))



    # extract content for each .pdf file
    def extract_pdf(self):
        self.load_pdf_filename()
        trainDF = pandas.DataFrame()
        # id save file id
        id = list(map(lambda x: x.split('/')[3][:9], self.report_name))
        trainDF['id'] = id
        # result save processed data
        result = []

        for filename in self.report_name:
                print(filename)
                # open allows you to read the file
                pdfFileObj = open(filename, 'rb')
                # The pdfReader variable is a readable object that will be parsed
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
                if pdfReader.isEncrypted:
                    try:
                        pdfReader.decrypt('')
                        print('File Decrypted (PyPDF2)')
                    except:
                        # command = "cp " + filename + " temp.pdf; qpdf --password='' --decrypt temp.pdf " + filename
                        # os.system(command)
                        # Use try/finally to ensure our cleanup code gets run
                        try:
                            # There are a lot of ways to mess up creating temporary files in a way
                            # that's free of race conditions, so just use mkdtemp() to safely
                            # create a temporary folder that only we have permission to work inside
                            # (We ask for it to be made in the same folder as filename because /tmp
                            #  might be on a different drive, which would make the final overwrite
                            #  into a slow "copy and delete" rather than a fast os.rename())
                            tempdir = tempfile.mkdtemp(dir=os.path.dirname(filename))

                            # I'm not sure if a qpdf failure could leave the file in a halfway
                            # state, so have it write to a temporary file instead of reading from one
                            temp_out = os.path.join(tempdir, 'qpdf_out.pdf')

                            # Avoid the shell when possible and integrate with Python errors
                            # (check_call() raises subprocess.CalledProcessError on nonzero exit)
                            check_call(['qpdf', "--password=", '--decrypt', filename, temp_out])

                            # I'm not sure if a qpdf failure could leave the file in a halfway
                            # state, so write to a temporary file and then use os.rename to
                            # overwrite the original atomically.
                            # (We use shutil.move instead of os.rename so it'll fall back to a copy
                            #  operation if the dir= argument to mkdtemp() gets removed)
                            shutil.move(temp_out, filename)
                            print('File Decrypted (qpdf)')

                        finally:
                            # Delete all temporary files
                            shutil.rmtree(tempdir)
                        # re-open the decrypted file
                        pdfFileObj = open(filename)
                        pdfReader = PdfFileReader(pdfFileObj)
                else:
                    print('File Not Encrypted')

                # discerning the number of pages will allow us to parse through all #the pages
                num_pages = pdfReader.numPages
                count = 0
                text = ""
                # The while loop will read each page
                while count < num_pages:
                    pageObj = pdfReader.getPage(count)
                    count += 1
                    text += pageObj.extractText()
                # This if statement exists to check if the above library returned #words. It's done because PyPDF2 cannot read scanned files.
                if text != "":
                    text = text
                # If the above returns as False, we run the OCR library textract to #convert scanned/image based PDF files into text
                else:
                    text = textract.process(filename, method='tesseract', language='eng')

                # The word_tokenize() function will break our text phrases into #individual words
                tokens = word_tokenize(text)
                # convert to lower case
                tokens = [w.lower() for w in tokens]
                # remove punctuation from each word
                table = str.maketrans('', '', string.punctuation)
                stripped = [w.translate(table) for w in tokens]
                # remove remaining tokens that are not alphabetic
                words = [word for word in stripped if word.isalpha()]
                # filter out stop words
                words = [w for w in words if not w in stop_words]
                stemmed = [stemmer.stem(word) for word in words]
                print(stemmed)
                result.append(stemmed)
        trainDF['text'] = result
        return trainDF
    # read label file
    def read_labels(self):
        label_file = self.run_config['label_file']
        label_data = pd.read_excel(label_file)
        label_data.replace({'Yes': True, 'No': False, 'Maybe': True}, inplace=True)
        label_data['label'] = label_data.iloc[:, 1:].any(axis=1)
        return label_data

    def count_vector(self):
        # create a count vectorizer object
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(self.trainDF['text'])

        # transform the training and validation data using count vectorizer object
        xtrain_count = count_vect.transform(self.trainDF)
        # xvalid_count = count_vect.transform(valid_x)
        return xtrain_count



    def feature(self):
        trainDF = self.extract_pdf()
        print(trainDF.columns)
        label_data = self.read_labels()
        
    #     # split the dataset into training and validation datasets
    #     train_x, valid_x = model_selection.train_test_split(self.trainDF['text'])
    #     xtrain_count = self.count_vector()
    #     print(xtrain_count)


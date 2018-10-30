
# TF-IDF word2vec method
# Author @Ethan 2018

import os
import re
import math
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


PORTER_STEMMER = PorterStemmer()


def clean_words(data):
    '''
    clean_words read from the raw .txt document.
    
    Firstly it tokenize original text into list of words. 
    Secondly skipping words that are not alphabetics.
    Thirdly it stems word into its root format to
    reduce forms of the same word in final list
    
    Argument:
    data - string, raw txt file
    
    Return:
    new_data - string, cleaned data
    '''
    new_data = []
    tokens = word_tokenize(data)
    for token in tokens:
        if token.isalpha():
            stemmed_word = PORTER_STEMMER.stem(token)
            new_data.append(stemmed_word)
        
    return new_data
        

def computeTF(data):
    '''
    cumputeTF reads a string based file 
    and return the tf_dict of words
    in the document
    
    Argument:
    data - string, document text
    
    Return:
    tf_dict - dict, TF dictionary of words in the
    document
    '''   
    tf_dict = {}
    total_words = 0
    for line in data:
        for word in line.split():
            total_words += 1
            if word in tf_dict:
                tf_dict[word] += 1
            else:
                tf_dict[word] = 1

    for key in tf_dict.keys():
        tf_dict[key] = tf_dict[key] / total_words
        
    return tf_dict


def computeIDF(tfDict_list):
    '''
    computeIDF reads a list of tf_dict
    and return the idf_dict of all words
    
    Argument:
    tfDict_list - list, list of TF dictionaries for
    used documents, in the same order
    
    Return:
    idf_dict - dict, IDF dictionary of words contained
    in all TF dictionaries
    '''
    idf_dict = {}
    N = len(tfDict_list)
    for dict in tfDict_list:
        for word, freq in dict.items():
            if word in idf_dict:
                idf_dict[word] += 1
            else:
                idf_dict[word] = 1
    
    for key in idf_dict.keys():
        idf_dict[key] = math.log(N / idf_dict[key])
        
    return idf_dict


def computeTF_IDF_Vec(tf_list, idf_dict):
    '''
    computeTF_IDF_Vec computes the word vector using
    TF_IDF method
    
    Arguments:
    tf_list - list, list of TF dictionaries
    idf_dict - dict, IDF dictionary of words in TF dicts
    
    Return:
    tf_idf_vec - dict, dictionary of word mapping to vectors
    '''
    tf_idf_vec = {}
    for word in idf_dict.keys():
        vec_tmp = []
        for tf_dict in tf_list:
            if word not in tf_dict:
                vec_tmp.append(0)
                continue
                
            score = tf_dict[word] * idf_dict[word]
            vec_tmp.append(score)
        
        tf_idf_vec[word] = vec_tmp
        
    return tf_idf_vec


# Create word list and vectors np array
# Write out to metadata.tsv for visualization
def create_metadata_tsv(tf_idf_vec):
    with open('tensorboard/metadata.tsv', 'w+') as metadata_f:
        words_list = [key for key in tf_idf_vec]
        word_vec = np.zeros((len(words_list), 10))
        for index, word in enumerate(words_list):
            metadata_f.write(word + '\n')
            vec = tf_idf_vec[word]
            for i in range(0, 10):
                word_vec[index][i] = vec[i]
        return word_vec



def visualize_wordvec(word_vec):
    sess = tf.InteractiveSession()
    with tf.device("/cpu:0"):
        embedding = tf.Variable(word_vec, trainable=False, name='Embedding')
    tf.global_variables_initializer().run()
    path = 'tensorboard'
    writer = tf.summary.FileWriter(path, sess.graph)
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'Embedding'
    embed.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(writer, config)
    saver = tf.train.Saver()
    saver.save(sess, path + '/model.ckpt', global_step=len(words_list) - 1)
    # RUN tensorboard --logdir="tensorboard" --port=8080


if __name__ == "__main__":
    # Compute tf_list and idf_dict for selected
    # series of documents
    n_of_documents = 10
    tf_list = []
    for index in range(n_of_documents):
        with open("text{0}.txt".format(index + 1), 'r') as f:
            f = clean_words(f.read())
            tf_dict_tmp = computeTF(f)
            tf_list.append(tf_dict_tmp)

    idf_dict = computeIDF(tf_list)

    # Word vectors based on TF-IDF
    tf_idf_vec = computeTF_IDF_Vec(tf_list, idf_dict)

    word_vec = create_metadata_tsv(tf_idf_vec)

    visualize_wordvec(word_vec)




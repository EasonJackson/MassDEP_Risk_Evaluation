import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing import sequence
import pandas as pd

# fix random seed for reproducibility
np.random.seed(7)

# Data file names
training_input_file = 'Data/Train.csv'
testing_input_file = 'Data/Test.csv'

file_word_to_int = 'Data/word_to_int.csv'

# Input data padding
max_sentence_length = 100

# Model parameters
vocab_size = 0
embedding_size = 128


def get_raw_text(indi):
    word_to_int_df = pd.read_csv(file_word_to_int, header=None)
    word_to_int = {}
    for key, value in zip(word_to_int_df[0], word_to_int_df[1]):
        word_to_int[str(key)] = int(value)

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
                    sentence_encoded, all_zero = encoding(sentence, word_to_int)
                    if not all_zero:
                        final_text.append(sentence_encoded)

    output = open('Data/{0}.csv'.format(indi), 'w')
    output.write('sequence,target\r')
    for sentence in final_text:
        output.write(sentence + ',' + '0' + '\r')

    final_text = []
    for file in files_Y:
        with open(file, 'r', encoding='utf-8') as f:
            print('Read file: {0}'.format(file))
            doc = f.read()
            sentences = doc.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 0:
                    sentence_encoded, all_zero = encoding(sentence, word_to_int)
                    if not all_zero:
                        final_text.append(sentence_encoded)

    for sentence in final_text:
        output.write(sentence + ',' + '1' + '\r')

    output.close()


def encoding(sentence, word_to_int):
    if len(sentence.split()) <= 1:
        return '', True
    res = ''
    flag = True
    for word in sentence.split():
        if word in word_to_int:
            res += str(word_to_int[word])
            flag = False
        else:
            res += '0'
        
        res += ' '

    return res, flag


def load_data():
    df_train = pd.read_csv(training_input_file)
    df_train['sequence'] = df_train['sequence'].apply(lambda x: [int(e) for e in x.split()])
    df_test = pd.read_csv(testing_input_file)
    df_test['sequence'] = df_test['sequence'].apply(lambda x: [int(e) for e in x.split()])
    X_train = np.array(df_train['sequence'].values[:])
    Y_train = np.array(df_train['target'].values[:])
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    X_test = np.array(df_test['sequence'].values[:])
    Y_test = np.array(df_test['target'].values[:])
    Y_test = Y_test.reshape((Y_test.shape[0], 1))

    # truncate and pad input sequences
    X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)

    print("X_train:")
    print(type(X_train))
    print(X_train.shape)

    print("Y_train:")
    print(type(Y_train))
    print(Y_train.shape)

    print("X_test:")
    print(type(X_test))
    print(X_test.shape)

    print("Y_test:")
    print(type(Y_test))
    print(Y_test.shape)        

    df = pd.read_csv(file_word_to_int, header=None)
    global vocab_size
    vocab_size = len(df)

    return X_train, Y_train, X_test, Y_test


def build_model():
    print("Creating model ... ")
    model = Sequential()
    print("Vocab size: " + str(vocab_size))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_sentence_length))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print("Compiling model ... ")
    # Optimizer adam or rmsprop
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    print(model.summary())
    print("Model completed.")
    return model
    

def train():
    X_train, Y_train, X_test, Y_test = load_data()
    model = build_model()
    print("Start fitting model ... ")
    model.fit(X_train, Y_train, epochs=10, batch_size=64, verbose=1)
    
    print("Evaluating model ... ")
    score, acc = model.evaluate(X_test, Y_test, batch_size=1)
    print("Test scure: " + str(score))
    print("Test accuracy: " + str(acc))


if __name__ == "__main__":
    train()
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing import sequence
import pandas as pd
from sklearn.metrics import confusion_matrix


# fix random seed for reproducibility
np.random.seed(7)

# Data file names
training_input_file = ''
testing_input_file = ''
file_word_to_int = ''
 
# Input data padding
max_sentence_length = 0

# Model parameters
vocab_size = 0
embedding_size = 128


def set_param(inidator):
    global training_input_file
    global testing_input_file
    global max_sentence_length
    global file_word_to_int
    if inidator == 'Sentence':
        training_input_file = 'Data/Train_Sentence.csv'
        testing_input_file = 'Data/Test_Sentence.csv'
        max_sentence_length = 100
    elif inidator == 'Document':
        training_input_file = 'Data/Train_Doc.csv'
        testing_input_file = 'Data/Test_Sentence.csv'
        max_sentence_length = 10000

    file_word_to_int = 'Data/word_to_int.csv'


def append_negative_data_as_sentence(indi):
    word_to_int_df = pd.read_csv(file_word_to_int, header=None)
    word_to_int = {}
    for key, value in zip(word_to_int_df[0], word_to_int_df[1]):
        word_to_int[str(key)] = int(value)

    files_n = os.listdir('Data/{0}/N'.format(indi))
    files_N = []

    for file in files_n:
        files_N.append('Data/{0}/N/'.format(indi) + file)

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

    output = open('Data/{0}_Sentence.csv'.format(indi), 'a')

    for sentence in final_text:
        output.write(sentence + ',' + '0' + '\r')

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
        
        res += ':'

    return res[:-1], flag


def transfer_raw_input():
    filename_train = 'Data/Train_Positive_mgf.csv'
    filename_test = 'Data/Test_Positive_mgf.csv'
    word_to_int_df = pd.read_csv(file_word_to_int, header=None)
    word_to_int = {}
    for key, value in zip(word_to_int_df[0], word_to_int_df[1]):
        word_to_int[str(key)] = int(value)

    train_raw = pd.read_csv(filename_train)
    train_raw['Sentence'] = train_raw['Sentence'].apply(lambda x: ':'.join([str(word_to_int[e.lower()]) if e.lower() in word_to_int else '0' for e in x.split()]))
    train_raw.to_csv('Data/Train_final.csv', index=False)

    test_raw = pd.read_csv(filename_test)
    test_raw['Sentence'] = test_raw['Sentence'].apply(lambda x: ':'.join([str(word_to_int[e.lower()]) if e.lower() in word_to_int else '0' for e in x.split()]))
    test_raw.to_csv('Data/Test_final.csv', index=False)


def encode_raw_doc(indi):
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
            sentence_encoded, _ = encoding(doc, word_to_int)
            final_text.append(sentence_encoded)

    output = open('Data/{0}_Document.csv'.format(indi), 'w')
    output.write('sequence,target\r')

    for doc in final_text:
        output.write(doc + ',' + '0' + '\r')

    final_text = []
    for file in files_Y:
        with open(file, 'r', encoding='utf-8') as f:
            print('Read file: {0}'.format(file))
            doc = f.read()
            sentence_encoded, _ = encoding(doc, word_to_int)
            final_text.append(sentence_encoded)   

    for doc in final_text:
        output.write(doc + ',' + '1' + '\r')

    output.close() 



def load_data():
    df_train = pd.read_csv(training_input_file)
    df_train['sequence'] = df_train['sequence'].apply(lambda x: [int(e) for e in x.split(':')])
    df_test = pd.read_csv(testing_input_file)
    df_test['sequence'] = df_test['sequence'].apply(lambda x: [int(e) for e in x.split(':')])
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
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_size, input_length=max_sentence_length))
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
    class_weight =  { 0: 1.0, 1: 50.0}
    model.fit(X_train, Y_train, epochs=50, batch_size=64, verbose=1, class_weight=class_weight)
    
    print("Evaluating model ... ")
    #score, acc = model.evaluate(X_test, Y_test, batch_size=1)
    Y_pred = model.predict(X_test)
    Y_pred = (Y_pred > 0.5)
    cm = confusion_matrix(Y_test, Y_pred)
    print('Confusion matrix:')
    print(cm)
    #print("Test scure: " + str(score))
    #print("Test accuracy: " + str(acc))


if __name__ == "__main__":
    #train()
    set_param('Sentence')
    train()


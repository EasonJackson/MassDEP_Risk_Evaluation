import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing import sequence
import pandas as pd

# fix random seed for reproducibility
numpy.random.seed(7)

# Data file names
training_input_file = ""
training_label_file = ""
testing_input_file = ""
testing_label_file = ""

# Input data padding
max_sentence_length = 50

# Model parameters
vocab_size = 2000
embedding_size = 128

def load_data():
    X_train = pd.read_csv(training_input_file)
    Y_train = pd.read_csv(training_label_file)
    X_test = pd.read_csv(testing_input_file)
    Y_test = pd.read_csv(testing_label_file)

    # truncate and pad input sequences
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    return X_train, Y_train, X_test, Y_test


def build_model()
    print("Creating model ... ")
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, outout_dim=embedding_size, input_length=max_sentence_length))
    model.add(LSTM(output_dim=100, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=100, activation='sigmoid', inner_activation='hard_sigmoid'))
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
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=64, verbose=1)
    
    print("Evaluating model ... ")
    score, acc = model.evaluate(X_test, Y_test, batch_size=1)
    print("Test scure: " + str(score))
    print("Test accuracy: " + str(acc))
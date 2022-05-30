##############################
# train.py
#

import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, LSTM, SimpleRNN
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


from plot import draw_plot


######################################
# some config values 
#
maxlen = 2000 #max number of words in a question to use
#max_features = 43142 #how many unique words to use(i.e num rows in embedding vector)
max_features = 50000 #how many unique words to use(i.e num rows in embedding vector)
embed_size = 100 #how big is each word vector


def model_save_to_file(model: Model, prefix:str = "normal"):
    import datetime
    now = datetime.datetime.now()
    day_date=f'{now.year}{now.month:02d}{now.day}'
    time_date=f'{now.hour:02d}{now.minute}'
    file_name=f'{prefix}_{day_date}_{time_date}.model'
    model.save(file_name)


def train_rnn(specific_model,
              x_train ,x_test,
              y_train, y_test,
              is_model_save: bool = False,
              is_simple_model: bool = True,
              epochs: int = 10,
              batch_size=512) -> Model:
 
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train)

    x_train_encoded = tokenizer.texts_to_sequences(x_train)
    x_train_features = pad_sequences(x_train_encoded,
                                     maxlen=maxlen)

    word_to_index = tokenizer.word_index

    vocab_size = len(word_to_index) + 1
    print('단어 집합의 크기: {}'.format((vocab_size)))

    embedding_dim = 32

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(specific_model)

    optimizer = 'adam'

    if is_simple_model:
        model.add(Dense(1, activation='sigmoid'))
        optimizer = 'rmsprop'
    else:
        model.add(GlobalMaxPool1D())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    x_test_encoded = tokenizer.texts_to_sequences(x_test)
    x_test_features = pad_sequences(x_test_encoded,
                                    maxlen=maxlen)


    model.layers[1].trainable=False
    history = model.fit(x_train_features,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs, 
                        validation_split=0.2)

    if is_model_save:
        model_save_to_file(model)

    key_dict = history.history.keys()
    print(f'history = {history}')
    print(f'history = {history.history}')

    draw_plot(history)

    return model, x_test_features

def train_simplernn(x_train ,x_test,
                   y_train, y_test,
                   is_model_save: bool = False,
                   epochs: int = 10,
                   batch_size=512) -> Model:
    hidden_units = 32
    return train_rnn(SimpleRNN(hidden_units),
                     x_train, x_test,
                     y_train, y_test,
                     is_model_save,
                     True,
                     epochs,
                     batch_size)

def train_lstm(x_train ,x_test,
               y_train, y_test,
               is_model_save: bool = False,
               epochs: int = 10,
               batch_size=512) -> Model:
    return train_rnn(Bidirectional(LSTM(64, return_sequences=True)),
                     x_train, x_test,
                     y_train, y_test,
                     is_model_save,
                     False,
                     epochs,
                     batch_size)


def train_classifier(x_test_features, 
                     x_train_features, 
                     y_test, y_train):

    clf = GaussianNB()
    clf.fit(x_train_features.toarray(),y_train)

    clf.score(x_test_features.toarray(),y_test)
    clf.score(x_train_features.toarray(),y_train)

    return clf

def train_naive(vectorizer, 
                x_train, x_test,
                y_train, y_test):
    x_train_raw_sentences = [' '.join(o) for o in x_train]
    x_test_raw_sentences = [' '.join(o) for o in x_test]

    # Learn vocabulary and idf from training set
    vectorizer.fit(x_train_raw_sentences)

    # Transform documents to document-term matrix
    x_train_features = vectorizer.transform(x_train_raw_sentences)
    x_test_features = vectorizer.transform(x_test_raw_sentences)

    clf = train_classifier(x_test_features,
                           x_train_features,
                           y_test, y_train)

    #len = len(vectorizer.vocabulary_)
    #print(f'Vocabulary(len={len}) = {vectorizer.vocabulary_}\n\n')

    key_names = vectorizer.get_feature_names_out()
    feature_count = pd.DataFrame(x_train_features.toarray(), columns=key_names)
    print(f'feature_count=\n{feature_count}\n\n')

    return clf, x_test_features


def train_naive_with_count(x_train, x_test,
                           y_train, y_test,
                           is_model_save=False,
                           epochs = 512,
                           batch_size = 20):
    clf, x_test_features = train_naive(CountVectorizer(),
                                       x_train, x_test,
                                       y_train, y_test)
    return clf, x_test_features


def train_naive_with_tfidf(x_train, x_test,
                           y_train, y_test,
                           is_model_save=False,
                           epochs = 512,
                           batch_size = 20):
    
    clf, x_test_features = train_naive(TfidfVectorizer(),
                                       x_train, x_test,
                                       y_train, y_test)

    return clf, x_test_features

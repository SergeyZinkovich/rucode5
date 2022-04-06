import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pickle
import keras
import tensorflow as tf


def text_classifier(vectorizer, transformer, classifier):
    return Pipeline(
        [("vectorizer", vectorizer),
         ("transformer", transformer),
         ("classifier", classifier)]
    )


def train_logreg():
    train = pd.read_csv('articles_train.csv')

    train.fillna('', inplace=True)

    texts_author = train.author
    texts_title = train.title
    texts_text = train.text
    labels = train.claps

    clf_author = text_classifier(CountVectorizer(), TfidfTransformer(), LinearRegression())
    print('-')
    clf_author.fit(texts_author, labels)
    print('-')
    clf_title = text_classifier(CountVectorizer(), TfidfTransformer(), LinearRegression())
    print('-')
    clf_title.fit(texts_title, labels)
    print('-')
    clf_text = text_classifier(CountVectorizer(), TfidfTransformer(), LinearRegression())
    print('-')
    clf_text.fit(texts_text, labels)
    print('-')

    confidence_author = clf_author.predict(train.author)
    confidence_title = clf_title.predict(train.title)
    confidence_text = clf_text.predict(train.text)

    x = list(zip(confidence_author, confidence_title, confidence_text))
    x = np.array(x)
    x = x.reshape((-1, 3))

    model = keras.Sequential()
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(12))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation('relu'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    model.fit(x, labels, batch_size=128, epochs=40)

    test = pd.read_csv('articles_test.csv')

    test.fillna('', inplace=True)

    test_confidence_author = clf_author.predict(test.author)
    test_confidence_title = clf_title.predict(test.title)
    test_confidence_text = clf_text.predict(test.text)
    x_test = list(zip(test_confidence_author, test_confidence_title, test_confidence_text))
    x_test = np.array(x_test)
    x_test = x_test.reshape((-1, 3))

    ans = model.predict(x_test)
    answer = pd.DataFrame(ans)
    answer.to_csv('ans_nlp.csv')


train_logreg()

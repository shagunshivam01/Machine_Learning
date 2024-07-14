import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import joblib

import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
label_encoder = LabelEncoder()
vectorizer = TfidfVectorizer(max_features = 5000)   # how to set max_features
model = Sequential()

def preprocess(df):
    df.drop(df.columns[[0, 1]], axis = 1, inplace = True)
    df.drop(df[df.Sentiment == 'Irrelevant'].index, axis = 0, inplace = True)
    # df["Sentiment"] = df["Sentiment"].apply(lambda x: 1 if x == 'Positive' else 0 if x == 'Neutral' else -1)
    df.dropna(inplace = True)
    df.drop_duplicates(inplace = True)
    df['Preprocessed_text'] = df['Content'].apply(preprocess_text)
    df.drop('Content', axis = 1, inplace = True)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # why use re.I|re.A
    text = text.lower()     # why not use lower before re.sub
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def train_model(df):
    x = np.array(df['Preprocessed_text'].values.astype('U'))
    y = np.array(df['Sentiment'])
    x = vectorizer.fit_transform(x)
    y = label_encoder.fit_transform(y)
    x.sort_indices()    # why is this used here

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    model.add(Dense(512, input_shape = (x_train.shape[1], ), activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation = "softmax"))
    model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])
    model.summary()

    checkpoint = ModelCheckpoint('../model/sentana.h5',monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=False)
    history = model.fit(x_train, y_train, epochs = 5, batch_size = 16, validation_data = (x_test, y_test), verbose = 1, callbacks = [checkpoint])

    scores = model.evaluate(x_test, y_test, verbose = 0)
    print(f"Test Loss : {scores[0]:.4f}")
    print(f"Test Accuracy : {scores[1]:.4f}")

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis = 1)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([preprocessed_text])
    text_tfidf.sort_indices()
    sentiment_prob = model.predict(text_tfidf)[0]
    sentiment = np.argmax(sentiment_prob)
    return label_encoder.inverse_transform([sentiment])[0]

def main():
    file_name = "../data/preprocessed_data.csv"
    try:
        df = pd.read_csv(file_name, header = 0) 
    except:
        column_names = ["Tweeet ID", "Entity", "Sentiment", "Content"]
        df_train = pd.read_csv("../data/twitter_training.csv", header = 0, names = column_names)
        df_test = pd.read_csv("../data/twitter_validation.csv", header = 0, names = column_names)
        df = df_train._append(df_test, ignore_index = True)
        preprocess(df)
        df.to_csv(file_name, index = False) 
    
    train_model(df)

    test_text = "I enjoyed the food and the service was excellent."
    predicted_sentiment = predict_sentiment(test_text)
    print(f"Predicted sentiment: {predicted_sentiment}")

    model.save('../model/sentiment_analysis_model.h5')
    joblib.dump(vectorizer, '../model/tfidf_vectorizer.pkl')
    joblib.dump(label_encoder, '../model/label_encoder.pkl')


if __name__ == "__main__":
    main()

from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk

vectorizer = joblib.load('../model/tfidf_vectorizer.pkl')
label_encoder = joblib.load('../model/label_encoder.pkl')
model = load_model('../model/sentiment_analysis_model.h5')
stop_words = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.stem.PorterStemmer()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = confidence = text = None
    if request.method == 'POST':
        text = request.form['text']
        prediction, confidence = predict_sentiment(text)
    return render_template('index.html', prediction=prediction, confidence=confidence, text=text)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # why use re.I|re.A
    text = text.lower()     # why not use lower before re.sub
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([preprocessed_text])
    text_tfidf.sort_indices()
    sentiment_prob = model.predict(text_tfidf)[0]
    sentiment = np.argmax(sentiment_prob)
    predicted_sentiment = label_encoder.inverse_transform([sentiment])[0]
    confidence = np.max(sentiment_prob)
    return predicted_sentiment, confidence

if __name__ == '__main__':
    app.run(debug=True)

#Import libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import string

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def get_model():
    '''
    Loading model for the server-side
    '''
    model = tf.keras.models.load_model('model.h5')
    print(" * Model loaded!")
    return model


@app.route('/predict', methods=['POST'])
def predict():
    '''
    A view for rendering results on HTML GUI
    '''
    # html -> py
    input_generator = request.form.values()

    #Gets values from form via POST request as a generator
    print(type(input_generator))
    
    #To get the value from a generator
    text = next(input_generator)
    print(type(text))

    #Call pre-processing function
    max_features=3000
    tokenizer=Tokenizer(num_words=max_features,split=' ')
    dump = preprocess_data(text)
    tw = tokenizer.texts_to_sequences([dump])
    tw = pad_sequences(tw,maxlen=200)

    #Predict the input text using your LSTM Model
    model = get_model()
    prediction = model.predict(tw)
    if prediction >=0.5:
        return render_template('index.html', prediction_placeholder=1)
    else:
        return render_template('index.html', prediction_placeholder=0)
    
    # py -> html
    # return render_template('index.html', prediction_placeholder=prediction)


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
    
def preprocess_data(text):
    '''Clean punctuations, remove stop_words, stemm all words. 
    Return the clean pre-processed text'''
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    more_stopwords = ['u', 'im', 'c']
    stop_words = stop_words + more_stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Stemm all the words in the sentence
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text



if __name__ == "__main__":
    app.run(debug=True) #debug=True means you won't have to run the server again & again, it'll update directly for you

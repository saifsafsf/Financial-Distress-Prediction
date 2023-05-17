#Import libraries
import numpy as np
import pandas as pd
import pickle

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

    
def preprocess_data(df):
    
    
    return df



if __name__ == "__main__":
    app.run(debug=True) #debug=True means you won't have to run the server again & again, it'll update directly for you

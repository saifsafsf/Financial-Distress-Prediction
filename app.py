#Import libraries
import pandas as pd
import pickle
from werkzeug.utils import secure_filename

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def __get_model():
    '''
    Loading model for the server-side
    '''
    file = open('model.pkl', 'rb')
    model = pickle.load(file)
    print(" * Model loaded!")
    return model

@app.route('/predict', methods=['POST'])
def predict():
    '''
    A view for rendering results on HTML GUI
    '''
    uploaded_file = request.files['uploadedCSV']
    filename = secure_filename(uploaded_file.filename)
    uploaded_file.save(filename)

    feat_imp = ['x35', 'x26', 'x83', 'x41', 'x12', 'x50', 'x75', 'x25', 'x34', 'x29', 'x65', 'x61', 'x79', 'x53', 'x23', 'x43', 'x36', 'x81', 'x14', 'x37', 'x5', 'x9', 'x60', 'x13', 'x15', 'x3', 'x22', 'x44', 'x42', 'x57', 'x10', 'x49', 'x31', 'x2', 'x16', 'x46', 'x52', 'x8', 'x38', 'x19', 'x59', 'x32', 'x47']

    header = request.form['header']

    if header == 0:
        df = pd.read_csv(filename, header=None)
        df.columns = feat_imp

    else:
        df = pd.read_csv(filename)
    
    df = df[feat_imp]

    #Predict the input text using your LSTM Model
    model = __get_model()
    prediction = model.predict(df)[0]

    if prediction == 0:
        return render_template('index.html', prediction_placeholder="Healthy")
    else:
        return render_template('index.html', prediction_placeholder="Bankrupt")


if __name__ == "__main__":
    app.run(debug=True)

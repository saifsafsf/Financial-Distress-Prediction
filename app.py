#Import libraries
import pandas as pd
import pickle
import os
from werkzeug.utils import secure_filename

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction_placeholder="Bankruptcy? Click to know!")


def __get_model():
    file = open('model/model.pkl', 'rb')
    model = pickle.load(file)
    # print(" * Model loaded!")
    return model

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['uploadedCSV']
    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join('uploads', filename)
    uploaded_file.save(filepath)

    feats = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 'x41', 'x42', 'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49', 'x50', 'x51', 'x52', 'x53', 'x54', 'x55', 'x56', 'x57', 'x58', 'x59', 'x60', 'x61', 'x62', 'x63', 'x64', 'x65', 'x66', 'x67', 'x68', 'x69', 'x70', 'x71', 'x72', 'x73', 'x74', 'x75', 'x76', 'x77', 'x78', 'x79', 'x80', 'x81', 'x82', 'x83']
    feat_imp = ['x35', 'x26', 'x83', 'x41', 'x12', 'x50', 'x75', 'x25', 'x34', 'x29', 'x65', 'x61', 'x79', 'x53', 'x23', 'x43', 'x36', 'x81', 'x14', 'x37', 'x5', 'x9', 'x60', 'x13', 'x15', 'x3', 'x22', 'x44', 'x42', 'x57', 'x10', 'x49', 'x31', 'x2', 'x16', 'x46', 'x52', 'x8', 'x38', 'x19', 'x59', 'x32', 'x47']

    header = request.form['header']

    if header == '0':
        df = pd.read_csv(filepath, header=None)
        df.columns = feats

    else:
        df = pd.read_csv(filepath)
    
    df = df[feat_imp]

    model = __get_model()
    prediction = model.predict(df)[0]

    if prediction == 0:
        return render_template('index.html', prediction_placeholder="Healthy!")
    else:
        return render_template('index.html', prediction_placeholder="Bankrupt!")


if __name__ == "__main__":
    app.run(debug=True)

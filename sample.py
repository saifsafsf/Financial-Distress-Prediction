#Import Flask Library
from flask import Flask

#app variable is defined for Flask to know where to look for templates, static files, and so on
app = Flask(__name__)
# print(__name__)

@app.route('/')
def home():
    # Return a string to display on your Flask Webpage
    return "We Rock"

if __name__ == '__main__':
    app.run(debug=True) #Debug mode is on for development mode

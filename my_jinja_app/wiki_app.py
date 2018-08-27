from collections import Counter
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import numpy as np
import pandas as pd

app = Flask(__name__,
            static_url_path='') 


@app.route('/', methods=['GET'])
def welcome_page():
    return render_template('home.html')


@app.route('/machine_learning', methods=['GET'])
def machine_learning_page():
    return render_template('index.html')

@app.route('/jinja', methods=['GET'])
def load_table():
    data = pickle.load( open( "/Users/austin/Documents/Galvanize/Capstone/Wikipedia_Knowledge_Graph/EDA/ML_data.pkl", "rb" ) )
    return render_template('index.html', data = data)

@app.route('/applied', methods=['GET'])
def load_table2():
    data = pickle.load( open( "/Users/austin/Documents/Galvanize/Capstone/Wikipedia_Knowledge_Graph/EDA/ML_pages.pkl", "rb" ) )
    data = data[data['category'] == 'Applied machine learning']
    pages = list(zip(data.page, data.Predicted_Quality))
    return render_template('applied_ml.html', pages = pages)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)



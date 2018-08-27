from collections import Counter
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import numpy as np
import pandas as pd
import urllib.parse

app = Flask(__name__,
            static_url_path='') 


@app.route('/', methods=['GET'])
def welcome_page():
    return render_template('home.html')


@app.route('/category/<category_name>', methods=['GET'])
def load_category(category_name):
    data = pickle.load( open( "/Users/austin/Documents/Galvanize/Capstone/Wikipedia_Knowledge_Graph/EDA/ML_data.pkl", "rb" ) )
    data['url_cat'] = data['category'].apply(urllib.parse.quote)
    return render_template('index.html', data=data, category_name=category_name)

@app.route('/sub_category/<sub_category_name>', methods=['GET'])
def load_sub_category(sub_category_name):
    data = pickle.load( open( "/Users/austin/Documents/Galvanize/Capstone/Wikipedia_Knowledge_Graph/EDA/ML_pages.pkl", "rb" ) )
    data = data[data['category'] == sub_category_name]
    pages = list(zip(data.page, data.Predicted_Quality))
    return render_template('sub_category.html', pages=pages, sub_category_name=sub_category_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)



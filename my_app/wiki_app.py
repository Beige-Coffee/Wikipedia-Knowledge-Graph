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
    return render_template('index.html')


@app.route('/category/<category_name>', methods=['GET'])
def load_category(category_name):
    data = pickle.load( open( "../web_data/ML_data.pkl", "rb" ) )
    data['url_cat'] = data['category'].apply(urllib.parse.quote)
    pages = list(zip(data['category'], data['Predicted_Quality']))
    return render_template('category.html', data=data, pages=pages, category_name=category_name)

@app.route('/sub_category/<sub_category_name>', methods=['GET'])
def load_sub_category(sub_category_name):
    data = pickle.load( open( "../web_data/ML_pages.pkl", "rb" ) )
    data = data[data['category'] == sub_category_name]
    data['url_cat'] = data['page'].apply(urllib.parse.quote)
    return render_template('sub_category.html', data=data, sub_category_name=sub_category_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)



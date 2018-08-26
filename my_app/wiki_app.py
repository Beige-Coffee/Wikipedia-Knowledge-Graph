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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)



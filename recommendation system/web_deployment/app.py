from flask import Flask, render_template, request
from TMDP import get_recommendations
import pandas as pd
from flask import Flask, abort
import numpy as np
import csv



app = Flask(__name__ , template_folder ='templates')



@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():

    data1 = request.form['a']
    x = pd.DataFrame(get_recommendations(data1))
    return render_template("after.html", name=x, data=x.to_html())

if __name__ == "__main__":
    app.run(debug=True)




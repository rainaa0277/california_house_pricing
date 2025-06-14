import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

#load the model

lr_model = pickle.load(open("lr_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print('data',data)
    print('list_of_data',list(data.values()))
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = lr_model.predict(new_data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    print('data',data)
    input_data = scaler.transform(np.array(data).reshape(1,-1))
    print('input_data',input_data)
    output = lr_model.predict(input_data)
    return render_template('home.html', prediction_text='The predicted value is {}'.format(output[0]))

if __name__ == "__main__":
    app.run(debug=True)

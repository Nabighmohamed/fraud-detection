import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd

import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    render_template('index.html', prediction_text='Employee Salary should be =====')
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    print(int_features,'\n')

    final_features = [np.array(int_features)]
    print(final_features, '\n')

    columns = ['step', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest',
               'newbalanceDest', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

    df2 = pd.DataFrame(final_features,
                       columns=columns)

    prediction = model.predict(df2)

    print(prediction, '\n')
    output = '---'


    if int(prediction[0])==0:
        output='not a fraud'
    elif int(prediction[0])==1:
        output='it s a fraud'
    else:
        output = '---'

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
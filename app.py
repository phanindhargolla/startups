from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            newYork = float(request.form['NewYork'])
            California = float(request.form['California'])
            Florida = float(request.form['Florida'])
            RnDSpend = float(request.form['RnDSpend'])
            Administration = float(request.form['Administration'])
            Market = float(request.form['Market'])

            data = [RnDSpend, Administration, Market, newYork, California, Florida]
            data_arr = np.array(data)
            data_arr = data_arr.reshape(1, -1)
            regression = pickle.load(open('multiple_linear_model.pkl', 'rb'))
            pred = regression.predict(data_arr)
            pred = round(float(pred), 2)

        except ValueError:
            return "Please check if all the values are entered correctly"
    return render_template('predict.html', prediction = pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

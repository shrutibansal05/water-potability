import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model

@app.route('/') # Homepage
def home():
    return render_template('start.html')

@app.route('/information',methods=['POST'])
def information():
    return render_template('info.html')

@app.route('/predict',methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction
    print(prediction)
    if(prediction==0):
        pred="Water not potable"
    else:
        pred="Water potable"
    return render_template('info.html', prediction_text='Prediction: {}'.format(pred)) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)

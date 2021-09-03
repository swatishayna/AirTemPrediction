from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = pickle.load(open('air_prediction.pkl', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        type = request.form["product_type"]
        if type =='Low':
            type = 0
        elif type =='Medium':
            type = 1
        else:
            type = 2
        Process_Temperature = float(request.form['Process_temp'])
        Process_Temperature = np.log1p(Process_Temperature)   # Normal Transformation

        Rotational_speed = float(request.form['rot_speed'])
        Rotational_speed = np.log1p(Rotational_speed)     # Normal Transformation

        tool_wear = float(request.form['tool_wear'])
        tool_wear = np.log1p(tool_wear)      # Normal Transformation

        Machine_Status = request.form['Machine_Type']
        if Machine_Status == "Fault":
            Machine_Status = 1
        else:
            Machine_Status = 0


        # Feature Scaling

        scaler = MinMaxScaler(feature_range=(0, 1))
        print("done")
        feature_scaled = scaler.fit_transform(np.array([[type,Process_Temperature,Rotational_speed,tool_wear,Machine_Status]]))
        output = model.predict(feature_scaled)
        print(output)
        return render_template('index.html', prediction_text=" Air Temperature is  {}".format(output))













if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
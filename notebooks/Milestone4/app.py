from flask import Flask, request, jsonify
import joblib
import numpy as np
import sklearn
import pandas as pd
## Import any other packages that are needed

app = Flask(__name__)

# 1. Load your model here
model = joblib.load("model.joblib")

# 2. Define a prediction function
def return_prediction(data):

    # format input_data here so that you can pass it to model.predict()
    df = pd.DataFrame(data).T

    return model.predict(df).tolist()

# 3. Set up home page using basic html
@app.route("/")
def index():
    # feel free to customize this if you like
    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 25 climate model outputs.
    """

# 4. define a new route which will accept POST requests and return model predictions
@app.route('/predict', methods=['POST'])
def rainfall_prediction():
    content = request.json  # this extracts the JSON content we sent
    prediction = return_prediction(content["data"])
    results = {
        "prediction": prediction
        "input": content["data"]
    }  # return whatever data you wish, it can be just the prediction
                     # or it can be the prediction plus the input data, it's up to you
    return jsonify(results)

import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)


model = pickle.load(open("dt_clf!!.sav", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(i) for i in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction == 0:
        prediction_val = "Non-Diabetes"
    else:
        prediction_val = "Diabetic"
    return render_template("index.html", prediction_text=f"{prediction_val}")


if __name__ == "__main__":
    app.run(debug=True)

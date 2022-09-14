import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# index_file = r"E:\Projects & Tutorial\CNN Project\Hyperparameter Tuning a Neural Network\templates\index.html"

# model_path = r"E:\Projects & Tutorial\CNN Project\Hyperparameter Tuning a Neural Network\model file\hyper_parameter tuning"
# list_model = os.listdir(model_path)
# for i in list_model:
#     print(i)
#     if i == "dt_clf!!.sav":
#         load_model = pickle.load(open(os.path.join(model_path, i), "rb"))
#         # result_ = load_model.score(x_test, y_test)
#         # result_ = (np.around(result_,2))*100
#         # print(f"{int(result_)}%")
#     else:
#         load_model = pickle.load(open(os.path.join(model_path, i), "rb"))
# result = (np.round(accuracy_score(y_test, load_model), 2))
# result = (np.round(result,2))*100
# result= int(result)
# print(f"{result}%")

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

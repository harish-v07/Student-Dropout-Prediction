from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("dropout_model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("dropout.html")

@app.route("/predict", methods=["POST"])
def predict():
   
    features = np.array([float(request.form.get(feature)) for feature in request.form])

   
    dropout_probability = round(model.predict_proba(features.reshape(1, -1))[0][1] * 100, 2)

    return render_template('dropout.html', dropout_probability=f"The probability that the individual will dropout is: {dropout_probability}%")


if __name__ == "__main__":
    app.run(debug=True)

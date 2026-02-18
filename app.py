from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model, scaler, and feature order
model = pickle.load(open("calories.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_cols = pickle.load(open("feature_selection.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Collect input values
            age = float(request.form["age"])
            gender = int(request.form["gender"])   # Male=1, Female=0
            height = float(request.form["height"])
            weight = float(request.form["weight"])
            duration = float(request.form["duration"])
            heart_rate = float(request.form["heart_rate"])
            body_temp = float(request.form["body_temp"])

            # Arrange input in SAME order as training
            input_data = np.array([[
                age,
                height,
                weight,
                duration,
                heart_rate,
                body_temp,
                gender
            ]])

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = round(model.predict(input_scaled)[0], 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
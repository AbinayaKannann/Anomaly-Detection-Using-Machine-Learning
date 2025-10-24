from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the fused model
fused_model = joblib.load("fused_model.pkl")
scaler = fused_model['scaler']
rnn_model = fused_model['rnn_model']
knn = fused_model['knn']
adaboost = fused_model['adaboost']
catboost = fused_model['catboost']
label_encoder = fused_model['label_encoder']

# Prediction function
def fused_predict(X):
    rnn_pred = rnn_model.predict(np.expand_dims(X, axis=-1))
    knn_pred = knn.predict_proba(X)
    adaboost_pred = adaboost.predict_proba(X)
    catboost_pred = catboost.predict_proba(X)
    final_pred = (0.25 * rnn_pred + 0.25 * knn_pred + 0.25 * adaboost_pred + 0.25 * catboost_pred)
    return np.argmax(final_pred, axis=1)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Handle file upload
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        # Read the test file
        test_data = pd.read_csv(file)

        # Preprocess test data
        categorical_cols = ['protocol_type', 'service', 'flag']
        test_data_encoded = pd.get_dummies(test_data, columns=categorical_cols)
        missing_cols = set(fused_model['scaler'].feature_names_in_) - set(test_data_encoded.columns)
        for col in missing_cols:
            test_data_encoded[col] = 0
        test_data_encoded = test_data_encoded[fused_model['scaler'].feature_names_in_]
        X_test_scaled = scaler.transform(test_data_encoded)

        # Make predictions
        y_pred = fused_predict(X_test_scaled)
        decoded_predictions = label_encoder.inverse_transform(y_pred)

        # Add predictions to the test data
        test_data['Predicted Attack Type'] = decoded_predictions
        results_html = test_data[['Predicted Attack Type']].to_html(classes="table table-bordered", index=False)

        return render_template("result.html", tables=results_html)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

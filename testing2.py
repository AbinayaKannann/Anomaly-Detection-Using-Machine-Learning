import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler



# Load the fused model
fused_model = joblib.load("fused_model.pkl")
scaler = fused_model['scaler']
rnn_model = fused_model['rnn_model']
knn = fused_model['knn']
adaboost = fused_model['adaboost']
catboost = fused_model['catboost']
label_encoder = fused_model['label_encoder']

# Load the test data
test_data = pd.read_csv("test.csv")  # Replace with your test CSV file

# Preprocess the test data (similar to training)
categorical_cols = ['protocol_type', 'service', 'flag']

# Apply one-hot encoding to categorical columns
test_data_encoded = pd.get_dummies(test_data, columns=categorical_cols)

# Align test data with the training data columns
missing_cols = set(fused_model['scaler'].feature_names_in_) - set(test_data_encoded.columns)
for col in missing_cols:
    test_data_encoded[col] = 0
test_data_encoded = test_data_encoded[fused_model['scaler'].feature_names_in_]

# Standardize test data
X_test_scaled = scaler.transform(test_data_encoded)

# Prediction function
def fused_predict(X):
    # RNN predictions (softmax output for multi-class)
    rnn_pred = rnn_model.predict(np.expand_dims(X, axis=-1))
    
    # KNN, AdaBoost, and CatBoost predictions (probabilities for multi-class)
    knn_pred = knn.predict_proba(X)
    adaboost_pred = adaboost.predict_proba(X)
    catboost_pred = catboost.predict_proba(X)

    # Weighted averaging of probabilities
    final_pred = (0.25 * rnn_pred + 0.25 * knn_pred + 0.25 * adaboost_pred + 0.25 * catboost_pred)
    return np.argmax(final_pred, axis=1)  # Choose the class with the highest probability

# Make predictions
y_pred = fused_predict(X_test_scaled)

# Decode predicted attack types
decoded_predictions = label_encoder.inverse_transform(y_pred)
# Add predictions to the test data
test_data['Predicted Attack Type'] = decoded_predictions

# Save the results
test_data.to_csv("test_results_with_predictions.csv", index=False)
print("Predictions saved to 'test_results_with_predictions.csv'.")

# Display the first few rows of the result
print(test_data[['Predicted Attack Type']].head())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load dataset
data = pd.read_csv("data.csv")

# Identify categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']

# Apply one-hot encoding to categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_cols)

# Encode 'Attack Type' target labels
label_encoder = LabelEncoder()
data_encoded['Attack Type'] = label_encoder.fit_transform(data['Attack Type'])

# Separate features and target
X = data_encoded.drop(['target', 'Attack Type'], axis=1)  # Features
y = data_encoded['Attack Type']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode the target for RNN
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# RNN Model
num_classes = y_train_onehot.shape[1]
rnn_model = Sequential()
rnn_model.add(LSTM(64, input_shape=(X_train_scaled.shape[1], 1), return_sequences=True))
rnn_model.add(LSTM(32))
rnn_model.add(Dense(num_classes, activation='softmax'))
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape data for RNN
X_train_rnn = np.expand_dims(X_train_scaled, axis=-1)
X_test_rnn = np.expand_dims(X_test_scaled, axis=-1)

# Train RNN
rnn_model.fit(X_train_rnn, y_train, epochs=5, batch_size=32, verbose=1)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=450)
knn.fit(X_train_scaled, y_train)

# AdaBoost Model
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train_scaled, y_train)

# CatBoost Model
catboost = CatBoostClassifier(iterations=50, learning_rate=0.1, depth=6, verbose=0)
catboost.fit(X_train_scaled, y_train)

# Fuse Models (Weighted Average Voting)
def fused_predict(X):
    rnn_pred = rnn_model.predict(np.expand_dims(X, axis=-1))  # Softmax probabilities
    knn_pred = knn.predict_proba(X)  # Probabilities
    adaboost_pred = adaboost.predict_proba(X)  # Probabilities
    catboost_pred = catboost.predict_proba(X)  # Probabilities

    # Weighted averaging of probabilities
    final_pred = (
        0.25 * rnn_pred + 0.20 * knn_pred + 0.20 * adaboost_pred + 0.20 * catboost_pred
    )
    return np.argmax(final_pred, axis=1)

# Test Fused Model
y_pred = fused_predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Fused Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:")
print(class_report)

# Save the Fused Model
fused_model = {
    'scaler': scaler,
    'rnn_model': rnn_model,
    'knn': knn,
    'adaboost': adaboost,
    'catboost': catboost,
    'label_encoder': label_encoder
}
joblib.dump(fused_model, "fused_model.pkl")
print("Fused model saved as 'fused_model.pkl'.")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from preprocess import load_data, preprocess_data, prepare_features


import os

def train():
 
 BASE_DIR = os.path.dirname(os.path.dirname(__file__))
 file_path = os.path.join(BASE_DIR, "data", "heart_dataset.csv")

 df = load_data(file_path)
    # Preprocess dataset
 df = preprocess_data(df)

    # Prepare features
 X, y, scaler = prepare_features(df)
 print("Data loaded successfuly")
 y = y.values.ravel()

    # Split data
 X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=4
    )

    # Train model
 model = LogisticRegression(max_iter=1000)
 model.fit(X_train, y_train)

    # Predictions
 y_pred = model.predict(X_test)

    # Accuracy
 accuracy = accuracy_score(y_test, y_pred)
 print(f"Model Accuracy: {accuracy:.2%}")

    # Save model
    # Ensure models folder exists
 os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

 model_path = os.path.join(BASE_DIR, "models", "heart_disease_model.pkl")
 scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

 joblib.dump(model, model_path)
 joblib.dump(scaler, scaler_path)

 print("✅ Model saved successfully!")
 
if __name__ == "__main__":
    train()


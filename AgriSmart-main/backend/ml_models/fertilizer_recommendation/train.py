import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("fertilizer.csv")

# Encode categorical columns
le_soil = LabelEncoder()
le_crop = LabelEncoder()

data["Soil Type"] = le_soil.fit_transform(data["Soil Type"])
data["Crop Type"] = le_crop.fit_transform(data["Crop Type"])

# Features & target
X = data.drop("Fertilizer Name", axis=1)
y = data["Fertilizer Name"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Fertilizer Recommendation Accuracy: {accuracy * 100:.2f}%")

# Save model & encoders
joblib.dump(model, "fertilizer_model.pkl")
joblib.dump(le_soil, "soil_encoder.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")

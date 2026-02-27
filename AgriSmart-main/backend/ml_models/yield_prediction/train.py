import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

print("🔹 Loading datasets...")

# LOAD DATA
yield_df = pd.read_csv("yield.csv")
rain_df = pd.read_csv("rainfall.csv")
temp_df = pd.read_csv("temp.csv")

# CLEAN COLUMN NAMES
yield_df.columns = yield_df.columns.str.strip().str.lower()
rain_df.columns = rain_df.columns.str.strip().str.lower()
temp_df.columns = temp_df.columns.str.strip().str.lower()

#  RENAME COLUMNS 
yield_df.rename(columns={
    "area": "location",
    "item": "crop",
    "value": "yield"
}, inplace=True)

rain_df.rename(columns={
    "area": "location",
    "average_rain_fall_mm_per_year": "rainfall"
}, inplace=True)

temp_df.rename(columns={
    "country": "location",
    "avg_temp": "temperature"
}, inplace=True)

# SELECT REQUIRED COLUMNS 
yield_df = yield_df[["location", "crop", "year", "yield"]]
rain_df = rain_df[["location", "year", "rainfall"]]
temp_df = temp_df[["location", "year", "temperature"]]

# HANDLE INVALID VALUES
yield_df.replace("..", pd.NA, inplace=True)
rain_df.replace("..", pd.NA, inplace=True)
temp_df.replace("..", pd.NA, inplace=True)

yield_df["yield"] = pd.to_numeric(yield_df["yield"], errors="coerce")
rain_df["rainfall"] = pd.to_numeric(rain_df["rainfall"], errors="coerce")
temp_df["temperature"] = pd.to_numeric(temp_df["temperature"], errors="coerce")

yield_df["year"] = pd.to_numeric(yield_df["year"], errors="coerce")
rain_df["year"] = pd.to_numeric(rain_df["year"], errors="coerce")
temp_df["year"] = pd.to_numeric(temp_df["year"], errors="coerce")

# DROP MISSING 
yield_df.dropna(inplace=True)
rain_df.dropna(inplace=True)
temp_df.dropna(inplace=True)

# MERGE DATA 
print("🔹 Merging datasets...")

merged = pd.merge(
    yield_df,
    rain_df,
    on=["location", "year"],
    how="inner"
)

merged = pd.merge(
    merged,
    temp_df,
    on=["location", "year"],
    how="inner"
)

merged.dropna(inplace=True)

print("Merged dataset shape:", merged.shape)

# ENCODE CATEGORICAL
encoders = {}

for col in ["location", "crop"]:
    le = LabelEncoder()
    merged[col] = le.fit_transform(merged[col])
    encoders[col] = le

# FEATURES & TARGET
X = merged.drop("yield", axis=1)
y = merged["yield"]

# SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL 
print("🔹 Training Gradient Boosting Regressor...")

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# EVALUATION 
r2 = r2_score(y_test, model.predict(X_test))
print(f"✅ Yield Prediction R² Score: {r2:.2f}")

# SAVE MODEL 
with open("yield_model.pkl", "wb") as f:
    pickle.dump((model, encoders), f)

print("🎉 Yield prediction model saved successfully as yield_model.pkl")

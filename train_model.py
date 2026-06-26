import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/FOOD-DATA-GROUP1.csv")

df["Healthy"] = (
    (df["Sugars"] < 10) &
    (df["Protein"] > 5) &
    (df["Dietary Fiber"] > 2)
).astype(int)

features = [
    "Caloric Value",
    "Fat",
    "Carbohydrates",
    "Sugars",
    "Protein",
    "Dietary Fiber",
    "Sodium"
]

X = df[features]
y = df["Healthy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print(f"Accuracy: {accuracy:.2f}")

joblib.dump(model, "models/nutrition_model.pkl")

print("Model Saved!")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/spotify_tracks.csv")

features = ["loudness","danceability","energy","instrumentalness","valence"]
X = df[features]
y = df["popularity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)

baseline_pred = np.full_like(y_test,y_test.mean())
baseline_mae = mean_absolute_error(y_test,baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print("Linear Regression Results")
print(f"Features:   {features}")
print(f"Model MAE:  {mae:.2f}")
print(f"Model R2:   {r2:.4f}")

print("\nBaseline (Always predicting average)")
print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Baseline R2: {baseline_r2:.4f}")

print("\n Feature Weights (coefficients)")
for feature, coef in zip(features,model.coef_):
    print(f"{feature:18}: {coef:+.4f}")
print(f"Intercept:     {model.intercept_:+.4f}")


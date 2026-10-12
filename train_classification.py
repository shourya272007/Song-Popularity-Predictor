import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/spotify_tracks.csv")

POPULARITY_THRESHOLD= 50
df["it_hit"]= (df["popularity"]> POPULARITY_THRESHOLD).astype(int)

hit_rate = df["it_hit"].mean()
print(f"Total songs: {len(df)}")
print(f"Percentage of Hits(>{POPULARITY_THRESHOLD}): {hit_rate *100:.2f}%\n")

features =[
    "loudness",
    "danceability",
    "energy",
    "instrumentalness",
    "valence",
    "tempo",
    "acousticness",
]
x =df[features]
y =df["it_hit"]

X_train, X_test,y_train, y_test = train_test_split(
    x,y,test_size=0.2,random_state=42, stratify=y
)

model =LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred= model.predict(X_test)

accuracy= accuracy_score(y_test,y_pred)

majority_class= y_train.mode()[0]
baseline_pred= np.full_like(y_test, fill_value=majority_class)
baseline_acc = accuracy_score(y_test, baseline_pred)

print("LOGISTIC REGRESSION ACCURACY")
print(f"model accuracy: {accuracy *100:.2f}%")
print(f"baseline accuracy: {baseline_acc *100:.2f}%")

print("\n CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred)
print(f"True Non-Hits (correct): {cm[0][0]}")
print(f"False Hits (Wrong): {cm[0][1]}")
print(f"Missed Hits (Wrong): {cm[1][0]}")
print(f"True Hits (correct): {cm[1][1]}")

print ("\n CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=["Non-Hit","Hit"], zero_division=0))

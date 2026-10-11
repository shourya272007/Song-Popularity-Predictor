import pandas as pd

df = pd.read_csv("data/spotify_tracks.csv")

print("___First 5 Rows___")
print(df.head())

print("\n ___data info___")
df.info()

print("\n___MISSING VALUES___")
print(df.isna().sum())

print("\n___SUMMARY STATISTICS___")
print(df.describe())

print("\n___CORRELATION WITH POPULARITY___")
numeric_df= df.select_dtypes(include=["number"])
if "popularity" in numeric_df.columns:
    corr= numeric_df.corr() ["popularity"].sort_values(ascending=False)
    print(corr)
else:
    print(" Popularity column is not available.")   
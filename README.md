# Spotify Song Popularity Predictor

An empirical analysis evaluating whether Spotify audio features (e.g., danceability, loudness, acousticness) can predict track popularity using classical machine learning models.

## Overview
A common machine learning objective is predicting song popularity from acoustic metadata. This project evaluates:
1. **Continuous Regression**: Predicting exact popularity scores (0–100) using Linear Regression.
2. **Binary Classification**: Classifying whether a track is a "Hit" (popularity > 50) using Logistic Regression.
3. **Baseline Comparison & Error Analysis**: Evaluating models against naive baselines to uncover the limits of audio-only feature sets.

## Dataset
- **Source**: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset) on Kaggle
- **Size**: 114,000 tracks across 21 columns
- **Features Used**: `loudness`, `danceability`, `energy`, `instrumentalness`, `valence`, `tempo`, `acousticness`
- **Target**: `popularity` (integer range: 0–100)

## Exploratory Findings
Computing Pearson correlation between audio features and track popularity revealed near-zero linear relationships across all variables:
- `loudness`: +0.050
- `danceability`: +0.035
- `energy`: +0.001
- `valence`: -0.041
- `instrumentalness`: -0.095

This indicates upfront that acoustic metrics carry minimal direct linear signal regarding mainstream popularity.

## Model Results

### 1. Linear Regression (Continuous Popularity)
- **Model MAE**: 18.47 points
- **Naive Baseline MAE (Mean Guess)**: 18.81 points
- **Model $R^2$**: 0.0167 (~1.7% variance explained)

The regression model marginally outperformed the baseline by 0.34 points, confirming that ~98.3% of the variance in popularity stems from external factors outside audio attributes.

### 2. Logistic Regression (Hit vs. Non-Hit)
A threshold of `popularity > 50` defines a "Hit" (~24.36% of the dataset):
- **Model Accuracy**: 75.64%
- **Majority-Class Baseline Accuracy**: 75.64%
- **Recall for Hits**: 0.00 (1 true hit identified out of 5,554 in the test set)

#### Key Takeaway: The Accuracy Paradox
While a 75.64% accuracy rate appears solid at first glance, the confusion matrix reveals that the model essentially predicted the majority class ("Not a Hit") across all instances. In an imbalanced setting, accuracy alone masks zero predictive power on the positive class.

## Conclusions
Acoustic production features alone are insufficient to predict popularity. Determinants of a song's reach—such as artist clout, playlist placement, marketing expenditure, and social media traction—are omitted variables that drive mainstream success.

## Project Structure
```text
├── data/
│   └── spotify_tracks.csv   # Kaggle dataset (git-ignored)
├── explore.py               # Exploratory data analysis & correlations
├── train_regression.py      # Linear regression & baseline evaluation
├── train_classification.py  # Logistic regression & confusion matrix analysis
├── .gitignore
└── README.md
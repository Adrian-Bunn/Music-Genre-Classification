# Music-Genre-Classification
# Music Genre Classification using Support Vector and Random Forest Classification

This project investigates the task of classifying music tracks into genres using two supervised machine learning models: Support Vector Classification (SVC) and Random Forest. The classification is based on audio features extracted from the GTZAN dataset, a widely used dataset in music information retrieval research. The models are compared based on their accuracy, performance metrics, and generalizability.

## Overview

- **Dataset**: GTZAN (via Kaggle) - 1000 samples across 10 genres, each described by 60 audio features.
- **Algorithms**: Support Vector Classification (SVC), Random Forest
- **Tools Used**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Key Steps**:
  - Data cleaning and preprocessing
  - Feature selection and dimensionality reduction (PCA, Chi-squared test)
  - Model training and evaluation
  - Cross-validation and confusion matrix analysis

## Objectives

- Identify features most important for distinguishing music genres.
- Implement SVC and Random Forest classifiers for genre prediction.
- Use feature selection and PCA to reduce model complexity.
- Evaluate and compare model performance.
- Discuss real-world implications, challenges, and future improvements.

## Dataset Description

The dataset contains 1000 audio samples, evenly distributed among the following genres:

- Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock

Each sample is described by 60 features derived from audio analysis techniques such as Short-Time Fourier Transform (STFT), MFCCs, pitch content, and rhythmic structure.

Source: [GTZAN on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## Methodology

### Data Preprocessing

- Dropped irrelevant and mostly empty features (e.g., filename, `rms_var`)
- Encoded genre labels numerically for supervised learning
- Scaled features using `StandardScaler` to standardize input distributions

### Feature Selection & Dimensionality Reduction

- Applied Chi-squared scoring to rank feature importance
- Dropped low-importance features to reduce noise
- Used Principal Component Analysis (PCA) to reduce dimensionality while retaining variance

### Classification Models

#### Support Vector Classification (SVC)

- Used RBF kernel to handle non-linear relationships in high-dimensional feature space
- Applied PCA before model training
- Tuned using cross-validation

#### Random Forest

- Initially showed overfitting with near-perfect training accuracy
- Applied depth and split constraints to reduce model complexity
- Used feature importance analysis to inform feature reduction

## Results

| Model          | Training Accuracy | Test Accuracy | Mean CV Accuracy |
|----------------|-------------------|---------------|-------------------|
| SVC (with PCA) | 85.1%             | 74.0%         | 70.0%             |
| Random Forest  | 83.8%             | 71.5%         | 68.7%             |

- SVC showed better generalization with PCA-reduced features.
- Random Forest provided stronger feature interpretability but initially overfit.
- Genres like classical and country were classified more accurately than disco and rock.

## Visualizations

- Bar chart of genre distribution
- Feature importance rankings using Chi-squared scores
- PCA explained variance plot
- Confusion matrices for both models
- Correlation heatmaps to assist feature pruning

## Conclusions

This project demonstrated that classical machine learning models can perform music genre classification effectively, achieving higher accuracy than human baseline performance (~55%). Feature selection and dimensionality reduction significantly improved both models' performance and interpretability.

While more sophisticated approaches like CNNs or CRNNs have shown higher accuracy in literature, the models used here offer a solid foundation and good transparency, making them useful for feature analysis and initial prototyping in music recommendation systems.

## Future Work

- Implement deep learning models (e.g., CNNs on spectrograms)
- Expand dataset to include hybrid genres and real-world noise
- Tune hyperparameters further for better generalization
- Apply the model to real-time audio clips for music recommendation

## How to Run

1. Clone the repository:

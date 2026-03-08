# Anomaly Detection in Credit Card Transactions using Bayesian Gaussian Mixture Model

This project aims to detect fraudulent credit card transactions using a Bayesian Gaussian Mixture Model (BGMM) with Gibbs sampling. An Autoencoder model was used for dimensionality reduction. The performance of the proposed approach was compared with Isolation Forest, K-Means, Autoencoder, and Gaussian Mixture Model (GMM). Experiments were conducted on the Kaggle ULB and IEEE-CIS datasets. All analysis was performed in Jupyter Notebook using a shared utility file.

## Project Files

- utils.py: Contains all preprocessing, models, training, evaluation, and plotting functions.
- Jupyter Notebook files (.ipynb): Used to load datasets, call functions from utils.py, train models, and display results.
- Dataset files:
  - creditcard.csv (Kaggle ULB dataset)
  - train_transaction.csv (IEEE-CIS dataset)
- Serialized files (.pkl or .joblib) are used to save and reload preprocessed data or experiment results.

## Required Packages

- pandas
- numpy
- matplotlib
- scipy
- scikit-learn
- imbalanced-learn
- tensorflow
- joblib

## Loading Dataset

### I. Kaggle ULB Dataset

The dataset was obtained from the Kaggle website in CSV format under the name creditcard.csv. It was placed in the same directory as the notebook and loaded using pandas. The dataset contains transactions labeled as either fraud or normal.

### II. IEEE-CIS Dataset

The train_transaction.csv dataset was downloaded from Kaggle and loaded using pandas. The dataset consists of transaction records labeled as fraudulent or non-fraudulent.

## Data Exploration and Balancing

Class distribution was examined using a bar plot showing the percentage of fraudulent and normal transactions. The datasets were highly imbalanced. Class balancing was explored using Random Over Sampling from the imblearn library for visualization and analysis purposes.

## Data Pre-processing

Data preprocessing was implemented in the anomaly_preprocess function in utils.py. The steps include:

- Dropping unused columns and separating features and target variable.
- Stratified train-test split.
- Handling missing values using mean imputation for symmetric distributions and median imputation for skewed distributions.
- Removing duplicate samples.
- Feature scaling using StandardScaler.
- Training models only on normal (non-fraudulent) transactions.

## Dimensionality Reduction using Autoencoder

An Autoencoder model was used to reduce the dimensionality of the data. The Autoencoder was trained only on normal transactions. The encoder output was used as input to all anomaly detection models. The encoded features were returned as pandas DataFrames for both training and test sets.

## Bayesian Gaussian Mixture Model

The encoded features were used to train a Bayesian Gaussian Mixture Model implemented from scratch in utils.py. Conjugate Normal–Inverse-Wishart priors were used for the component parameters. Gibbs sampling was applied with 1000 iterations and a burn-in period of 500 iterations to estimate component means, covariances, and mixture weights. Log-likelihood and parameter norm traces were recorded for convergence monitoring.

Anomaly scores were computed as the negative log-likelihood of the mixture model.

## Baseline Models

The following models were implemented and evaluated using the same encoded features:

- Isolation Forest
- K-Means clustering
- Autoencoder-based reconstruction error
- Gaussian Mixture Model trained using the EM algorithm

Each model returns anomaly scores for both training and test sets.

## Model Evaluation

Model evaluation was performed using:

- Precision, Recall, and F1-score
- ROC-AUC and PR-AUC
- Confusion Matrix
- Bootstrap-based confidence intervals
- Cross-validation using K-Fold strategy
- DeLong test for statistical comparison between ROC-AUC scores

ROC and Precision-Recall curves were generated for each model.

## Results

Model performance was evaluated on the Kaggle ULB and IEEE-CIS datasets using ROC-AUC and PR-AUC metrics. On the Kaggle ULB dataset, the BGMM with Gibbs sampling achieved a ROC-AUC of 0.6098 and a PR-AUC of 0.0029, while some baseline models (e.g., GMM-EM and Autoencoder) obtained higher ROC-AUC values under the same evaluation protocol.

For the IEEE-CIS dataset, BGMM achieved a ROC-AUC of 0.5422 and a PR-AUC of 0.0489, showing performance comparable to other unsupervised models such as Isolation Forest and K-Means. Overall, the results indicate that BGMM provides reasonable anomaly detection performance but does not consistently outperform all baseline methods across both datasets, especially under highly imbalanced conditions.

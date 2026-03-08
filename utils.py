# ===================== imports =====================
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import numpy as np
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model

from scipy import stats
from scipy.stats import invwishart, multivariate_normal
from scipy.stats import skew
from scipy.special import logsumexp
from random import uniform

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

warnings.filterwarnings("ignore")








# ===================== CONFIG =====================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_BOOTSTRAP = 1000
SKEW_THRESHOLD = 0.5
DE_LONG_ALPHA = 0.05

# BGMM
BGMM_N_COMPONENTS = 5
BGMM_N_ITER = 1000
BGMM_N_BURNIN = 500

# GMM
GMM_N_COMPONENTS = 5
GMM_MAX_ITER = 1000

# Isolation Forest 
IF_N_ESTIMATORS = 100      


# KMeans
KMEANS_N_CLUSTERS = 7
KMEANS_MAX_ITER = 300
KMEANS_INIT = 'k-means++'

# Autoencoder DR
AUTOENCODER_ENCODING_DIM = 8
AUTOENCODER_ACTIVATION_ENC = 'tanh'
AUTOENCODER_ACTIVATION_DEC = 'sigmoid'
AUTOENCODER_LOSS = 'binary_crossentropy'
AUTOENCODER_EPOCHS = 10
AUTOENCODER_BATCH_SIZE = 32
AUTOENCODER_OPTIMIZER = "adam"

# Autoencoder
AE_HIDDEN_DIM = 32
AE_ENCODING_DIM = 8
AE_EPOCHS = 20
AE_BATCH_SIZE = 32
AE_ACTIVATION_ENC = 'tanh'
AE_ACTIVATION_DEC = 'sigmoid'
AE_LOSS = 'mean_squared_error' 







# ===================== CLASSES =====================
class BGMM:
    def __init__(self, n_components=BGMM_N_COMPONENTS, n_iter=BGMM_N_ITER, n_burnin=BGMM_N_BURNIN, random_state=RANDOM_STATE):
        self.n_components = int(n_components)
        self.n_iter = int(n_iter)
        self.n_burnin = int(n_burnin)
        self.random_state = int(random_state)
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.trace_loglik_ = []
        self.trace_mu_norm_ = []

    def _fast_logpdf(self, X, mean, cov):
        X = np.asarray(X, dtype=float)
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        d = X.shape[1]
        X0 = X - mean
        L = np.linalg.cholesky(cov)
        y = np.linalg.solve(L, X0.T)
        maha = np.sum(y * y, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        n_components = int(self.n_components)
        mean_prior = np.zeros(d)
        sigma_prior = invwishart(df=d + 1, scale=np.eye(d))
        nu_prior = d + 1
        W_prior = np.eye(d)
        z = np.zeros((n, n_components))
        sigma = np.zeros((n_components, d, d))
        mu = np.zeros((n_components, d))
        for k in range(n_components):
            sigma[k] = sigma_prior.rvs(random_state=rng)
            mu[k] = multivariate_normal(mean_prior, sigma_prior.scale / float(nu_prior)).rvs(random_state=rng)
        mu_sum = np.zeros_like(mu)
        sigma_sum = np.zeros_like(sigma)
        w_sum = np.zeros(n_components, dtype=float)
        kept = 0
        I = np.eye(d)
        for i in range(self.n_iter):
            log_probs = np.zeros((n, n_components))
            for k in range(n_components):
                try:
                    log_probs[:, k] = self._fast_logpdf(X, mu[k], sigma[k])
                except np.linalg.LinAlgError:
                    sigma[k] = sigma[k] + I * 1e-6
                    log_probs[:, k] = self._fast_logpdf(X, mu[k], sigma[k])
            z = np.exp(log_probs - np.max(log_probs, axis=1)[:, None])
            z /= (np.sum(z, axis=1)[:, None] + 1e-16)
            for k in range(n_components):
                if np.any(z[:, k] > 0):
                    mu[k] = np.average(X, axis=0, weights=z[:, k] + 1e-8)
                dev = X - mu[k]
                A = (dev.T * z[:, k]) @ dev + W_prior
                Wk = np.linalg.solve(A, I)
                nu_post = nu_prior + np.sum(z[:, k])
                sigma[k] = invwishart(df=nu_post, scale=Wk).rvs(random_state=rng)
            z_sum = np.sum(z, axis=0)
            empty = np.where(z_sum == 0)[0]
            if empty.size > 0:
                z = np.delete(z, empty, axis=1)
                mu = np.delete(mu, empty, axis=0)
                sigma = np.delete(sigma, empty, axis=0)
                w_sum = np.delete(w_sum, empty, axis=0)
                n_components -= empty.size
                if n_components <= 0:
                    break
            for k in range(n_components):
                try:
                    np.linalg.cholesky(sigma[k])
                except np.linalg.LinAlgError:
                    sigma[k] += I * 1e-6
            ll = float(np.sum(z * log_probs[:, :n_components]))
            self.trace_loglik_.append(ll)
            self.trace_mu_norm_.append(float(np.linalg.norm(mu)))
            w = np.sum(z, axis=0)
            if i >= self.n_burnin:
                mu_sum[:n_components] += mu
                sigma_sum[:n_components] += sigma
                w_sum[:n_components] += w
                kept += 1
        if n_components > 0:
            if kept > 0:
                mu_final = mu_sum[:n_components] / kept
                sigma_final = sigma_sum[:n_components] / kept
                pi_final = w_sum[:n_components] / (w_sum[:n_components].sum() + 1e-16)
            else:
                w = np.sum(z, axis=0)
                mu_final = mu
                sigma_final = sigma
                pi_final = w / (w.sum() + 1e-16)
            self.means_ = mu_final
            self.covariances_ = sigma_final
            self.weights_ = pi_final
            self.n_components = int(mu_final.shape[0])
        return self

    def score_samples(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        K = self.means_.shape[0]
        logp = np.zeros((n, K))
        for k in range(K):
            try:
                lp = self._fast_logpdf(X, self.means_[k], self.covariances_[k])
            except np.linalg.LinAlgError:
                cov = self.covariances_[k] + np.eye(self.means_.shape[1]) * 1e-6
                lp = self._fast_logpdf(X, self.means_[k], cov)
            logp[:, k] = np.log(self.weights_[k] + 1e-16) + lp
        return -logsumexp(logp, axis=1)

    def fit_predict_scores(self, X_train, X_test):
        self.fit(X_train)
        train_scores = self.score_samples(X_train)
        test_scores = self.score_samples(X_test)
        model = {
            "mu": self.means_,
            "sigma": self.covariances_,
            "pi": self.weights_,
            "trace_loglik_": self.trace_loglik_,
            "trace_mu_norm_": self.trace_mu_norm_,
            "n_iter": self.n_iter,
            "n_burnin": self.n_burnin
        }
        return train_scores, test_scores, self


    

    
    
class GaussianMixture:
    def __init__(self, n_components, tol=1e-3):
        self.n_components = int(n_components)
        self.tol = tol
        self.means_ = None
        self.stds_ = None
        self.weights_ = None
        self.n_iter_ = None

    def _log_gaussian(self, X, mean, std):
        return -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((X - mean) ** 2) / (std**2)

    def _initialize(self, X, random_state):
        X = X.reshape(-1, 1)
        labels = KMeans(n_clusters=self.n_components, n_init=1, random_state=random_state).fit(X).labels_
        resp = np.zeros((X.shape[0], self.n_components))
        resp[np.arange(X.shape[0]), labels] = 1.0
        self._m_step(X.flatten(), resp)

    def _e_step(self, X):
        log_prob = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            log_prob[:, k] = np.log(self.weights_[k] + 1e-16) + self._log_gaussian(X, self.means_[k], self.stds_[k])
        log_norm = logsumexp(log_prob, axis=1)
        resp = np.exp(log_prob - log_norm[:, None])
        return log_norm.mean(), resp

    def _m_step(self, X, resp):
        nk = resp.sum(axis=0) + 1e-16
        self.weights_ = nk / nk.sum()
        self.means_ = (resp * X[:, None]).sum(axis=0) / nk
        var = (resp * (X[:, None] - self.means_)**2).sum(axis=0) / nk
        self.stds_ = np.sqrt(var + 1e-6)

    def fit(self, X, max_iter=100, random_state=0):
        X = np.asarray(X, dtype=float).reshape(-1)
        self._initialize(X, random_state)
        prev_ll = -np.inf
        for it in range(int(max_iter)):
            ll, resp = self._e_step(X)
            self._m_step(X, resp)
            if abs(ll - prev_ll) < self.tol:
                self.n_iter_ = it + 1
                return self
            prev_ll = ll
            if it % 100 == 0:
                print("Iteration", it)            
        self.n_iter_ = int(max_iter)
        return self

    def score_samples(self, X):
        X = np.asarray(X, dtype=float).reshape(-1)
        log_prob = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            log_prob[:, k] = np.log(self.weights_[k] + 1e-16) + self._log_gaussian(X, self.means_[k], self.stds_[k])
        return logsumexp(log_prob, axis=1)

class GMM_EM:
    def __init__(self, n_components, max_iter, random_state):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.gmm_list = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.gmm_list = []

        for j in range(X.shape[1]):
            gm = GaussianMixture(n_components=self.n_components)
            gm.fit(X[:, j], max_iter=self.max_iter, random_state=self.random_state)
            self.gmm_list.append(gm)
            if j % 1 == 0:
                print("Iteration", j)

        return self

    def score_samples(self, X):
        X = np.asarray(X, dtype=float)
        ll = np.zeros(X.shape[0])

        for j, gm in enumerate(self.gmm_list):
            ll += gm.score_samples(X[:, j])

        return -ll

    def fit_predict_scores(self, X_train, X_test):
        self.fit(X_train)
        train_scores = -self.score_samples(X_train)
        test_scores = -self.score_samples(X_test)
        return train_scores, test_scores, self



    
    
    
    
    
    
    
    
    
    
    
    
# ===================== FUNCTIONS =====================



# --------------------- Pre-Train Functions ---------------------

# --------------------- Class Distribution ---------------------
def plot_class_distribution(df, target_col):

    freq_counts = df[target_col].value_counts().sort_index()
    print(freq_counts)

    class_props = freq_counts / freq_counts.sum()

    ax = class_props.plot(kind='bar', color=['green', 'red'])

    yticks = mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))
    ax.yaxis.set_major_formatter(yticks)
    plt.ylim(0, 1.15)

    ax.set_xticklabels([str(i) for i in class_props.index], rotation=0)  

    plt.title("Normal vs. Fraud Transactions")
    plt.xlabel("Class")
    plt.ylabel("Percentage")

    legend_elements = [
        Patch(facecolor='green', label='Normal'),
        Patch(facecolor='red', label='Fraud')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    for p in ax.patches:
        ax.annotate(f"{p.get_height() * 100:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', xytext=(0, 6), textcoords='offset points')

    plt.tight_layout()
    plt.show()

    
    
    
    
    
    
    
    
    

def detect_distribution_type(df, skew_threshold=SKEW_THRESHOLD):

    results = {}

    for col in df.select_dtypes(include="number").columns:
        col_skew = skew(df[col].dropna())

        if abs(col_skew) < skew_threshold:
            dist_type = "symmetric"
        else:
            dist_type = "skewed"

        results[col] = dist_type

    return results



# --------------------- Preprocessing ---------------------
def anomaly_preprocess( df, target_col, drop_cols=None, test_size=TEST_SIZE, random_state=RANDOM_STATE):

    if drop_cols is None:
        drop_cols = []

    # ------------------ Feature / Target split ------------------
    X = df.drop(columns=drop_cols + [target_col], errors="ignore")
    y = df[target_col]


    # ------------------ Train / Test split (STRATIFIED) ------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


    # ------------------ Imputation ------------------
    X_train_skewness = detect_distribution_type(X_train)
    
    for column_name, dist_type in X_train_skewness.items():
        if dist_type == "symmetric":
            imputer = SimpleImputer(strategy="mean")
        else:
            imputer = SimpleImputer(strategy="median")

        X_train[[column_name]] = imputer.fit_transform(X_train[[column_name]])
        X_test[[column_name]]  = imputer.transform(X_test[[column_name]])

            

    # ------------------ Cleaning ------------------
    X_train = X_train.drop_duplicates()
    y_train = y_train.loc[X_train.index]

    X_test = X_test.loc[y_test.index]


    # ------------------ Scaling ------------------
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index)


    # ------------------ normal only ------------------
    X_train_normal = X_train_scaled[y_train == 0]
    y_train_normal = y_train[y_train == 0]


    return (X_train_normal, X_test_scaled, y_train_normal.to_frame(name=target_col), y_test.to_frame(name=target_col))



# --------------------- Dimensionality Reduction ---------------------
def apply_autoencoder(X_train, X_test):
    input_dim = X_train.shape[1]
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(AUTOENCODER_ENCODING_DIM, activation=AUTOENCODER_ACTIVATION_ENC)(input_layer)
    decoded = layers.Dense(input_dim, activation=AUTOENCODER_ACTIVATION_DEC)(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    autoencoder.compile(optimizer=AUTOENCODER_OPTIMIZER, loss=AUTOENCODER_LOSS)

      
    # ------------------ Train on non-fraudulent data ------------------
    autoencoder.fit(X_train.values, X_train.values, epochs=AUTOENCODER_EPOCHS, batch_size=AUTOENCODER_BATCH_SIZE, shuffle=True, verbose=1)

    # ------------------ Encoder Model ------------------    
    encoder = Model(inputs=input_layer, outputs=encoded)

    
    
    # ------------------ Transform both train and test ------------------
    X_train_encoded = encoder.predict(X_train.values)
    X_test_encoded = encoder.predict(X_test.values)

    
    X_train_encoded = pd.DataFrame(X_train_encoded, index=X_train.index, columns=[f"AE_{i}" for i in range(X_train_encoded.shape[1])])
    X_test_encoded = pd.DataFrame(X_test_encoded, index=X_test.index, columns=[f"AE_{i}" for i in range(X_test_encoded.shape[1])])

    return X_train_encoded, X_test_encoded












# --------------------- Models Training Functions ---------------------

# ===================== 1. BGMM with Gibbs algorithm =====================
def run_bgmm_gibbs(X_train, X_test, n_components=BGMM_N_COMPONENTS, n_iter=BGMM_N_ITER, n_burnin=BGMM_N_BURNIN, random_state=RANDOM_STATE):
    
    bgmm = BGMM(n_components=n_components, n_iter=n_iter, n_burnin=n_burnin, random_state=random_state)
    train_scores, test_scores, bgmm_obj = bgmm.fit_predict_scores(X_train, X_test)
    
    return train_scores, test_scores, bgmm_obj





# ===================== 2. Isolation Forest =====================
def run_isolation_forest(X_train, X_test, n_estimators=IF_N_ESTIMATORS, random_state=RANDOM_STATE):

    model = IsolationForest(n_estimators=n_estimators,random_state=random_state)
    model.fit(X_train)

    train_scores = model.decision_function(X_train) 
    test_scores = model.decision_function(X_test)
    
    return train_scores, test_scores, model
 



# ===================== 3. K-Means =====================
def run_kmeans(X_train, X_test, n_clusters=KMEANS_N_CLUSTERS, random_state=RANDOM_STATE):

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    model = KMeans(n_clusters=n_clusters, init=KMEANS_INIT, random_state=random_state)
    model.fit(X_train)

    train_scores = -np.min(np.linalg.norm(X_train[:, None, :] - model.cluster_centers_[None, :, :], axis=2), axis=1)
    test_scores  = -np.min(np.linalg.norm(X_test[:, None, :] - model.cluster_centers_[None, :, :], axis=2), axis=1)

    return train_scores, test_scores, model





# ===================== 4. Autoencoder =====================
def run_autoencoder(X_train, X_test, encoding_dim=AE_ENCODING_DIM, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, activation_enc=AE_ACTIVATION_ENC, activation_dec=AE_ACTIVATION_DEC, random_state=RANDOM_STATE):
    
    np.random.seed(random_state)
    
    
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
     
    
    
    # Encoder
    encoder = Dense(encoding_dim, activation=AE_ACTIVATION_ENC)(input_layer)
    # Decoder
    decoder = Dense(input_dim, activation=AE_ACTIVATION_DEC)(encoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(), loss=AE_LOSS)
    
    # Train autoencoder
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test), verbose=1)
     
    
    # Scores
    train_pred = autoencoder.predict(X_train, verbose=1)
    test_pred  = autoencoder.predict(X_test, verbose=1)
    
    train_scores = np.mean(np.square(X_train - train_pred), axis=1)
    test_scores  = np.mean(np.square(X_test - test_pred), axis=1)
    
    return train_scores, test_scores, autoencoder




# ===================== 5. GMM + EM =====================
def run_gmm(X_train, X_test, n_components=GMM_N_COMPONENTS, max_iter=GMM_MAX_ITER, random_state=RANDOM_STATE):

    gmm = GMM_EM(n_components=n_components, max_iter=max_iter, random_state=random_state)
    train_scores, test_scores, gmm_model = gmm.fit_predict_scores(X_train, X_test)
    
    return train_scores, test_scores, gmm_model


























# --------------------- Post Training Functions ---------------------

# ===================== CV + CI =====================
def fit_cross_validate(X_train, y_train, X_test, y_test, model_func, dataset_name, model_name, cv=CV_FOLDS, random_state=RANDOM_STATE, **model_kwargs):

    if hasattr(X_train, "reset_index"):
        X_train = X_train.reset_index(drop=True)
    if hasattr(X_test, "reset_index"):
        X_test = X_test.reset_index(drop=True)

    y_train_vector = np.asarray(y_train).ravel()
    y_test_vector  = np.asarray(y_test).ravel()

    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    for fold, (train_idx, _) in enumerate(kfold.split(X_train), start=1):

        X_train_fold = X_train.iloc[train_idx]


        _, test_scores, _ = model_func(X_train_fold, X_test, **model_kwargs)
        test_scores = np.asarray(test_scores).ravel()

        metrics = store_anomaly_metrics(
            y_true=y_test_vector,
            scores=test_scores,
            dataset_name=dataset_name,
            model_name=f"{model_name}_Fold{fold}"
        )

        results.append({
            "fold": fold,
            "best_threshold": None,
            "best_f1_on_val": None,
            "test_scores": test_scores,
            "test_metrics": metrics
        })

    return results




# ===================== Evaluation with Threshold & CI =====================
def store_anomaly_metrics(y_true, scores, dataset_name, model_name, threshold=None, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):

    scores = np.asarray(scores).ravel()
    y_true = np.asarray(y_true).ravel()

    if threshold is None:
        threshold = np.percentile(scores, 95)

    predicted_labels = (scores >= threshold).astype(int)

    precision_value = precision_score(y_true, predicted_labels, zero_division=0)
    recall_value    = recall_score(y_true, predicted_labels, zero_division=0)
    f1_value        = f1_score(y_true, predicted_labels, zero_division=0)
    roc_auc_value   = roc_auc_score(y_true, scores)
    pr_auc_value    = average_precision_score(y_true, scores)

    confusion_matrix_value = confusion_matrix(y_true, predicted_labels)

    rng = np.random.default_rng(random_state)
    prec_b, rec_b, f1_b, roc_b, pr_b = [], [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_b = y_true[idx]
        s_b = scores[idx]
        p_b = (s_b >= threshold).astype(int)

        prec_b.append(precision_score(y_b, p_b, zero_division=0))
        rec_b.append(recall_score(y_b, p_b, zero_division=0))
        f1_b.append(f1_score(y_b, p_b, zero_division=0))

        if np.unique(y_b).size > 1:
            roc_b.append(roc_auc_score(y_b, s_b))
            pr_b.append(average_precision_score(y_b, s_b))

    def ci(x):
        return f"{np.percentile(x,2.5):.3f}-{np.percentile(x,97.5):.3f}" if len(x) else "nan-nan"

    metrics_df = pd.DataFrame([{
        "Model": model_name,
        "Data": dataset_name,
        "Precision": precision_value,
        "Precision_CI": ci(prec_b),
        "Recall": recall_value,
        "Recall_CI": ci(rec_b),
        "F1": f1_value,
        "F1_CI": ci(f1_b),
        "ROC_AUC": roc_auc_value,
        "ROC_AUC_CI": ci(roc_b),
        "PR_AUC": pr_auc_value,
        "PR_AUC_CI": ci(pr_b),
        "Confusion_Matrix": confusion_matrix_value,
        "Threshold": float(threshold)
    }])

    return metrics_df








  


# ===================== Plotting =====================
def plot_anomaly_roc_pr(y_true, scores, dataset_name, model_name, fold=None):

    scores = np.asarray(scores).ravel()
    y_true = np.asarray(y_true).ravel()
    
        
        
    if np.unique(y_true).size < 2:
        raise ValueError("y_true must contain both classes (0 and 1) to plot ROC/PR.")



    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)

    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, scores)

    title = dataset_name + " - " + model_name
    subtitle = (f" | CV Fold {fold}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], lw=1.5, linestyle='--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, lw=2, label=f'PR (AP = {pr_auc:.4f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)

    plt.suptitle(f"{title}{subtitle}", fontsize=14)
    plt.tight_layout()
    plt.show()


    
    
def delong_auc_validity_table(true_labels, data_name, model1_scores, model2_scores, model1_name, model2_name, alpha=DE_LONG_ALPHA):
    true_labels   = np.asarray(true_labels).ravel()
    model1_scores = np.asarray(model1_scores).ravel()
    model2_scores = np.asarray(model2_scores).ravel()


    def auc(scores, labels):
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        return np.mean([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])

    def v_stats(scores, labels):
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        v10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
        v01 = np.array([np.mean(pos > n) + 0.5 * np.mean(pos == n) for n in neg])
        return v10, v01

    auc1 = auc(model1_scores, true_labels)
    auc2 = auc(model2_scores, true_labels)

    v10_1, v01_1 = v_stats(model1_scores, true_labels)
    v10_2, v01_2 = v_stats(model2_scores, true_labels)

    var1 = np.var(v10_1) / len(v10_1) + np.var(v01_1) / len(v01_1)
    var2 = np.var(v10_2) / len(v10_2) + np.var(v01_2) / len(v01_2)
    cov = np.cov(v10_1, v10_2)[0,1] / len(v10_1) + np.cov(v01_1, v01_2)[0,1] / len(v01_1)

    delta_auc = auc1 - auc2
    se = np.sqrt(var1 + var2 - 2 * cov)
    z = delta_auc / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    validity = "Valid" if p_value < alpha else "Not Valid"

    return pd.DataFrame({
        "Dataset": [data_name],
        "Model_1": [model1_name],
        "Model_2": [model2_name],
        "AUC_Model_1": [auc1],
        "AUC_Model_2": [auc2],
        "Delta_AUC": [delta_auc],
        "p_value": [p_value],
        "Validity": [validity]
    })




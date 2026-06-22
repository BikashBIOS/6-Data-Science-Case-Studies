# 🏦 Credit Card Customer Segmentation — Marketing Analytics

> **Case Study:** A New York City bank wants to launch a targeted ad marketing campaign by segmenting their 9,000 credit card customers into distinct behavioural groups using unsupervised machine learning and deep learning (Autoencoders).

---

## 📋 Table of Contents

- [Business Problem](#-business-problem)
- [Dataset Overview](#-dataset-overview)
- [Feature Descriptions](#-feature-descriptions)
- [Project Architecture](#-project-architecture)
- [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Clustering with K-Means](#-clustering-with-k-means)
- [Dimensionality Reduction with PCA](#-dimensionality-reduction-with-pca)
- [Deep Learning: Autoencoder (TensorFlow/Keras)](#-deep-learning-autoencoder-tensorflowkeras)
- [Results & Cluster Profiles](#-results--cluster-profiles)
- [Model Evaluation & Metrics](#-model-evaluation--metrics)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)

---

## 💼 Business Problem

Marketing is crucial for the growth and sustainability of any business. One of the key pain points for marketers is **knowing their customers and identifying their needs**. By understanding customers deeply, marketers can launch **targeted marketing campaigns** tailored for specific needs.

**Objective:** The bank has 6 months of credit card customer data. The marketing team needs to divide their customers into **at least 3 distinctive groups** to enable precise, targeted advertising. This is a classic **unsupervised learning** problem — there are no pre-labelled segments; the algorithm must discover natural groupings in the data.

---

## 📂 Dataset Overview

| Property         | Value                                    |
|------------------|------------------------------------------|
| **Source**       | `Marketing_data.csv`                     |
| **Customers**    | ~9,000 credit card holders               |
| **Time Period**  | 6 months of transaction history          |
| **Features**     | 18 (17 after dropping `CUST_ID`)         |
| **Target**       | None (Unsupervised Learning)             |
| **Missing Data** | Yes — `MINIMUM_PAYMENTS` & `CREDIT_LIMIT` |

**Notable Observation:** One customer made a single one-off purchase of **$40,761.25** — identified as an outlier during initial exploration.

---

## 📊 Feature Descriptions

| Feature | Description |
|---|---|
| `CUST_ID` | Unique ID of the credit card holder *(dropped before modelling)* |
| `BALANCE` | Balance amount left in account to make purchases |
| `BALANCE_FREQUENCY` | Frequency of balance updates (0 = rarely, 1 = frequently) |
| `PURCHASES` | Total amount of purchases made from account |
| `ONEOFF_PURCHASES` | Maximum purchase amount done in one transaction |
| `INSTALLMENTS_PURCHASES` | Amount of purchases done via installments |
| `CASH_ADVANCE` | Cash advance amount given by user |
| `PURCHASES_FREQUENCY` | Frequency of purchases (0 = rare, 1 = frequent) |
| `ONEOFF_PURCHASES_FREQUENCY` | Frequency of one-off purchases (0–1 scale) |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Frequency of installment purchases (0–1 scale) |
| `CASH_ADVANCE_FREQUENCY` | Frequency of cash advances |
| `CASH_ADVANCE_TRX` | Number of cash advance transactions |
| `PURCHASES_TRX` | Number of purchase transactions |
| `CREDIT_LIMIT` | Credit limit on the card |
| `PAYMENTS` | Total amount of payments made by user |
| `MINIMUM_PAYMENTS` | Minimum payment amount made by user |
| `PRC_FULL_PAYMENT` | Percent of months where full payment was made |
| `TENURE` | Tenure of credit card service (in months) |

---

## 🏗️ Project Architecture

```
marketing_data.ipynb
│
├── TASK 1 — Import Libraries & Load Dataset
├── TASK 2 — Visualise & Explore Dataset (EDA)
├── TASK 3 — Data Cleaning & Preprocessing
├── TASK 4 — Find Optimal K using Elbow Method
├── TASK 5 — Apply K-Means Clustering (K=8)
├── TASK 6 — Apply PCA (2 Components)
├── TASK 7 — Cluster Profiling & Visualisation
└── TASK 8 — Apply Autoencoder (TensorFlow/Keras) + Re-cluster (K=4)
```

---

## 🧹 Data Cleaning & Preprocessing

### 1. Missing Value Detection
A heatmap (Seaborn) was used to visually inspect missing values across all features.

```python
sns.heatmap(creditcard_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
creditcard_df.isnull().sum()
```

| Column | Missing Values |
|---|---|
| `MINIMUM_PAYMENTS` | Yes |
| `CREDIT_LIMIT` | Yes |
| All others | None |

### 2. Imputation Strategy
Missing values were filled with the **column mean** — appropriate for numerical, continuous financial data:

```python
# Impute MINIMUM_PAYMENTS
creditcard_df.loc[creditcard_df['MINIMUM_PAYMENTS'].isnull(), 'MINIMUM_PAYMENTS'] = \
    creditcard_df['MINIMUM_PAYMENTS'].mean()

# Impute CREDIT_LIMIT
creditcard_df.loc[creditcard_df['CREDIT_LIMIT'].isnull(), 'CREDIT_LIMIT'] = \
    creditcard_df['CREDIT_LIMIT'].mean()
```

### 3. Duplicate Check
```python
creditcard_df.duplicated().sum()  # Result: 0 duplicates
```

### 4. Drop Non-Informative Column
The `CUST_ID` column is just a row identifier — it was dropped to prevent it from influencing distance-based clustering:

```python
creditcard_df.drop("CUST_ID", axis=1, inplace=True)
# Remaining shape: (N, 17)
```

### 5. Feature Scaling
K-Means is distance-based and extremely sensitive to feature scale. All 17 features were standardised to zero mean and unit variance:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(creditcard_df)
```

---

## 🔍 Exploratory Data Analysis (EDA)

### Distribution Plots (KDE + Histogram)
All 17 features were plotted using Seaborn's `distplot` with KDE overlays to understand the distribution shape:

```python
plt.figure(figsize=(10, 50))
for i in range(len(creditcard_df.columns)):
    plt.subplot(17, 1, i+1)
    sns.distplot(creditcard_df[creditcard_df.columns[i]],
                 kde_kws={"color": "b", "lw": 3, "label": "KDE"},
                 hist_kws={"color": "g"})
    plt.title(creditcard_df.columns[i])
plt.tight_layout()
```

**Key Findings from Distributions:**

| Feature | Insight |
|---|---|
| `BALANCE` | Mean ~$1,500; right-skewed with high-balance outliers |
| `BALANCE_FREQUENCY` | Most customers update balance frequently (~1.0) |
| `PURCHASES_FREQUENCY` | Bimodal — two distinct groups: frequent vs. infrequent buyers |
| `ONEOFF_PURCHASES_FREQUENCY` | Most users rarely make one-off purchases |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Most users rarely use installments |
| `PRC_FULL_PAYMENT` | Very few customers (~0%) pay the full balance each month |
| `CREDIT_LIMIT` | Average ~$4,500 |
| `TENURE` | Most customers have ~11 years of service |

### Pair Plot
```python
sns.pairplot(creditcard_df)
```
Revealed multi-dimensional relationships and outlier clusters, especially for `PURCHASES` vs `CASH_ADVANCE`.

### Correlation Heatmap
```python
correlations = creditcard_df.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(correlations, annot=True)
```

**Strong Correlations Identified:**
- `PURCHASES` ↔ `ONEOFF_PURCHASES` *(positive)*
- `PURCHASES` ↔ `INSTALLMENTS_PURCHASES` *(positive)*
- `PURCHASES` ↔ `PURCHASES_TRX` *(positive)*
- `PURCHASES` ↔ `CREDIT_LIMIT` *(positive)*
- `PURCHASES_FREQUENCY` ↔ `PURCHASES_INSTALLMENTS_FREQUENCY` *(strong positive)*

---

## 🔵 Clustering with K-Means

### Algorithm Overview
K-Means is an **unsupervised clustering** algorithm that groups data points by minimising the Within-Cluster Sum of Squares (WCSS), measuring Euclidean distance between each point and its cluster centroid.

**K-Means Steps:**
1. Choose number of clusters **K**
2. Randomly place K centroids
3. Assign each data point to the nearest centroid
4. Recompute centroids as the mean of assigned points
5. Repeat steps 3–4 until convergence

### Elbow Method — Choosing Optimal K
The Elbow Method evaluates WCSS (inertia) for K = 1 to 19 and looks for the inflection point:

```python
scores_1 = []
range_values = range(1, 20)

for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(creditcard_df_scaled)
    scores_1.append(kmeans.inertia_)

plt.plot(scores_1, 'bx-')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('WCSS Score')
plt.show()
```

**Result:** The elbow appears at **K = 8**, indicating 8 natural customer segments in the raw feature space.

### Fitting K-Means (K=8)

```python
kmeans = KMeans(8)
kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_

# Cluster centres (inverse-transformed to original scale)
cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=creditcard_df.columns)
cluster_centers = scaler.inverse_transform(cluster_centers)
```

**Output shape:** `cluster_centers.shape` → `(8, 17)`

Cluster labels were appended to the original dataframe for analysis:

```python
creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster': labels})], axis=1)
```

Histograms for each feature were plotted across all 8 clusters to understand segment characteristics.

---

## 📉 Dimensionality Reduction with PCA

**Principal Component Analysis (PCA)** is an unsupervised algorithm that reduces high-dimensional data to fewer components while retaining maximum variance — enabling 2D visualisation of clusters.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_df_scaled)

pca_df = pd.DataFrame(data=principal_comp, columns=['pca1', 'pca2'])
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)

plt.figure(figsize=(10, 10))
sns.scatterplot(x="pca1", y="pca2", hue="cluster", data=pca_df,
                palette=['red','green','blue','pink','yellow','gray','purple','black'])
plt.show()
```

The 2D PCA scatter plot confirmed visible separation between the 8 K-Means clusters in reduced feature space.

---

## 🤖 Deep Learning: Autoencoder (TensorFlow/Keras)

### What is an Autoencoder?
An **Autoencoder** is a type of neural network that performs **representation learning (dimensionality reduction)**. It learns to:
1. **Encode** — compress the input into a lower-dimensional latent representation (bottleneck layer)
2. **Decode** — reconstruct the original input from the compressed representation

The bottleneck layer forces the network to learn only the most essential features of the data. Autoencoders work best when correlations exist between input features — which is clearly the case in this financial dataset.

### Architecture

```
Input (17 features)
     ↓
Dense(7,  activation='relu')            ← Encoder begins
     ↓
Dense(500, activation='relu', glorot_uniform)
     ↓
Dense(500, activation='relu', glorot_uniform)
     ↓
Dense(2000, activation='relu', glorot_uniform)
     ↓
Dense(10,  activation='relu', glorot_uniform)  ← BOTTLENECK (Code Layer)
     ↓
Dense(2000, activation='relu', glorot_uniform) ← Decoder begins
     ↓
Dense(500, activation='relu', glorot_uniform)
     ↓
Dense(17,  kernel_initializer='glorot_uniform') ← Output (reconstruction)
```

**Weight initialisation:** Glorot Uniform (Xavier) — draws from a truncated normal distribution, helping avoid vanishing/exploding gradients.

### Code Implementation

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from keras.optimizers import SGD

encoding_dim = 7
input_df = Input(shape=(17,))

# Encoder
x = Dense(encoding_dim, activation='relu')(input_df)
x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(x)
encoded = Dense(10, activation='relu', kernel_initializer='glorot_uniform')(x)

# Decoder
x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(encoded)
x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
decoded = Dense(17, kernel_initializer='glorot_uniform')(x)

# Models
autoencoder = Model(input_df, decoded)   # Full autoencoder
encoder = Model(input_df, encoded)       # Encoder only (for compression)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
```

### Training

```python
autoencoder.fit(
    creditcard_df_scaled,
    creditcard_df_scaled,   # Input = Output (self-supervised)
    batch_size=128,
    epochs=25,
    verbose=1
)

# Save trained weights
autoencoder.save_weights('autoencoder.weights.h5')
```

| Parameter | Value |
|---|---|
| **Input & Target** | `creditcard_df_scaled` (same — self-supervised) |
| **Loss Function** | Mean Squared Error (MSE) |
| **Optimiser** | Adam |
| **Batch Size** | 128 |
| **Epochs** | 25 |
| **Bottleneck Dimensions** | 10 |

### Encoding the Data

```python
pred = encoder.predict(creditcard_df_scaled)
# pred.shape → (N, 10)
```

The 17-dimensional customer data was compressed to **10 latent dimensions** that capture the most meaningful patterns.

### Re-Clustering on Encoded Representations

After encoding, the Elbow Method was re-applied on the 10-dimensional latent space:

```python
scores_2 = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(pred)
    scores_2.append(kmeans.inertia_)
```

**Comparison of Elbow Curves:**
```python
plt.plot(scores_1, 'bx-', color='r')  # Raw features (K=8 optimal)
plt.plot(scores_2, 'bx-', color='g')  # Encoded features (K=4 optimal)
```

The encoded representation revealed **4 cleaner, more compact clusters** — a reduction from 8, indicating that some of the original 8 clusters were partially artefacts of high-dimensional noise.

```python
kmeans = KMeans(4)
kmeans.fit(pred)
labels = kmeans.labels_
```

### Final PCA Visualisation (Post-Autoencoder)

```python
pca = PCA(n_components=2)
prin_comp = pca.fit_transform(pred)
pca_df = pd.DataFrame(data=prin_comp, columns=['pca1', 'pca2'])
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)

plt.figure(figsize=(10, 10))
sns.scatterplot(x="pca1", y="pca2", hue="cluster", data=pca_df,
                palette=['red', 'green', 'blue', 'yellow'])
plt.show()
```

The 4 post-autoencoder clusters showed **significantly better separation** in PCA space compared to the 8 raw K-Means clusters.

---

## 📈 Results & Cluster Profiles

### K-Means on Raw Features — 8 Clusters

| Cluster | Profile | Key Traits |
|---|---|---|
| 0 | **Low Engagers** | Low balance, low purchases, low credit limit |
| 1 | **Cash Advance Users** | High cash advance, low purchases |
| 2 | **Installment Shoppers** | High installment frequency, moderate balance |
| 3 | **Active One-Off Buyers** | High one-off purchases, high credit limit |
| 4 | **Revolvers** | High balance, low full payment percentage |
| 5 | **Premium Customers** | High credit limit, high payments, good payers |
| 6 | **Infrequent Users** | Very low purchase frequency, low balance |
| 7 | **High Spenders** | Very high purchases, high credit limit, high payments |

### Autoencoder + K-Means — 4 Refined Clusters

After autoencoder dimensionality reduction, the 4 final segments are:

| Cluster | Segment Name | Description | Marketing Strategy |
|---|---|---|---|
| 0 | **Transactors** | Pay in full, low cash advance, active buyers | Reward loyalty programmes, cashback offers |
| 1 | **Revolvers** | Carry high balances, low full payment rate | Balance transfer offers, low-interest products |
| 2 | **Max Payers** | High credit limit, high payments, premium segment | Premium card upgrades, exclusive benefits |
| 3 | **Low Activity** | Rarely use the card, low engagement | Re-engagement campaigns, introductory offers |

---

## 📊 Model Evaluation & Metrics

> **Note:** This is an unsupervised learning project — there are no ground-truth labels, so traditional supervised metrics (accuracy, precision, recall, F1-score, confusion matrix) do not apply in their standard form. The quality of clustering is evaluated using the metrics below.

### 1. Inertia (Within-Cluster Sum of Squares — WCSS)

The primary K-Means optimisation objective. Lower inertia = tighter, more compact clusters.

| Model | Optimal K | Inertia at Optimal K |
|---|---|---|
| K-Means (raw features) | 8 | Identified via elbow inflection |
| K-Means (autoencoder latent space) | 4 | Lower (more compact) |

### 2. Autoencoder Reconstruction Loss (MSE)

The autoencoder was evaluated by its ability to reconstruct the original scaled data from the compressed bottleneck representation.

| Metric | Value |
|---|---|
| **Loss Function** | Mean Squared Error (MSE) |
| **Optimiser** | Adam |
| **Training Epochs** | 25 |
| **Batch Size** | 128 |

The loss decreased progressively over 25 epochs, confirming the autoencoder successfully learned a compressed representation of customer behaviour patterns.

### 3. Dimensionality Reduction Effectiveness

| Stage | Input Dimensions | Output Dimensions | Reduction |
|---|---|---|---|
| Raw → Scaled | 17 | 17 | 0% (scaling only) |
| Scaled → Autoencoder Latent | 17 | 10 | **41.2%** |
| Latent → PCA (Visualisation) | 10 | 2 | **80%** |

### 4. Cluster Quality — Visual Separation (PCA)

| Clustering Method | Clusters | Visual PCA Separation |
|---|---|---|
| K-Means on raw scaled features | 8 | Moderate — some overlap |
| K-Means on autoencoder encoding | 4 | High — clear visual separation |

### 5. Comparative Elbow Analysis

Plotting both inertia curves (raw vs. encoded) demonstrates that the autoencoder-compressed representation yields a sharper, earlier elbow at **K=4**, confirming superior cluster compactness vs. the raw-feature elbow at K=8.

---

## 🔧 Tech Stack

| Category | Library / Tool | Version |
|---|---|---|
| **Language** | Python | 3.x |
| **Data Manipulation** | Pandas, NumPy | Latest |
| **Visualisation** | Matplotlib, Seaborn | Latest |
| **Machine Learning** | Scikit-learn | Latest |
| **Clustering** | `sklearn.cluster.KMeans` | — |
| **Dimensionality Reduction** | `sklearn.decomposition.PCA` | — |
| **Scaling** | `sklearn.preprocessing.StandardScaler` | — |
| **Deep Learning** | TensorFlow / Keras | Latest |
| **Autoencoder Layers** | `Dense`, `Input`, `BatchNormalization` | — |
| **Weight Init** | `glorot_uniform` (Xavier) | — |
| **Optimiser** | Adam | — |
| **Loss** | Mean Squared Error | — |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/marketing-customer-segmentation.git
cd marketing-customer-segmentation
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras
```

### 3. Add the Dataset
Place `Marketing_data.csv` in the root directory.

### 4. Run the Notebook
```bash
jupyter notebook marketing_data.ipynb
```

### 5. Run Tasks in Order
Execute cells from **Task 1 → Task 8** sequentially. Autoencoder weights will be saved as `autoencoder.weights.h5` after Task 8 training.

---

## 📌 Key Takeaways

- **K-Means alone** on 17 raw features found 8 segments — useful but noisy.
- **Autoencoder (TensorFlow/Keras)** compressed 17 features → 10 latent dimensions, filtering out irrelevant noise and capturing the core variance.
- **Re-clustering on encoded data** revealed 4 cleaner, more actionable customer segments with better PCA visual separation.
- **PCA** served as a 2D visualisation lens throughout — not as a modelling step, but as an interpretability tool.
- The entire pipeline is **unsupervised** — no labels were used at any point, making it suitable for real-world CRM segmentation where ground truth is unavailable.

---

## 📁 File Structure

```
├── marketing_data.ipynb       # Main Jupyter Notebook
├── Marketing_data.csv         # Dataset (add manually)
├── Marketing_slides.pptx      # Conceptual reference slides
├── autoencoder.weights.h5     # Saved autoencoder weights (generated on run)
└── README.md                  # This file
```

---

*Built as a data science case study for marketing segmentation using unsupervised ML and deep learning.*
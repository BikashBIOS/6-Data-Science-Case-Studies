# 🏢 Employee Attrition Prediction — HR Analytics

> **Predicting which employees are likely to leave the company using Machine Learning and Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-green?logo=scikitlearn)](https://scikit-learn.org)
[![Keras](https://img.shields.io/badge/Keras-ANN-red?logo=keras)](https://keras.io)

---

## 📋 Table of Contents

1. [Problem Statement & Business Case](#-problem-statement--business-case)
2. [Dataset Overview](#-dataset-overview)
3. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
4. [Data Preprocessing & Cleaning](#-data-preprocessing--cleaning)
5. [Feature Engineering](#-feature-engineering)
6. [ML Algorithms & Architecture](#-ml-algorithms--architecture)
   - [Logistic Regression](#1-logistic-regression)
   - [Random Forest Classifier](#2-random-forest-classifier)
   - [Artificial Neural Network (ANN) via TensorFlow](#3-artificial-neural-network-ann-via-tensorflow)
7. [Model Training & Testing](#-model-training--testing)
8. [Model Evaluation & Results](#-model-evaluation--results)
   - [Confusion Matrices](#confusion-matrices)
   - [Classification Reports](#classification-reports)
   - [Model Comparison Summary](#model-comparison-summary)
9. [Key Findings & Insights](#-key-findings--insights)
10. [Project Structure](#-project-structure)
11. [Tech Stack](#-tech-stack)
12. [Getting Started](#-getting-started)

---

## 🎯 Problem Statement & Business Case

Employee attrition — the voluntary departure of employees — is one of the costliest challenges facing modern organizations. Replacing a single employee can cost **50–200% of their annual salary** when accounting for recruiting, onboarding, and lost productivity.

**Objective:** Build a predictive system that identifies employees at risk of leaving the organization, enabling HR departments to take proactive retention measures.

**Business Impact:**
- Reduce turnover costs by targeting at-risk employees early
- Improve workforce planning and succession management
- Enhance employee satisfaction through data-driven HR policy
- Allow management to understand which factors most contribute to attrition

Three classification approaches are implemented and compared:
- Logistic Regression (baseline)
- Random Forest Classifier (ensemble method)
- Artificial Neural Network via TensorFlow/Keras (deep learning)

---

## 📊 Dataset Overview

**Source:** IBM HR Analytics Employee Attrition & Performance Dataset (`Human_Resources.csv`)

| Property | Value |
|----------|-------|
| Total Records | **1,470 employees** |
| Total Features | **35 columns** |
| Features Used (after cleaning) | **31 columns** |
| Target Variable | `Attrition` (Binary: Yes / No) |
| Task Type | Binary Classification |

### Class Distribution

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| Stayed | 0 | 1,233 | **83.9%** |
| Left | 1 | 237 | **16.1%** |


### Feature Categories

**Categorical Features (6):**

| Feature | Description | Example Values |
|---------|-------------|----------------|
| `BusinessTravel` | Frequency of work travel | Non-Travel, Travel_Rarely, Travel_Frequently |
| `Department` | Employee's department | Sales, R&D, HR |
| `EducationField` | Field of study | Life Sciences, Medical, Marketing |
| `Gender` | Employee gender | Male, Female |
| `JobRole` | Job title/role | Sales Executive, Research Scientist |
| `MaritalStatus` | Marital status | Single, Married, Divorced |

**Numerical Features (25 key):**

| Feature | Description |
|---------|-------------|
| `Age` | Employee age |
| `DailyRate` | Daily pay rate |
| `DistanceFromHome` | Distance from home to office (miles) |
| `Education` | Education level (1–5 scale) |
| `EnvironmentSatisfaction` | Workplace satisfaction (1–4) |
| `HourlyRate` | Hourly pay rate |
| `JobInvolvement` | Job involvement level (1–4) |
| `JobLevel` | Seniority level (1–5) |
| `JobSatisfaction` | Job satisfaction rating (1–4) |
| `MonthlyIncome` | Monthly salary |
| `MonthlyRate` | Monthly rate |
| `NumCompaniesWorked` | Number of companies worked at |
| `OverTime` | Whether employee works overtime (Yes/No → 1/0) |
| `PercentSalaryHike` | Last salary increase (%) |
| `PerformanceRating` | Last performance rating |
| `RelationshipSatisfaction` | Satisfaction with work relationships |
| `StockOptionLevel` | Stock option allocation level |
| `TotalWorkingYears` | Total career experience (years) |
| `TrainingTimesLastYear` | Training sessions attended last year |
| `WorkLifeBalance` | Work-life balance rating (1–4) |
| `YearsAtCompany` | Tenure at current company |
| `YearsInCurrentRole` | Years in current role |
| `YearsSinceLastPromotion` | Years since last promotion |
| `YearsWithCurrManager` | Years under current manager |

**Dropped Features (4 — zero variance / non-informative):**

| Feature | Reason for Dropping |
|---------|---------------------|
| `EmployeeCount` | Constant value (1) for all rows |
| `StandardHours` | Constant value (80) for all rows |
| `Over18` | All employees are 'Y' — no variance |
| `EmployeeNumber` | Unique identifier — not a predictive feature |

---

## 📈 Exploratory Data Analysis (EDA)

### 1. Missing Value Analysis
A heatmap was plotted using Seaborn to inspect missing data:

```python
sns.heatmap(employee_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
```

✅ **Result:** No missing values were found across any of the 35 features.

---

### 2. Feature Distribution — Histograms
```python
employee_df.hist(bins=30, figsize=(20,20), color='r')
```

**Key Observations:**
- `MonthlyIncome` is **right-skewed** (tail-heavy) — most employees earn lower salaries with a few high earners
- `TotalWorkingYears` is also **right-skewed** — most employees have fewer years of experience
- `Age` follows a **near-normal distribution** centered around mid-30s
- `YearsAtCompany` is heavily skewed toward shorter tenures

---

### 3. Attrition Count — Stayed vs. Left

```
Total Employees         : 1,470
Employees who Left      : 237   (16.12%)
Employees who Stayed    : 1,233 (83.88%)
```

---

### 4. Correlation Matrix

```python
emp_df = employee_df.drop(['BusinessTravel','Department','EducationField',
                            'Gender','JobRole','MaritalStatus'], axis=1)
correlations = emp_df.corr()
sns.heatmap(correlations, annot=True, figsize=(20,20))
```

**Strong Positive Correlations Found:**

| Feature Pair | Correlation | Interpretation |
|--------------|-------------|----------------|
| `JobLevel` ↔ `TotalWorkingYears` | ~0.78 | More experienced → Higher job level |
| `MonthlyIncome` ↔ `JobLevel` | ~0.95 | Senior roles pay significantly more |
| `MonthlyIncome` ↔ `TotalWorkingYears` | ~0.77 | Experience directly increases salary |
| `Age` ↔ `MonthlyIncome` | ~0.50 | Older employees tend to earn more |
| `YearsAtCompany` ↔ `YearsInCurrentRole` | ~0.76 | Long tenure → More time in role |

---

### 5. Age vs. Attrition (Count Plot)

```python
sns.countplot(x='Age', hue='Attrition', data=employee_df, figsize=(25,12))
```

**Findings:** Employees in the **26–35 age range** show the highest attrition rates, while employees aged 45+ are more likely to stay.

---

### 6. Categorical Feature Analysis (Subplots)

```python
# 4 subplots: JobRole, MaritalStatus, JobInvolvement, JobLevel vs Attrition
```

**Key Findings:**

| Feature | Attrition Pattern |
|---------|------------------|
| `JobRole` | **Sales Representatives** have the highest attrition rate |
| `MaritalStatus` | **Single employees** leave more than married or divorced |
| `JobInvolvement` | **Low involvement** employees are more likely to leave |
| `JobLevel` | **Entry-level (Level 1)** employees have significantly higher attrition |

---

### 7. KDE (Kernel Density Estimate) Plots

Probability density comparisons between employees who left vs. stayed:

**Distance From Home:**
```python
sns.kdeplot(left_df['DistanceFromHome'], label='Left', shade=True, color='r')
sns.kdeplot(stayed_df['DistanceFromHome'], label='Stayed', shade=True, color='b')
```
➡ Employees who live **farther from the office** have a higher probability of leaving.

**Years With Current Manager:**
➡ Employees who left tended to have spent **fewer years with their current manager**, indicating that strong manager relationships improve retention.

**Total Working Years:**
➡ Employees with **fewer total working years** (early career stage) are more likely to leave.

---

### 8. Income Analysis

```python
sns.boxplot(x='Gender', y='MonthlyIncome', data=employee_df)
sns.boxplot(x='MonthlyIncome', y='JobRole', data=employee_df)
```

**Findings:**
- Monthly income is broadly similar between male and female employees
- **Managers** and **Research Directors** earn the highest salaries
- **Sales Representatives** and **HR** roles earn the least — consistent with high attrition in these roles

---

## 🛠 Data Preprocessing & Cleaning

### Step 1 — Encode Binary Text Columns
```python
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['OverTime']  = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['Over18']    = employee_df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
```

### Step 2 — Drop Non-Informative Columns
```python
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'],
                  axis=1, inplace=True)
```

### Step 3 — One-Hot Encode Categorical Features
```python
from sklearn.preprocessing import OneHotEncoder

x_cat = employee_df[['BusinessTravel','Department','EducationField',
                      'Gender','JobRole','MaritalStatus']]
onehotencoder = OneHotEncoder()
x_cat = onehotencoder.fit_transform(x_cat).toarray()
x_cat = pd.DataFrame(x_cat)
```

**Resulting one-hot encoded features:**

| Original Column | Categories | Encoded Columns |
|----------------|------------|-----------------|
| `BusinessTravel` | 3 | 3 |
| `Department` | 3 | 3 |
| `EducationField` | 6 | 6 |
| `Gender` | 2 | 2 |
| `JobRole` | 9 | 9 |
| `MaritalStatus` | 3 | 3 |
| **Total** | | **26 encoded features** |

### Step 4 — Concatenate Numerical & Encoded Features
```python
x_num = emp_df.drop(['Attrition'], axis=1)          # 24 numerical features
x_all = pd.concat([x_cat, x_num], axis=1)           # 26 + 24 = 50 total features
x_all.columns = x_all.columns.astype(str)
```

### Step 5 — Normalize Features (MinMaxScaler)
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x = scaler.fit_transform(x_all)                     # All values scaled to [0, 1]
```

### Step 6 — Separate Target Variable
```python
y = employee_df['Attrition']                         # Binary target: 0 = Stayed, 1 = Left
```

---

## ⚙️ Feature Engineering

| Step | Technique | Purpose |
|------|-----------|---------|
| Label Encoding | `lambda` function | Convert Yes/No text → 0/1 integers |
| One-Hot Encoding | `sklearn.OneHotEncoder` | Handle multi-class categorical variables |
| Feature Concatenation | `pd.concat` | Merge encoded + numerical into one matrix |
| Feature Scaling | `sklearn.MinMaxScaler` | Normalize all features to \[0, 1\] range |
| Feature Removal | `DataFrame.drop()` | Remove constant and ID columns |

**Final Feature Space: 50 input features → 1 binary output (Attrition)**

---

## 🤖 ML Algorithms & Architecture

### Train / Test Split

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
```

| Split | Size | Records |
|-------|------|---------|
| Training Set | 75% | ~1,102 samples |
| Test Set | 25% | ~368 samples |

---

### 1. Logistic Regression

A linear baseline classifier estimating the **probability of attrition** using a sigmoid function.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```

**Why Logistic Regression?**
- Interpretable, fast baseline
- Works well with scaled, encoded binary targets
- Provides class probability scores (useful for threshold tuning)

---

### 2. Random Forest Classifier

An ensemble of **multiple decision trees** that votes on the final prediction. Naturally handles non-linear relationships and feature interactions.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```

**Why Random Forest?**
- Resistant to overfitting via bagging (bootstrap aggregation)
- Handles feature correlations better than single trees
- Provides implicit feature importance scores

---

### 3. Artificial Neural Network (ANN) via TensorFlow

A deep learning approach using TensorFlow + Keras for **binary classification**.

#### Network Architecture

```
Input Layer : 50 features
    ↓
Dense Layer 1 : 500 neurons, ReLU activation
    ↓
Dense Layer 2 : 500 neurons, ReLU activation
    ↓
Dense Layer 3 : 500 neurons, ReLU activation
    ↓
Output Layer  : 1 neuron, Sigmoid activation
```

```python
import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(50,)))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=1,   activation='sigmoid'))
```

#### Model Summary

| Layer | Type | Units | Activation | Parameters |
|-------|------|-------|------------|------------|
| Layer 1 (Input) | Dense | 500 | ReLU | 50 × 500 + 500 = **25,500** |
| Layer 2 (Hidden) | Dense | 500 | ReLU | 500 × 500 + 500 = **250,500** |
| Layer 3 (Hidden) | Dense | 500 | ReLU | 500 × 500 + 500 = **250,500** |
| Layer 4 (Output) | Dense | 1 | Sigmoid | 500 × 1 + 1 = **501** |
| **Total** | | | | **527,001 trainable parameters** |

#### Activation Functions

**ReLU (Rectified Linear Unit)** — used in hidden layers:
```
f(x) = max(0, x)
```
- Introduces non-linearity, enabling the network to learn complex patterns
- Computationally efficient; avoids vanishing gradient problem

**Sigmoid** — used in the output layer:
```
f(x) = 1 / (1 + e^(-x))
```
- Outputs a probability in range (0, 1)
- Values > 0.5 → predicted as "Left" (1)
- Values ≤ 0.5 → predicted as "Stayed" (0)

#### Compilation & Training

```python
model.compile(
    optimizer='Adam',                  # Adaptive Moment Estimation
    loss='binary_crossentropy',        # Standard loss for binary classification
    metrics=['accuracy']
)

epochs_hist = model.fit(
    x_train, y_train,
    epochs=100,                        # 100 full passes through training data
    batch_size=50                      # Update weights every 50 samples
)
```

**Optimizer — Adam (Adaptive Moment Estimation):**
- Maintains adaptive learning rates per parameter
- Stores first and second moments of gradients
- Robust to sparse gradients; converges faster than SGD

**Loss Function — Binary Cross-Entropy:**
```
Loss = -[y·log(ŷ) + (1−y)·log(1−ŷ)]
```
Measures the divergence between predicted probabilities and true labels.

**Training Configuration:**

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Optimizer | Adam | Adaptive learning rate |
| Loss Function | Binary Cross-Entropy | Binary classification standard |
| Epochs | 100 | Full dataset passes |
| Batch Size | 50 | Mini-batch gradient descent |
| Input Shape | (50,) | 50 engineered features |
| Output Activation | Sigmoid | Probability output |
| Decision Threshold | 0.5 | `y_pred = (y_pred > 0.5)` |

---

## 📦 Model Training & Testing

### Training Progress — ANN

The model was trained for 100 epochs and monitored via loss and accuracy:

```python
# Loss Plot
plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')

# Accuracy Plot
plt.plot(epochs_hist.history['accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
```

**Training Progression:**

| Epoch Range | Loss (approx.) | Accuracy (approx.) |
|-------------|----------------|--------------------|
| 1–10 | 0.45–0.38 | 83–86% |
| 11–30 | 0.38–0.29 | 86–88% |
| 31–60 | 0.29–0.24 | 88–90% |
| 61–100 | 0.24–0.21 | 90–92% |

> Training loss **decreases** gradually; training accuracy **increases** steadily — no signs of early divergence.

---

## 📊 Model Evaluation & Results

### Confusion Matrices

A confusion matrix shows the breakdown of correct and incorrect predictions across both classes.

**Legend:**
```
TN (True Negative)  = Correctly predicted "Stayed"
FP (False Positive) = Incorrectly predicted "Left" (actually Stayed)
FN (False Negative) = Incorrectly predicted "Stayed" (actually Left)
TP (True Positive)  = Correctly predicted "Left"
```

---

#### Model 1 — Logistic Regression

```
Confusion Matrix:
                  Predicted Stayed   Predicted Left
Actual Stayed  [       299        |       10       ]
Actual Left    [        35        |       24       ]
```

| Metric | Value |
|--------|-------|
| True Negatives (TN) | 299 |
| False Positives (FP) | 10 |
| False Negatives (FN) | 35 |
| True Positives (TP) | 24 |
| **Accuracy** | **88.3%** |

---

#### Model 2 — Random Forest Classifier

```
Confusion Matrix:
                  Predicted Stayed   Predicted Left
Actual Stayed  [       299        |       10       ]
Actual Left    [        39        |       20       ]
```

| Metric | Value |
|--------|-------|
| True Negatives (TN) | 299 |
| False Positives (FP) | 10 |
| False Negatives (FN) | 39 |
| True Positives (TP) | 20 |
| **Accuracy** | **86.7%** |

---

#### Model 3 — ANN (Deep Learning / TensorFlow)

```
Confusion Matrix:
                  Predicted Stayed   Predicted Left
Actual Stayed  [       298        |       11       ]
Actual Left    [        37        |       22       ]
```

| Metric | Value |
|--------|-------|
| True Negatives (TN) | 298 |
| False Positives (FP) | 11 |
| False Negatives (FN) | 37 |
| True Positives (TP) | 22 |
| **Accuracy** | **87.0%** |

---

### Classification Reports

#### Model 1 — Logistic Regression

```
              precision    recall  f1-score   support

           0       0.90      0.97      0.93       309
           1       0.71      0.41      0.52        59

    accuracy                           0.88       368
   macro avg       0.80      0.69      0.73       368
weighted avg       0.87      0.88      0.87       368
```

---

#### Model 2 — Random Forest Classifier

```
              precision    recall  f1-score   support

           0       0.88      0.97      0.92       309
           1       0.67      0.34      0.45        59

    accuracy                           0.87       368
   macro avg       0.77      0.65      0.69       368
weighted avg       0.85      0.87      0.85       368
```

---

#### Model 3 — ANN (TensorFlow / Keras)

```
              precision    recall  f1-score   support

           0       0.89      0.96      0.92       309
           1       0.67      0.37      0.48        59

    accuracy                           0.87       368
   macro avg       0.78      0.67      0.70       368
weighted avg       0.85      0.87      0.86       368
```

---

### Model Comparison Summary

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Macro F1 |
|-------|----------|---------------------|------------------|--------------------|----------|
| **Logistic Regression** | **88.3%** | 0.71 | 0.41 | 0.52 | 0.73 |
| Random Forest | 86.7% | 0.67 | 0.34 | 0.45 | 0.69 |
| ANN (TensorFlow) | 87.0% | 0.67 | 0.37 | 0.48 | 0.70 |

**Metric Definitions:**

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Accuracy** | (TP + TN) / Total | Overall correct predictions |
| **Precision** | TP / (TP + FP) | Of all "Left" predictions, how many were correct |
| **Recall** | TP / (TP + FN) | Of all employees who actually left, how many were caught |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision and Recall |

> ⚠️ **Note on Class Imbalance:** All models show lower recall for Class 1 (Left) due to the 84:16 class imbalance. The SMOTE oversampling approach (commented in the notebook) could be applied to improve minority class recall. Logistic Regression achieves the best overall accuracy and F1-score in this setup.

---

## 💡 Key Findings & Insights

### Factors Most Associated with Employee Attrition

| Factor | Finding |
|--------|---------|
| **Job Role** | Sales Representatives have the highest attrition rate |
| **Marital Status** | Single employees are most likely to leave |
| **Job Level** | Entry-level (Level 1) employees leave most frequently |
| **Job Involvement** | Low-involvement employees are at highest risk |
| **Overtime** | Employees working overtime have higher attrition |
| **Distance From Home** | Longer commutes increase attrition probability |
| **Years with Manager** | Short tenure under current manager correlates with leaving |
| **Total Working Years** | Less experienced employees leave more |
| **Stock Options** | Employees with no stock options are more likely to leave |
| **Age** | Younger employees (26–35) have the highest attrition risk |

### Statistical Comparisons (Left vs. Stayed)

| Feature | Left (mean) | Stayed (mean) | Interpretation |
|---------|-------------|---------------|----------------|
| `Age` | ~33 | ~37 | Younger employees leave more |
| `DailyRate` | ~750 | ~820 | Higher pay → higher retention |
| `DistanceFromHome` | ~10 | ~9 | Farther distance → more attrition |
| `EnvironmentSatisfaction` | ~2.5 | ~2.8 | Low satisfaction → risk of leaving |
| `JobSatisfaction` | ~2.5 | ~2.8 | Low satisfaction → risk of leaving |
| `StockOptionLevel` | ~0.5 | ~0.9 | Higher options → better retention |

### Income Analysis

- **Manager** and **Research Director** roles have the highest median income
- **Sales Representatives** and **HR** have the lowest median income — directly correlating with the highest attrition
- **Gender income parity** is broadly observed across the dataset

---

## 📁 Project Structure

```
hr-attrition-prediction/
│
├── 📓 1_Human_Resources_Department.ipynb    # Main Jupyter notebook
├── 📄 Human_Resources.csv                   # Dataset (IBM HR Analytics)
├── 📋 README.md                             # This file
│
└── outputs/
    ├── 📊 correlation_heatmap.png
    ├── 📊 attrition_age_countplot.png
    ├── 📊 kde_distance_from_home.png
    ├── 📊 kde_years_with_manager.png
    ├── 📊 kde_total_working_years.png
    ├── 📊 confusion_matrix_logreg.png
    ├── 📊 confusion_matrix_rf.png
    ├── 📊 confusion_matrix_ann.png
    ├── 📊 ann_training_loss.png
    └── 📊 ann_training_accuracy.png
```

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥1.3 | Data loading, manipulation, EDA |
| `numpy` | ≥1.21 | Numerical computation |
| `matplotlib` | ≥3.4 | Static visualizations, subplots |
| `seaborn` | ≥0.11 | Statistical plots, heatmaps, KDE |
| `scikit-learn` | ≥0.24 | Preprocessing, LR, RF, metrics |
| `tensorflow` | ≥2.6 | Deep learning (ANN via Keras) |
| `keras` | (via TF) | Sequential model, Dense layers |

**Preprocessing Tools (scikit-learn):**
- `OneHotEncoder` — Categorical feature encoding
- `MinMaxScaler` — Feature normalization
- `train_test_split` — Data splitting

**Evaluation Tools (scikit-learn):**
- `confusion_matrix` — Visual performance breakdown
- `classification_report` — Precision / Recall / F1
- `accuracy_score` — Overall accuracy

---

## 🚀 Getting Started

### Prerequisites

```bash
python --version    # Python 3.8+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/hr-attrition-prediction.git
cd hr-attrition-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Running in Google Colab

1. Upload `Human_Resources.csv` to your Google Drive under `My Drive/Colab Notebooks/1. Human Resources Data/`
2. Open `1_Human_Resources_Department.ipynb` in Google Colab
3. Mount Google Drive when prompted
4. Run all cells sequentially (`Runtime → Run all`)

### Running Locally

```bash
# Launch Jupyter
jupyter notebook 1_Human_Resources_Department.ipynb
```

Update the dataset path in Task #2:
```python
employee_df = pd.read_csv('Human_Resources.csv')   # local path
```

---

## 📌 Future Improvements

- **Handle Class Imbalance** — Apply SMOTE (Synthetic Minority Oversampling Technique) to improve minority class recall
- **Hyperparameter Tuning** — GridSearchCV for Logistic Regression and Random Forest
- **Feature Importance** — Extract and visualize Random Forest feature importances
- **Cross-Validation** — K-Fold CV for more robust performance estimates
- **Regularization in ANN** — Add Dropout layers to reduce overfitting
- **Threshold Tuning** — Adjust classification threshold (e.g., 0.3) to prioritize recall for at-risk employees
- **SHAP Values** — Model explainability for HR stakeholder communication
- **Deployment** — Wrap model in Flask/FastAPI REST API for HR system integration

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Deep Learning Framework: [TensorFlow / Keras](https://www.tensorflow.org/)
- ML Library: [scikit-learn](https://scikit-learn.org/)

---

*Built with ❤️ using Python, TensorFlow, and scikit-learn*
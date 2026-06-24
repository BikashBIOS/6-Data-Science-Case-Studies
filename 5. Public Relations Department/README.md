# 🎙️ Amazon Alexa Sentiment Analysis — Public Relations NLP Project

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn) ![NLTK](https://img.shields.io/badge/NLTK-NLP-green) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-lightblue?logo=pandas) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Business Context

> **Scenario:** You work as a Data Scientist at a multinational corporation. The **Public Relations department** has collected extensive customer review data for their Amazon Alexa products. Based on these reviews (in text format), the team needs to **automatically predict whether customers are satisfied or not** — eliminating the need to manually read through thousands of reviews.

Natural Language Processing (NLP) is applied here to build a **binary sentiment classification model** that converts raw text reviews into numerical features and predicts customer satisfaction at scale.

---

## 📂 Dataset Description

| Attribute | Details |
|---|---|
| **File** | `amazon_alexa.tsv` (Tab-Separated Values) |
| **Domain** | E-commerce / Consumer Electronics |
| **Reviews Source** | Amazon Alexa product verified customer reviews |
| **Task Type** | Binary Text Classification |

### Columns

| Column | Description |
|---|---|
| `rating` | Customer star rating (1–5 scale) |
| `date` | Date of review submission |
| `variation` | Alexa product variant (e.g., colour/model) |
| `verified_reviews` | Full-text review written by the customer |
| `feedback` | **Target label**: `1` = Positive (satisfied), `0` = Negative (dissatisfied) |

### Class Distribution

- **Positive (feedback = 1):** Majority class — customers satisfied with the product
- **Negative (feedback = 0):** Minority class — customers who expressed dissatisfaction

---

## 🔄 Project Workflow

```
Raw TSV Data
     │
     ▼
┌─────────────────────┐
│ 1. Data Loading &   │
│    Exploration (EDA)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 2. Visualisation    │
│    (Plots, WordCloud│
│     Heatmap, etc.)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 3. Data Cleaning    │
│  (Drop cols, Encode │
│   Categorical Vars) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 4. NLP Pipeline     │
│  ├─ Punctuation     │
│  │  Removal         │
│  ├─ Stopword        │
│  │  Removal (NLTK)  │
│  └─ Count           │
│     Vectorization   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 5. Model Training   │
│  ├─ Naive Bayes     │
│  └─ Logistic        │
│     Regression      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 6. Evaluation       │
│  ├─ Confusion Matrix│
│  ├─ Accuracy Score  │
│  └─ Classification  │
│     Report          │
└─────────────────────┘
```

---

## 🔍 Task-by-Task Breakdown

---

### TASK 1 — Import Libraries and Dataset

**Libraries Used:**

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `seaborn` | Statistical visualisations |
| `matplotlib` | Plotting and figure rendering |
| `nltk` | Natural language processing (stopwords) |
| `sklearn` | ML models, vectorizer, metrics |
| `wordcloud` | Visual word frequency maps |
| `string` | Punctuation character set |

**Data Loading:**
```python
reviews_df = pd.read_csv('amazon_alexa.tsv', sep='\t')
```

- Dataset is loaded as a TSV (tab-separated) file into a Pandas DataFrame.
- `.info()` reveals column types and non-null counts.
- `.describe()` provides summary statistics for numerical columns like `rating`.

---

### TASK 2 — Exploratory Data Analysis (EDA) & Visualisation

A thorough visual exploration was conducted to understand data distribution, missing values, and text characteristics.

#### 2.1 Missing Value Heatmap
```python
sns.heatmap(reviews_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
```
- Blue heatmap highlights cells with null values.
- Identified null entries in `verified_reviews`; these were filled with empty strings (`fillna('')`).

#### 2.2 Feature Distribution Histogram
```python
reviews_df.hist(bins=30, figsize=(13,5), color='r')
```
- Histograms across all numeric columns (`rating`, `feedback`) reveal skew and class imbalance.

#### 2.3 Review Length Analysis
```python
reviews_df['length'] = reviews_df['verified_reviews'].apply(len)
reviews_df['length'].plot(bins=100, kind='hist')
```
- A new `length` column was engineered to capture text length of each review.
- **Shortest review:** 1 character (effectively empty — a noise entry).
- **Longest review:** 2,851 characters (a highly detailed, elaborate customer review).
- **Mean review length:** ~133 characters — typical short-form review.

#### 2.4 Feedback Class Count Plot
```python
sns.countplot(reviews_df['feedback'], label="Count")
```
- Reveals a **strong class imbalance**: positive reviews vastly outnumber negative reviews, reflecting typical product satisfaction distributions.

#### 2.5 Rating Distribution
```python
reviews_df['rating'].hist(bins=5)
```
- Majority of ratings cluster around **4 and 5 stars**, consistent with high positive feedback proportion.

#### 2.6 Product Variation vs. Average Rating
```python
sns.barplot(x='variation', y='rating', data=reviews_df, palette='deep')
```
- Compares mean ratings across different Alexa product variants (e.g., Charcoal Fabric, Heather Gray, Walnut Finish).
- Identifies which product variants are most/least well-received.

#### 2.7 Word Cloud — All Reviews
```python
sentences_as_one_string = " ".join(reviews_df['verified_reviews'].tolist())
plt.imshow(WordCloud().generate(sentences_as_one_string))
```
- Dominant terms: `love`, `Alexa`, `music`, `great`, `easy`, `use` — reflecting a generally positive sentiment.

#### 2.8 Word Cloud — Negative Reviews Only
```python
negative_list = negative['verified_reviews'].tolist()
negative_sentences_as_one_string = " ".join(negative_list)
plt.imshow(WordCloud().generate(negative_sentences_as_one_string))
```
- Reveals terms specific to dissatisfaction: `return`, `disappointed`, `doesn't work`, `problems` — providing interpretable signals for the model.

---

### TASK 3 — Data Cleaning

#### 3.1 Drop Irrelevant Columns
```python
reviews_df = reviews_df.drop(['date', 'rating', 'length'], axis=1)
```
- `date`, `rating`, and `length` are dropped — these carry either temporal leakage risk or are non-NLP features not required for text-based classification.

#### 3.2 One-Hot Encoding of `variation`
```python
variation_dummies = pd.get_dummies(reviews_df['variation'], drop_first=True)
reviews_df.drop(['variation'], axis=1, inplace=True)
reviews_df = pd.concat([reviews_df, variation_dummies], axis=1)
```
- The `variation` categorical column is encoded using **dummy variables** via `pd.get_dummies()`.
- `drop_first=True` avoids the **Dummy Variable Trap** — a situation where one dummy column can be perfectly predicted from the others, causing multicollinearity.

---

### TASK 4 — Remove Punctuation from Text

```python
import string

def remove_punctuation(message):
    return [char for char in message if char not in string.punctuation]
```

**Example:**
- **Input:** `"Hello Mr. Future, I am so happy to be learning AI now!!"`
- **After punctuation removal:** `"Hello Mr Future I am so happy to be learning AI now"`

- `string.punctuation` provides a comprehensive set of all punctuation characters.
- List comprehension filters characters not in the punctuation set, then joins them back.

---

### TASK 5 — Remove Stopwords

```python
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

cleaned = [word for word in text.split() if word.lower() not in stopwords.words('english')]
```

**Why remove stopwords?**

Stopwords are high-frequency, low-meaning words like `"is"`, `"the"`, `"and"`, `"I"`, `"to"` that occur in virtually all documents. Keeping them would pollute the feature space with noise, not signal.

**Example:**
- **Before:** `"Hello Mr Future I am so happy to be learning AI now"`
- **After stopword removal:** `["Hello", "Mr", "Future", "happy", "learning", "AI"]`
- Only **semantically meaningful** words remain.

---

### TASK 6 — Count Vectorization (Tokenization)

**Concept (Tokenization):**

Count Vectorization converts a collection of text documents into a **Bag-of-Words (BoW)** numerical matrix, where:
- Each **row** represents a document (review).
- Each **column** represents a unique word (token) in the vocabulary.
- Each **cell value** is the count of that word in the document.

**Illustration from slides:**

| | `and` | `document` | `first` | `is` | `one` | `second` | `the` | `third` | `this` |
|---|---|---|---|---|---|---|---|---|---|
| Sample 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 |
| Sample 2 | 0 | 2 | 0 | 1 | 0 | 1 | 1 | 0 | 1 |
| Sample 3 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 1 |
| Sample 4 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 |

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

---

### TASK 7 — Full NLP Preprocessing Pipeline

A unified `message_cleaning()` function encapsulates all text preprocessing steps:

```python
def message_cleaning(message):
    # Step 1: Remove punctuation
    no_punct = [char for char in message if char not in string.punctuation]
    no_punct_str = ''.join(no_punct)
    # Step 2: Remove stopwords
    clean_words = [word for word in no_punct_str.split()
                   if word.lower() not in stopwords.words('english')]
    return clean_words
```

This function is passed directly as the `analyzer` argument into `CountVectorizer`, creating an end-to-end text feature extraction pipeline:

```python
vectorizer = CountVectorizer(analyzer=message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])
```

**Outcome:**
- Produces a sparse feature matrix of shape `(n_reviews, n_vocabulary_tokens)`.
- Numeric columns from the variation dummies are concatenated to form the final feature matrix `X`.
- Target vector `y` = `feedback` column.

---

### TASK 8 — Train-Test Split & Naive Bayes Model Training

#### 8.1 Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

| Split | Proportion | Purpose |
|---|---|---|
| Training Set | 80% | Model fitting and learning |
| Testing Set | 20% | Unbiased model evaluation |

#### 8.2 Multinomial Naive Bayes

**Algorithm Intuition (from slides):**

Naive Bayes is a probabilistic classifier based on **Bayes' Theorem**. It combines:

| Component | Meaning |
|---|---|
| **Prior Probability** | Background class frequency (e.g., 2x more positive reviews than negative) |
| **Likelihood** | Probability of observing specific words given a class |
| **Posterior Probability** | Combined probability used to make the final prediction |

**Why Naive?** It assumes all features (words) are **conditionally independent** given the class label — a simplification that works surprisingly well for text.

**Bayes' Theorem:**
```
P(Class | Words) ∝ P(Words | Class) × P(Class)
```

**Confusion Matrix Terminology:**

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) — Type II Error |
| **Actual Negative** | False Positive (FP) — Type I Error | True Negative (TN) |

**Training:**
```python
from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
```

- `MultinomialNB` is ideal for text classification tasks where features represent word counts.

---

### TASK 9 (TASK 10 in notebook) — Evaluate Naive Bayes Model

#### Training Set Confusion Matrix
```python
y_predict_train = NB_classifier.predict(X_train)
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)
```
- Evaluates model performance on **training data** — checks for underfitting.

#### Test Set Confusion Matrix
```python
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
```
- Evaluates model performance on **unseen test data** — primary evaluation benchmark.

#### Classification Report
```python
print(classification_report(y_test, y_predict_test))
```

Outputs:

| Metric | Definition |
|---|---|
| **Precision** | Of all predicted positives, how many were actually positive? |
| **Recall** | Of all actual positives, how many were correctly predicted? |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Support** | Actual number of instances per class in the test set |

---

### TASK 10 (TASK 11 in notebook) — Logistic Regression Model

A second model is trained and compared as a **baseline alternative** to Naive Bayes.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy {} %'.format(100 * accuracy_score(y_pred, y_test)))
```

**Why Logistic Regression?**
- Despite its name, it's a **classification** algorithm.
- Uses a sigmoid function to output class probabilities.
- Works well as a linear text classifier, especially with high-dimensional BoW features.

```python
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_pred))
```

---

## 📊 Model Comparison Summary

| Model | Type | Best For | Output |
|---|---|---|---|
| **Multinomial Naive Bayes** | Probabilistic | Text/count-based features, fast training | Class label + probability |
| **Logistic Regression** | Discriminative | Linear separable text features, interpretable weights | Class label + probability |

> Both models were evaluated using **Accuracy, Precision, Recall, F1-Score, and Confusion Matrix heatmaps** for comprehensive performance assessment.

---

## 🛠️ Technologies & Libraries

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Data Manipulation** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn, WordCloud |
| **NLP** | NLTK (stopwords), Python `string` module |
| **Feature Engineering** | scikit-learn `CountVectorizer` |
| **Machine Learning** | scikit-learn `MultinomialNB`, `LogisticRegression` |
| **Model Evaluation** | scikit-learn `confusion_matrix`, `classification_report`, `accuracy_score` |
| **Environment** | Jupyter Notebook |

---

## 📁 Project Structure

```
📦 amazon-alexa-sentiment-analysis/
│
├── 📓 public_relations.ipynb        # Main Jupyter Notebook (full pipeline)
├── 📄 amazon_alexa.tsv              # Dataset (Tab-Separated Values)
└── 📝 README.md                     # Project documentation (this file)
```

---

## ⚙️ How to Run


### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

### 2. Download NLTK Stopwords
```python
import nltk
nltk.download('stopwords')
```

### 3. Launch the Notebook
```bash
jupyter notebook public_relations.ipynb
```

> Run cells in order from top to bottom for the complete pipeline execution.

---

## 💡 Key Takeaways & Insights

- **NLP for Business Impact:** Automated sentiment analysis eliminates manual review processing, enabling PR teams to act on customer feedback at scale.

- **Text Preprocessing Matters:** Removing punctuation and stopwords significantly reduces vocabulary noise and improves model signal quality.

- **Bag-of-Words Limitation:** Count Vectorization ignores word order and semantic relationships. Future extensions could explore TF-IDF, Word2Vec, or transformer-based embeddings (BERT).

- **Class Imbalance Awareness:** The strong skew towards positive reviews (reflected in the count plot) means accuracy alone can be misleading — Precision, Recall, and F1-score provide a more complete picture.

- **Model Choice:** Multinomial Naive Bayes is the go-to for text classification with word counts; Logistic Regression offers a linear discriminative alternative that can outperform NB when enough data is available.

---

*Built with 💖 by BikashBIOS*
# 🛒 Rossmann Store Sales Prediction
### Time Series Forecasting with Facebook Prophet

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Prophet](https://img.shields.io/badge/Facebook-Prophet-blue?logo=meta)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-orange)

---

## 📌 Project Overview

This project demonstrates how businesses can leverage **AI/ML to develop predictive models** for forecasting daily sales. For companies to become competitive and skyrocket their growth, they need predictive models that attempt to forecast future sales based on historical data — while taking into account **seasonality effects, demand, holidays, promotions, and competition**.

> **Business Context:** You work as a Data Scientist in the sales department. The sales team has provided data from **1,115 Rossmann stores**. The objective is to **predict future daily sales** based on a rich set of features.

---

## 📂 Repository Structure

```
rossmann-sales-prediction/
│
├── data/
│   ├── train.csv                  # Historical sales training data
│   └── store.csv                  # Store metadata and attributes
│
├── notebooks/
│   └── sales_data.ipynb           # Full end-to-end analysis notebook
│
├── outputs/
│   └── test.csv                   # Merged dataset (generated)
│
└── README.md
```

---

## 🗃️ Dataset Description

The project uses **two datasets** that are merged for analysis:

### 1. `train.csv` — Sales Transaction Data

| Column | Type | Description |
|---|---|---|
| `Store` | int | Unique store identifier |
| `Date` | datetime | Transaction date |
| `Sales` | int | **Target variable** — Daily sales revenue |
| `Customers` | int | Number of customers on a given day |
| `Open` | bool | Whether the store is open (1) or closed (0) |
| `Promo` | bool | Whether the store is running a promotion that day |
| `StateHoliday` | str | State holiday type: `a`=Public, `b`=Easter, `c`=Christmas, `0`=None |
| `SchoolHoliday` | bool | Whether the store was affected by school closure |

### 2. `store.csv` — Store Metadata

| Column | Type | Description |
|---|---|---|
| `StoreType` | str | Store category: `a`, `b`, `c`, or `d` |
| `Assortment` | str | Assortment level: `a`=Basic, `b`=Extra, `c`=Extended |
| `CompetitionDistance` | float | Distance in meters to nearest competitor |
| `CompetitionOpenSinceMonth` | float | Month when nearest competitor opened |
| `CompetitionOpenSinceYear` | float | Year when nearest competitor opened |
| `Promo2` | bool | Whether store participates in a continuous promotion (0=No, 1=Yes) |
| `Promo2SinceWeek` | float | Week when store started participating in Promo2 |
| `Promo2SinceYear` | float | Year when store started participating in Promo2 |
| `PromoInterval` | str | Months when Promo2 restarts (e.g., `"Feb,May,Aug,Nov"`) |

---

## 🔧 Tech Stack & Libraries

```python
import pandas as pd          # Data manipulation
import numpy as np           # Numerical operations
import seaborn as sns        # Statistical visualization
import matplotlib.pyplot as plt  # Plotting
import datetime              # Date handling
from prophet import Prophet  # Time series forecasting
```

---

## 🚀 Full Pipeline Walkthrough

### Task 1 — Import Libraries and Load Datasets

Both datasets are loaded using `pd.read_csv()`. Initial exploration is performed using:

```python
sales_train_df.head(5)
sales_train_df.info()
sales_train_df.describe()

store_info_df.head(5)
store_info_df.info()
store_info_df.describe()
```

These steps provide a quick snapshot of data types, shape, summary statistics, and the presence of missing values.

---

### Task 2 — Exploratory Data Analysis: Sales Dataset

#### 2.1 Null Value Detection

A heatmap is plotted to visually detect missing values across the sales dataset:

```python
sns.heatmap(sales_train_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
```

#### 2.2 Distribution Analysis

Histograms are generated to understand the distribution of all numerical features:

```python
sales_train_df.hist(bins=30, figsize=(20, 20), color='r')
```

#### 2.3 Open vs. Closed Store Analysis

Stores that are **closed contribute zero sales** and would distort the model. They are identified and removed:

```python
closed_train_df = sales_train_df[sales_train_df['Open'] == 0]
open_train_df   = sales_train_df[sales_train_df['Open'] == 1]

# Only keep open stores
sales_train_df = sales_train_df[sales_train_df['Open'] == 1]

# Drop the Open column — no longer needed
sales_train_df.drop(['Open'], axis=1, inplace=True)
```

---

### Task 3 — Exploratory Data Analysis: Store Dataset

#### 3.1 Null Value Detection

```python
sns.heatmap(store_info_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
```

#### 3.2 Handling Missing Values

Missing values in promotion and competition columns are handled strategically:

**Strategy 1 — Fill with 0 (for stores not participating in Promo2 or competition)**

Stores with `Promo2 == 0` logically have no promo-related metadata. These NaN values are filled with `0`:

```python
fill_dict = {
    'PromoInterval':             '0',
    'Promo2SinceWeek':           0.0,
    'Promo2SinceYear':           0.0,
    'CompetitionOpenSinceYear':  0.0,
    'CompetitionOpenSinceMonth': 0.0
}
store_info_df = store_info_df.fillna(value=fill_dict)
```

**Strategy 2 — Fill with Mean (for `CompetitionDistance`)**

The distance to the nearest competitor is filled with the column mean to avoid data loss:

```python
store_info_df['CompetitionDistance'] = store_info_df['CompetitionDistance'].fillna(
    store_info_df['CompetitionDistance'].mean()
)
```

A final heatmap confirms that all missing values have been resolved.

---

### Task 4 — Merged Dataset Analysis & Feature Engineering

#### 4.1 Merging the Datasets

Both datasets are merged on the `Store` column using an inner join to create a unified analytical dataset:

```python
sales_train_all_df = pd.merge(sales_train_df, store_info_df, how='inner', on='Store')
sales_train_all_df.to_csv('test.csv', index=False)
```

#### 4.2 Correlation Analysis

The correlation of all numeric features against `Sales` is calculated and sorted:

```python
correlations = sales_train_all_df.corr(numeric_only=True)['Sales'].sort_values()
```

A full **correlation heatmap** is plotted to understand inter-feature relationships:

```python
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(correlations, annot=True)
```

Key correlations help identify which features have the strongest positive or negative relationship with daily sales.

#### 4.3 Feature Engineering — Date Decomposition

The `Date` column is decomposed into individual time components to capture seasonality signals:

```python
sales_train_all_df['Year']  = pd.DatetimeIndex(sales_train_all_df['Date']).year
sales_train_all_df['Month'] = pd.DatetimeIndex(sales_train_all_df['Date']).month
sales_train_all_df['Day']   = pd.DatetimeIndex(sales_train_all_df['Date']).day
```

#### 4.4 Temporal Trend Visualizations

**Average Sales & Customers Per Month**

```python
sales_train_all_df.groupby('Month')[['Sales']].mean().plot(marker='o', color='r')
sales_train_all_df.groupby('Month')[['Customers']].mean().plot(marker='^', color='b')
```

**Average Sales & Customers Per Day**

```python
sales_train_all_df.groupby('Day')[['Sales']].mean().plot(marker='o', color='r')
sales_train_all_df.groupby('Day')[['Customers']].mean().plot(marker='^', color='b')
```

**Average Sales & Customers Per Day of Week**

```python
sales_train_all_df.groupby('DayOfWeek')[['Sales']].mean().plot(marker='o', color='r')
sales_train_all_df.groupby('DayOfWeek')[['Customers']].mean().plot(marker='^', color='b')
```

**Sales Over Time by Store Type**

```python
sales_train_all_df.groupby(['Date', 'StoreType'])['Sales'].mean().unstack().plot(figsize=(20, 10))
```

This reveals how different store types (a, b, c, d) behave over time.

#### 4.5 Promotion Impact Analysis

The effect of promotions on both Sales and Customer count is visualized using bar plots and violin plots:

```python
# Bar Plot
sns.barplot(x='Promo', y='Sales',     data=sales_train_all_df)
sns.barplot(x='Promo', y='Customers', data=sales_train_all_df)

# Violin Plot (distribution shape)
sns.violinplot(x='Promo', y='Sales',     data=sales_train_all_df)
sns.violinplot(x='Promo', y='Customers', data=sales_train_all_df)
```

Promotions clearly show a positive uplift in both sales and foot traffic.

---

### Task 5 — Time Series Forecasting with Facebook Prophet

#### 5.1 What is Facebook Prophet?

Facebook Prophet is an **open-source forecasting tool** released by Meta's Core Data Science team. It is a procedure for forecasting time series data based on an **additive model** where non-linear trends are fit with:

- **Yearly seasonality**
- **Weekly seasonality**
- **Daily seasonality**
- **Holiday effects**

> Prophet works best with time series that have **strong seasonal effects** and **several seasons of historical data**.

📎 References:
- [Prophet Research Paper](https://research.fb.com/prophet-forecasting-at-scale/)
- [Prophet Python Quickstart](https://facebook.github.io/prophet/docs/quick_start.html#python-api)

#### 5.2 Basic Sales Prediction (Without Holidays)

A reusable prediction function is defined that:
1. Filters data for a specific store
2. Renames columns to Prophet's required format (`ds`, `y`)
3. Sorts data chronologically
4. Fits the Prophet model
5. Generates future forecast for a specified number of periods
6. Plots the forecast and its components

```python
def sales_prediction(Store_ID, sales_df, periods):
    sales_df = sales_df[sales_df['Store'] == Store_ID]
    sales_df = sales_df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    sales_df = sales_df.sort_values('ds')

    model    = Prophet()
    model.fit(sales_df)
    future   = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    figure  = model.plot(forecast, xlabel='Date', ylabel='Sales')
    figure2 = model.plot_components(forecast)

# Example: Predict 60 days ahead for Store 10
sales_prediction(10, sales_train_all_df, 60)
```

#### 5.3 Advanced Sales Prediction (With Holiday Effects)

Holiday information is extracted from the dataset and passed into Prophet to improve forecast accuracy.

**Step 1 — Extract Holiday Dates**

```python
# School holidays
school_holidays = sales_train_all_df[
    sales_train_all_df['SchoolHoliday'] == 1
].loc[:, 'Date'].values

# State holidays (Public, Easter, Christmas)
state_holidays = sales_train_all_df[
    (sales_train_all_df['StateHoliday'] == 'a') |
    (sales_train_all_df['StateHoliday'] == 'b') |
    (sales_train_all_df['StateHoliday'] == 'c')
].loc[:, 'Date'].values
```

**Step 2 — Format as Prophet Holiday DataFrames**

```python
state_holidays = pd.DataFrame({
    'ds':      pd.to_datetime(state_holidays),
    'holiday': 'state_holiday'
})

school_holidays = pd.DataFrame({
    'ds':      pd.to_datetime(school_holidays),
    'holiday': 'school_holiday'
})

# Concatenate both
school_state_holidays = pd.concat((state_holidays, school_holidays))
```

**Step 3 — Define Holiday-Aware Prediction Function**

```python
def sales_prediction(Store_ID, sales_df, holidays, periods):
    sales_df = sales_df[sales_df['Store'] == Store_ID]
    sales_df = sales_df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    sales_df = sales_df.sort_values('ds')

    model    = Prophet(holidays=holidays)
    model.fit(sales_df)
    future   = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    figure  = model.plot(forecast, xlabel='Date', ylabel='Sales')
    figure2 = model.plot_components(forecast)

# Example: Predict 60 days ahead for Store 6, with holiday effects
sales_prediction(6, sales_train_all_df, school_state_holidays, 60)
```

---

## 📊 Visualizations Summary

| Visualization | Purpose |
|---|---|
| Null value heatmap | Detect and confirm missing data |
| Feature histograms | Understand data distributions |
| Correlation heatmap | Identify feature relationships with Sales |
| Monthly sales & customers | Detect seasonal monthly trends |
| Daily sales & customers | Identify intra-month sales patterns |
| Day-of-week analysis | Spot weekly sales cycles |
| Sales by store type over time | Compare store category performance |
| Promo bar & violin plots | Quantify promotion impact on sales |
| Prophet forecast plot | Future sales prediction with uncertainty bands |
| Prophet component plot | Decompose trend, weekly, and yearly seasonality |

---

## 📈 Model Details

| Attribute | Value |
|---|---|
| **Model** | Facebook Prophet |
| **Model Type** | Additive Time Series Forecasting |
| **Input Format** | `ds` (date), `y` (sales) |
| **Forecast Horizon** | Configurable (e.g., 60 days) |
| **Holiday Support** | School Holidays + State Holidays |
| **Seasonality** | Yearly, Weekly, Daily |
| **Scope** | Per-store individual forecasting |

> **Note:** Prophet is a univariate time series model — it forecasts using `Date → Sales` directly. It is not a traditional train/test classification model; therefore, standard metrics like a confusion matrix and classification accuracy are not applicable here. Instead, Prophet provides **uncertainty intervals** (upper/lower bounds) around each forecast point, which serve as its built-in evaluation mechanism for regression-based time series tasks.

---

## 📉 Evaluation Approach

Since Prophet is a **time series regression model**, evaluation focuses on forecast quality rather than classification metrics:

**Uncertainty Intervals** — Prophet automatically generates `yhat_lower` and `yhat_upper` alongside the point forecast `yhat`, quantifying prediction confidence.

**Component Decomposition** — `model.plot_components()` breaks the forecast into trend, weekly seasonality, and yearly seasonality — making the model's reasoning fully interpretable.

**Business Validation** — Forecasts for individual stores can be visually validated against historical actuals in the plot output, providing intuitive, stakeholder-friendly evaluation.

For a more rigorous numeric evaluation, standard regression metrics can be computed:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Merge forecast with actuals on date
merged = forecast[['ds', 'yhat']].merge(actuals[['ds', 'y']], on='ds')

mae  = mean_absolute_error(merged['y'], merged['yhat'])
rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
```

---

## ⚙️ Setup & Installation

```bash

# Install dependencies
pip install pandas numpy seaborn matplotlib prophet scikit-learn

# Launch the notebook
jupyter notebook notebooks/sales_data.ipynb
```

---

## 🔑 Key Insights

- **Promotions** significantly increase both sales and customer traffic
- **Store Type** has a material impact on sales volume over time — Type `b` stores tend to generate higher average sales
- **Seasonality** is strong: sales peak around December (Christmas) and dip mid-year
- **Day of Week** matters — weekday vs. weekend patterns vary by store type
- **Holidays** (school and state) visibly affect sales patterns and improve forecast accuracy when included in the Prophet model
- **Competition Distance** has a negative correlation with sales — closer competitors reduce revenue

---

## 📚 References

- [Facebook Prophet — Official Documentation](https://facebook.github.io/prophet/)
- [Prophet Forecasting at Scale — Research Paper](https://research.fb.com/prophet-forecasting-at-scale/)
- [Rossmann Store Sales — Kaggle Dataset](https://www.kaggle.com/competitions/rossmann-store-sales)

---
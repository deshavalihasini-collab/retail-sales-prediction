# 🛒 Automated Retail Sales Analysis and Profit Prediction System
### Using Machine Learning and Python

> **Internship Project — Thiranex Data Science Internship**
> **Author:** Hasini | B.Tech IT, JBIET Hyderabad (JNTUH)

---

## 📌 Project Overview

This project performs end-to-end automated analysis of retail sales data and predicts profit using supervised machine learning models. It covers data generation, exploratory data analysis (EDA), feature engineering, model training, evaluation, and business insights — all on a Superstore-style retail dataset.

---

## 🎯 Objectives

- Analyze retail sales data across categories, regions, and segments
- Identify key factors affecting profitability (discount, sales, category)
- Build and compare ML models to predict profit on new transactions
- Generate actionable business insights for retail decision-making

---

## 🗂️ Project Structure

```
retail-sales-prediction/
│
├── retail_sales_analysis.py   # Main Python script (all sections)
├── README.md                  # Project documentation
│
└── outputs/
    ├── superstore_data.csv              # Generated dataset (1000 rows)
    ├── model_results.csv                # Model evaluation metrics
    ├── 01_sales_profit_by_category.png  # EDA: Category analysis
    ├── 02_monthly_sales_trend.png       # EDA: Time series trend
    ├── 03_region_segment_analysis.png   # EDA: Region & segment
    ├── 04_discount_profit_analysis.png  # EDA: Discount impact
    ├── 05_correlation_heatmap.png       # Feature correlations
    ├── 06_model_comparison.png          # R², RMSE, MAE comparison
    ├── 07_actual_vs_predicted.png       # Best model performance
    └── 08_feature_importance.png        # Random Forest importances
```

---

## 📊 Dataset

**Source:** Superstore-style synthetic retail dataset (industry-standard format)

| Feature | Description |
|---|---|
| Sales | Transaction sale amount ($) |
| Quantity | Units sold |
| Discount | Discount rate (0.0 – 0.5) |
| Profit | Actual profit earned ($) |
| Category | Furniture / Office Supplies / Technology |
| Sub_Category | Chairs, Phones, Binders, etc. |
| Region | East / West / Central / South |
| Segment | Consumer / Corporate / Home Office |
| Ship_Mode | Standard / Second / First / Same Day |

**Size:** 1000 transactions | **Date Range:** 2020–2023 | **Missing Values:** None

---

## 🔍 EDA Highlights

| Insight | Value |
|---|---|
| Total Sales | $537,459 |
| Total Profit | $71,650 |
| Average Profit Margin | 13.72% |
| Loss Orders | 3.9% |
| Best Category (Profit) | Technology |
| Best Region (Sales) | East |
| Avg Profit (Discount < 30%) | $82.30 |
| Avg Profit (Discount ≥ 30%) | $31.59 |

---

## 🤖 ML Models & Results

Three regression models were trained and compared:

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.9052 | 22.33 | 17.33 |
| Decision Tree | 0.9628 | 13.99 | 10.97 |
| **Random Forest** ✅ | **0.9754** | **11.39** | **9.27** |

**Best Model: Random Forest** with R² = 0.9754 (97.54% variance explained)

---

## ⚙️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, preprocessing, evaluation |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/retail-sales-prediction.git
cd retail-sales-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the project
```bash
python retail_sales_analysis.py
```

All outputs (plots + CSVs) will be saved to the `outputs/` folder automatically.

---

## 🔮 Profit Prediction Example

```python
# Predict profit for a new transaction
predict_profit(
    sales=1200, quantity=3, discount=0.0,
    category='Technology', sub_category='Phones',
    region='East', segment='Corporate',
    ship_mode='First Class'
)
# Output: +$237.40
```

---

## 💡 Key Business Insights

1. **Discount impact is critical** — orders with ≥30% discount earn 62% less profit on average
2. **Technology** is the most profitable category
3. **East region** generates the highest sales volume
4. **Sales amount** is the strongest predictor of profit (feature importance)
5. Random Forest outperforms Linear Regression and Decision Tree for this dataset

---

## 📋 Project Sections

| Section | Description |
|---|---|
| 1. Dataset | Superstore-style retail data (1000 rows, 14 features) |
| 2. EDA | 5 visualization plots covering category, trend, region, discount |
| 3. Preprocessing | Label encoding, train-test split (80/20), StandardScaler |
| 4. Modeling | Linear Regression, Decision Tree, Random Forest |
| 5. Evaluation | R², RMSE, MAE with visual comparison |
| 6. Prediction | `predict_profit()` function for new data |
| 7. Insights | Business conclusions and recommendations |

---

## 🏫 Academic Details

- **Institution:** J.B. Institute of Engineering and Technology (JBIET), Hyderabad
- **Affiliation:** JNTUH | Regulation: R22
- **Internship:** Thiranex — Data Science Internship (Task 3 / Final Project)
- **Domain:** Data Science | Machine Learning | Retail Analytics

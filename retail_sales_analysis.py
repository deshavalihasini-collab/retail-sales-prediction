# ============================================================
# Automated Retail Sales Analysis and Profit Prediction System
# Using Machine Learning and Python
# Author: Hasini | Internship Project - Thiranex
# Dataset: Superstore Sales Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# ── Output folder ──────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'font.family': 'DejaVu Sans'})

print("=" * 65)
print("  Automated Retail Sales Analysis & Profit Prediction System")
print("=" * 65)


# ============================================================
# SECTION 1 – DATASET GENERATION (Superstore-style)
# ============================================================
print("\n[1] Generating Superstore-style dataset...")

np.random.seed(42)
n = 1000

categories    = ['Furniture', 'Office Supplies', 'Technology']
sub_cats      = {
    'Furniture'       : ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
    'Office Supplies' : ['Binders', 'Paper', 'Storage', 'Art', 'Fasteners'],
    'Technology'      : ['Phones', 'Accessories', 'Machines', 'Copiers'],
}
regions       = ['East', 'West', 'Central', 'South']
segments      = ['Consumer', 'Corporate', 'Home Office']
ship_modes    = ['Standard Class', 'Second Class', 'First Class', 'Same Day']

cat_list = np.random.choice(categories, n)
sub_list = [np.random.choice(sub_cats[c]) for c in cat_list]

base_sales  = {'Furniture': 400, 'Office Supplies': 80,  'Technology': 600}
base_margin = {'Furniture': 0.10,'Office Supplies': 0.25, 'Technology': 0.20}

sales_vals  = np.array([base_sales[c]  * np.random.uniform(0.5, 3.0) for c in cat_list])
discounts   = np.random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], n,
                                p=[0.35, 0.25, 0.20, 0.10, 0.07, 0.03])
sales_final = sales_vals * (1 - discounts)
margins     = np.array([base_margin[c] for c in cat_list])
profit_vals = sales_final * margins * (1 - discounts * 1.5) + np.random.normal(0, 10, n)
quantities  = np.random.randint(1, 15, n)

df = pd.DataFrame({
    'Order_ID'    : [f'ORD-{2020 + i//365}-{i%10000:04d}' for i in range(n)],
    'Order_Date'  : pd.date_range('2020-01-01', periods=n, freq='8h'),
    'Ship_Mode'   : np.random.choice(ship_modes, n),
    'Segment'     : np.random.choice(segments, n),
    'Region'      : np.random.choice(regions, n),
    'Category'    : cat_list,
    'Sub_Category': sub_list,
    'Sales'       : np.round(sales_final, 2),
    'Quantity'    : quantities,
    'Discount'    : discounts,
    'Profit'      : np.round(profit_vals, 2),
})
df['Year']  = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['Profit_Margin'] = np.round(df['Profit'] / df['Sales'] * 100, 2)

df.to_csv("outputs/superstore_data.csv", index=False)
print(f"   Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")


# ============================================================
# SECTION 2 – EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n[2] Exploratory Data Analysis...")

print("\n   ── Basic Info ──")
print(df[['Sales','Quantity','Discount','Profit','Profit_Margin']].describe().round(2).to_string())

print(f"\n   Missing values : {df.isnull().sum().sum()}")
print(f"   Duplicate rows : {df.duplicated().sum()}")
print(f"   Date range     : {df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}")

# ── Plot 1: Sales & Profit by Category ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Sales & Profit by Category", fontsize=14, fontweight='bold', y=1.01)

cat_sales  = df.groupby('Category')['Sales'].sum().sort_values()
cat_profit = df.groupby('Category')['Profit'].sum().sort_values()
colors = ['#4E79A7','#F28E2B','#E15759']

cat_sales.plot(kind='barh', ax=axes[0], color=colors, edgecolor='white')
axes[0].set_title("Total Sales by Category")
axes[0].set_xlabel("Total Sales ($)")
for bar in axes[0].patches:
    axes[0].text(bar.get_width()+500, bar.get_y()+bar.get_height()/2,
                 f"${bar.get_width():,.0f}", va='center', fontsize=9)

cat_profit.plot(kind='barh', ax=axes[1], color=colors, edgecolor='white')
axes[1].set_title("Total Profit by Category")
axes[1].set_xlabel("Total Profit ($)")
for bar in axes[1].patches:
    axes[1].text(bar.get_width()+100, bar.get_y()+bar.get_height()/2,
                 f"${bar.get_width():,.0f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig("outputs/01_sales_profit_by_category.png", bbox_inches='tight')
plt.close()
print("   Saved: 01_sales_profit_by_category.png")

# ── Plot 2: Monthly Sales Trend ──────────────────────────────
monthly = df.groupby(['Year','Month'])['Sales'].sum().reset_index()
monthly['Period'] = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(monthly['Period'], monthly['Sales'], marker='o', color='#4E79A7',
        linewidth=2, markersize=5)
ax.fill_between(range(len(monthly)), monthly['Sales'], alpha=0.15, color='#4E79A7')
ax.set_xticks(range(0, len(monthly), 3))
ax.set_xticklabels(monthly['Period'].iloc[::3], rotation=45, ha='right', fontsize=8)
ax.set_title("Monthly Sales Trend (2020–2023)", fontsize=13, fontweight='bold')
ax.set_xlabel("Period")
ax.set_ylabel("Total Sales ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
plt.savefig("outputs/02_monthly_sales_trend.png", bbox_inches='tight')
plt.close()
print("   Saved: 02_monthly_sales_trend.png")

# ── Plot 3: Regional Analysis ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

region_data = df.groupby('Region')[['Sales','Profit']].sum()
region_data.plot(kind='bar', ax=axes[0], color=['#4E79A7','#59A14F'], edgecolor='white')
axes[0].set_title("Sales & Profit by Region", fontsize=12, fontweight='bold')
axes[0].set_xlabel("")
axes[0].set_xticklabels(region_data.index, rotation=0)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

seg_sales = df.groupby('Segment')['Sales'].sum()
wedge_props = dict(linewidth=2)
axes[1].pie(seg_sales, labels=seg_sales.index, autopct='%1.1f%%',
            colors=['#4E79A7','#F28E2B','#E15759'],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title("Sales Distribution by Segment", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/03_region_segment_analysis.png", bbox_inches='tight')
plt.close()
print("   Saved: 03_region_segment_analysis.png")

# ── Plot 4: Discount vs Profit ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(df['Discount'], df['Profit'], alpha=0.4,
                c=df['Profit'].apply(lambda x: '#E15759' if x < 0 else '#59A14F'),
                edgecolors='none', s=20)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_title("Discount vs Profit", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Discount")
axes[0].set_ylabel("Profit ($)")
green_patch = mpatches.Patch(color='#59A14F', label='Profit > 0')
red_patch   = mpatches.Patch(color='#E15759', label='Loss')
axes[0].legend(handles=[green_patch, red_patch])

disc_profit = df.groupby('Discount')['Profit'].mean()
axes[1].bar(disc_profit.index.astype(str), disc_profit.values,
            color=['#59A14F' if v > 0 else '#E15759' for v in disc_profit.values],
            edgecolor='white')
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title("Avg Profit per Discount Level", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Discount Rate")
axes[1].set_ylabel("Avg Profit ($)")

plt.tight_layout()
plt.savefig("outputs/04_discount_profit_analysis.png", bbox_inches='tight')
plt.close()
print("   Saved: 04_discount_profit_analysis.png")

# ── Plot 5: Correlation Heatmap ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr_cols = ['Sales','Quantity','Discount','Profit','Profit_Margin']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/05_correlation_heatmap.png", bbox_inches='tight')
plt.close()
print("   Saved: 05_correlation_heatmap.png")


# ============================================================
# SECTION 3 – FEATURE ENGINEERING & PREPROCESSING
# ============================================================
print("\n[3] Feature Engineering & Preprocessing...")

le = LabelEncoder()
df_ml = df.copy()
for col in ['Category', 'Sub_Category', 'Region', 'Segment', 'Ship_Mode']:
    df_ml[col + '_enc'] = le.fit_transform(df_ml[col])

features = ['Sales','Quantity','Discount','Year','Month',
            'Category_enc','Sub_Category_enc','Region_enc',
            'Segment_enc','Ship_Mode_enc']
target   = 'Profit'

X = df_ml[features]
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")
print(f"   Features used    : {len(features)}")


# ============================================================
# SECTION 4 – MODEL TRAINING & EVALUATION
# ============================================================
print("\n[4] Training ML Models...")

models = {
    'Linear Regression' : LinearRegression(),
    'Decision Tree'     : DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

results = {}
predictions = {}

for name, model in models.items():
    X_tr = X_train_sc if name == 'Linear Regression' else X_train
    X_te = X_test_sc  if name == 'Linear Regression' else X_test
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    predictions[name] = y_pred
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    print(f"   {name:22s} | R²={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f}")

results_df = pd.DataFrame(results).T
results_df.to_csv("outputs/model_results.csv")

best_model_name = results_df['R²'].idxmax()
best_model      = models[best_model_name]
print(f"\n   ✅ Best Model: {best_model_name}  (R² = {results_df.loc[best_model_name,'R²']:.4f})")


# ============================================================
# SECTION 5 – MODEL VISUALIZATIONS
# ============================================================
print("\n[5] Generating model performance plots...")

# ── Plot 6: Model Comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Performance Comparison", fontsize=14, fontweight='bold')
metrics = ['R²', 'RMSE', 'MAE']
colors_m = ['#4E79A7','#F28E2B','#E15759']

for i, metric in enumerate(metrics):
    vals = results_df[metric]
    bars = axes[i].bar(results_df.index, vals, color=colors_m, edgecolor='white')
    axes[i].set_title(metric, fontsize=12, fontweight='bold')
    axes[i].set_xticklabels(results_df.index, rotation=15, ha='right', fontsize=8)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("outputs/06_model_comparison.png", bbox_inches='tight')
plt.close()
print("   Saved: 06_model_comparison.png")

# ── Plot 7: Actual vs Predicted (Best Model) ─────────────────
y_pred_best = predictions[best_model_name]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Actual vs Predicted Profit — {best_model_name}", fontsize=13, fontweight='bold')

axes[0].scatter(y_test, y_pred_best, alpha=0.4, color='#4E79A7', s=20, edgecolors='none')
lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
axes[0].set_xlabel("Actual Profit ($)")
axes[0].set_ylabel("Predicted Profit ($)")
axes[0].set_title("Scatter: Actual vs Predicted")
axes[0].legend()

residuals = y_test.values - y_pred_best
axes[1].hist(residuals, bins=40, color='#4E79A7', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel("Residual ($)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("outputs/07_actual_vs_predicted.png", bbox_inches='tight')
plt.close()
print("   Saved: 07_actual_vs_predicted.png")

# ── Plot 8: Feature Importance (Random Forest) ───────────────
rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
importances.plot(kind='barh', ax=ax, color='#4E79A7', edgecolor='white')
ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight='bold')
ax.set_xlabel("Importance Score")
for bar in ax.patches:
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("outputs/08_feature_importance.png", bbox_inches='tight')
plt.close()
print("   Saved: 08_feature_importance.png")


# ============================================================
# SECTION 6 – PROFIT PREDICTION FUNCTION
# ============================================================
print("\n[6] Profit Prediction System...")

def predict_profit(sales, quantity, discount, category,
                   sub_category, region, segment, ship_mode,
                   year=2024, month=6):
    """
    Predict profit for a new retail transaction.

    Parameters
    ----------
    sales        : float  – sale amount in $
    quantity     : int    – number of units
    discount     : float  – discount rate (0.0 – 0.5)
    category     : str    – 'Furniture' | 'Office Supplies' | 'Technology'
    sub_category : str    – e.g. 'Chairs', 'Phones', 'Binders' …
    region       : str    – 'East' | 'West' | 'Central' | 'South'
    segment      : str    – 'Consumer' | 'Corporate' | 'Home Office'
    ship_mode    : str    – 'Standard Class' | 'Second Class' | 'First Class' | 'Same Day'
    year, month  : int    – date context

    Returns
    -------
    float – predicted profit in $
    """
    cat_map  = {c: i for i, c in enumerate(sorted(df['Category'].unique()))}
    sub_map  = {c: i for i, c in enumerate(sorted(df['Sub_Category'].unique()))}
    reg_map  = {c: i for i, c in enumerate(sorted(df['Region'].unique()))}
    seg_map  = {c: i for i, c in enumerate(sorted(df['Segment'].unique()))}
    shp_map  = {c: i for i, c in enumerate(sorted(df['Ship_Mode'].unique()))}

    row = np.array([[sales, quantity, discount, year, month,
                     cat_map.get(category, 0),
                     sub_map.get(sub_category, 0),
                     reg_map.get(region, 0),
                     seg_map.get(segment, 0),
                     shp_map.get(ship_mode, 0)]])

    pred = rf_model.predict(row)[0]
    return round(pred, 2)

# Demo predictions
test_cases = [
    (1200, 3, 0.0, 'Technology',       'Phones',    'East',    'Corporate',  'First Class'),
    (250,  5, 0.2, 'Office Supplies',  'Binders',   'West',    'Consumer',   'Standard Class'),
    (800,  2, 0.4, 'Furniture',        'Chairs',    'Central', 'Home Office','Second Class'),
]

print("\n   ── Sample Predictions (Random Forest) ──")
print(f"   {'Sales':>8} {'Qty':>4} {'Disc':>5} {'Category':<18} {'Predicted Profit':>16}")
print("   " + "-" * 60)
for sales, qty, disc, cat, sub, reg, seg, ship in test_cases:
    pred = predict_profit(sales, qty, disc, cat, sub, reg, seg, ship)
    sign = "+" if pred >= 0 else ""
    print(f"   ${sales:>7,.0f} {qty:>4}  {disc:.0%}   {cat:<18} {sign}${pred:>12,.2f}")


# ============================================================
# SECTION 7 – BUSINESS INSIGHTS SUMMARY
# ============================================================
print("\n[7] Business Insights Summary...")
print("\n   ── Key Findings ──")
print(f"   • Total Sales    : ${df['Sales'].sum():>12,.2f}")
print(f"   • Total Profit   : ${df['Profit'].sum():>12,.2f}")
print(f"   • Avg Margin     : {df['Profit_Margin'].mean():.2f}%")
loss_pct = (df['Profit'] < 0).mean() * 100
print(f"   • Loss Orders    : {loss_pct:.1f}% of all orders")
best_cat = df.groupby('Category')['Profit'].sum().idxmax()
print(f"   • Best Category  : {best_cat}")
best_reg = df.groupby('Region')['Sales'].sum().idxmax()
print(f"   • Best Region    : {best_reg}")
high_disc_loss = df[df['Discount'] >= 0.3]['Profit'].mean()
low_disc_loss  = df[df['Discount'] <  0.3]['Profit'].mean()
print(f"   • Avg Profit (Discount ≥ 30%) : ${high_disc_loss:.2f}")
print(f"   • Avg Profit (Discount < 30%) : ${low_disc_loss:.2f}")
print(f"   • Best ML Model  : {best_model_name}")
print(f"   • Best R² Score  : {results_df['R²'].max():.4f}")

print("\n" + "=" * 65)
print("  ✅  All outputs saved to the 'outputs/' folder")
print("=" * 65)

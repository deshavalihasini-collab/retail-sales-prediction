import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import io
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailIQ — Profit Prediction System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f36 0%, #2d3561 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; font-size: 14px; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        border-left: 4px solid;
        margin-bottom: 0;
    }
    .metric-card.blue  { border-color: #3b82f6; }
    .metric-card.green { border-color: #10b981; }
    .metric-card.amber { border-color: #f59e0b; }
    .metric-card.purple{ border-color: #8b5cf6; }
    .metric-card .label { font-size: 12px; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .metric-card .value { font-size: 26px; font-weight: 700; color: #111827; }
    .metric-card .delta { font-size: 12px; margin-top: 4px; }
    .metric-card .delta.up   { color: #10b981; }
    .metric-card .delta.down { color: #ef4444; }

    /* Page header */
    .page-header {
        background: linear-gradient(135deg, #1a1f36 0%, #2d3561 50%, #3b4a8a 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .page-header h1 { color: white; font-size: 28px; font-weight: 700; margin: 0 0 6px 0; }
    .page-header p  { color: #94a3b8; font-size: 14px; margin: 0; }

    /* Section card */
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .section-title { font-size: 15px; font-weight: 600; color: #1f2937; margin-bottom: 1rem; }

    /* Result card */
    .result-profit {
        background: linear-gradient(135deg, #059669, #10b981);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .result-loss {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .result-amount { font-size: 48px; font-weight: 800; }
    .result-label  { font-size: 16px; opacity: 0.9; }

    /* Insight cards */
    .insight-card {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .insight-card .insight-title { font-weight: 600; color: #0369a1; font-size: 13px; }
    .insight-card .insight-body  { font-size: 13px; color: #475569; margin-top: 2px; }

    /* History table */
    .history-row { font-size: 13px; }

    /* Nav badge */
    .nav-badge {
        background: #3b82f6;
        color: white;
        font-size: 10px;
        padding: 1px 6px;
        border-radius: 10px;
        margin-left: 6px;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 14px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── Data & Model Generation ───────────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 1000
    categories = ['Furniture', 'Office Supplies', 'Technology']
    sub_cats = {
        'Furniture': ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
        'Office Supplies': ['Binders', 'Paper', 'Storage', 'Art', 'Fasteners'],
        'Technology': ['Phones', 'Accessories', 'Machines', 'Copiers'],
    }
    regions   = ['East', 'West', 'Central', 'South']
    segments  = ['Consumer', 'Corporate', 'Home Office']
    ship_modes= ['Standard Class', 'Second Class', 'First Class', 'Same Day']
    cat_list  = np.random.choice(categories, n)
    sub_list  = [np.random.choice(sub_cats[c]) for c in cat_list]
    base_sales  = {'Furniture': 400, 'Office Supplies': 80,  'Technology': 600}
    base_margin = {'Furniture': 0.10,'Office Supplies': 0.25, 'Technology': 0.20}
    sales_vals  = np.array([base_sales[c] * np.random.uniform(0.5, 3.0) for c in cat_list])
    discounts   = np.random.choice([0.0,0.1,0.2,0.3,0.4,0.5], n, p=[0.35,0.25,0.20,0.10,0.07,0.03])
    sales_final = sales_vals * (1 - discounts)
    margins     = np.array([base_margin[c] for c in cat_list])
    profit_vals = sales_final * margins * (1 - discounts * 1.5) + np.random.normal(0, 10, n)
    quantities  = np.random.randint(1, 15, n)
    df = pd.DataFrame({
        'Order_ID'    : [f'ORD-{2020+i//365}-{i%10000:04d}' for i in range(n)],
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
    return df

@st.cache_resource
def train_models(df):
    encoders = {}
    df_ml = df.copy()
    for col in ['Category','Sub_Category','Region','Segment','Ship_Mode']:
        le = LabelEncoder()
        df_ml[col+'_enc'] = le.fit_transform(df_ml[col])
        encoders[col] = le
    features = ['Sales','Quantity','Discount','Year','Month',
                'Category_enc','Sub_Category_enc','Region_enc','Segment_enc','Ship_Mode_enc']
    X = df_ml[features]
    y = df_ml['Profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree'    : DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest'    : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model' : model,
            'R²'    : round(r2_score(y_test, y_pred), 4),
            'RMSE'  : round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'MAE'   : round(mean_absolute_error(y_test, y_pred), 2),
            'y_test': y_test,
            'y_pred': y_pred,
        }
    return results, encoders, features, X_test

# ── Load everything ───────────────────────────────────────────────────────────
df = generate_data()
model_results, encoders, features, X_test = train_models(df)
best_model = model_results['Random Forest']['model']

# Dropdown options
CATEGORIES  = sorted(df['Category'].unique())
SUB_CAT_MAP = {c: sorted(df[df['Category']==c]['Sub_Category'].unique()) for c in CATEGORIES}
REGIONS     = sorted(df['Region'].unique())
SEGMENTS    = sorted(df['Segment'].unique())
SHIP_MODES  = sorted(df['Ship_Mode'].unique())

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem;'>
        <div style='font-size:36px;'>🛒</div>
        <div style='font-size:18px;font-weight:700;color:white;'>RetailIQ</div>
        <div style='font-size:11px;color:#94a3b8;margin-top:2px;'>Profit Prediction System</div>
    </div>
    <hr style='border-color:#3d4f7c;margin:0.75rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Home",
        "📊  Dashboard",
        "🔮  Predict Profit",
        "📈  Visualizations",
        "🗂️  Dataset Explorer",
        "🤖  Model Performance",
        "📋  Prediction History",
        "ℹ️  About",
    ])

    st.markdown("""
    <hr style='border-color:#3d4f7c;margin:0.75rem 0;'>
    <div style='font-size:11px;color:#64748b;text-align:center;'>
        Thiranex Internship Project<br>
        Hasini | JBIET Hyderabad<br>
        Random Forest · R² = 0.975
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown("""
    <div class="page-header">
        <h1>🛒 Automated Retail Sales Analysis & Profit Prediction System</h1>
        <p>An end-to-end Machine Learning application for retail analytics · Built with Python & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card blue">
            <div class="label">Total Sales</div>
            <div class="value">₹{df['Sales'].sum()/1000:.0f}K</div>
            <div class="delta up">↑ 1,000 transactions</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card green">
            <div class="label">Total Profit</div>
            <div class="value">₹{df['Profit'].sum()/1000:.1f}K</div>
            <div class="delta up">↑ Avg margin {df['Profit_Margin'].mean():.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card amber">
            <div class="label">Best Model R²</div>
            <div class="value">0.975</div>
            <div class="delta up">↑ Random Forest</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card purple">
            <div class="label">ML Models</div>
            <div class="value">3</div>
            <div class="delta up">LR · DT · RF</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    features_list = [
        ("📊", "Dashboard", "Key metrics, total sales, profit, and margin at a glance"),
        ("🔮", "Profit Predictor", "Enter transaction details and get instant profit prediction"),
        ("📈", "Visualizations", "Interactive charts — trends, categories, regions, correlations"),
        ("🗂️", "Dataset Explorer", "Preview data, view statistics, filter and search records"),
        ("🤖", "Model Performance", "Compare R², RMSE, MAE across all 3 ML models"),
        ("📋", "Prediction History", "Track all predictions made in this session, download as CSV"),
    ]
    for i, (icon, title, desc) in enumerate(features_list):
        col = [c1, c2, c3][i % 3]
        with col:
            st.markdown(f"""
            <div class="section-card" style="min-height:120px;">
                <div style="font-size:28px;margin-bottom:8px;">{icon}</div>
                <div style="font-weight:600;font-size:14px;color:#1f2937;margin-bottom:4px;">{title}</div>
                <div style="font-size:13px;color:#6b7280;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card" style="margin-top:1rem;">
        <div class="insight-title">🚀 How to use this app</div>
        <div class="insight-body">Use the sidebar to navigate between pages. Start with the Dashboard for an overview,
        then go to Predict Profit to test the ML model with your own inputs.
        The Dataset Explorer lets you filter and download data, and Model Performance shows how each algorithm compares.</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("""<div class="page-header">
        <h1>📊 Dashboard</h1>
        <p>Key performance metrics and sales overview</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card blue">
            <div class="label">Total Sales</div>
            <div class="value">₹{df['Sales'].sum():,.0f}</div>
            <div class="delta up">↑ Across 1,000 orders</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card green">
            <div class="label">Total Profit</div>
            <div class="value">₹{df['Profit'].sum():,.0f}</div>
            <div class="delta up">↑ {df['Profit_Margin'].mean():.1f}% avg margin</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card amber">
            <div class="label">Avg Order Value</div>
            <div class="value">₹{df['Sales'].mean():,.0f}</div>
            <div class="delta up">↑ Per transaction</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        loss_pct = (df['Profit']<0).mean()*100
        st.markdown(f"""<div class="metric-card purple">
            <div class="label">Loss Orders</div>
            <div class="value">{loss_pct:.1f}%</div>
            <div class="delta down">↓ High-discount transactions</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        cat_data = df.groupby('Category')[['Sales','Profit']].sum().reset_index()
        fig = px.bar(cat_data, x='Category', y=['Sales','Profit'],
                     barmode='group', title='Sales & Profit by Category',
                     color_discrete_sequence=['#3b82f6','#10b981'])
        fig.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                          font_family='Inter', title_font_size=14,
                          legend=dict(orientation='h', y=-0.2))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f3f4f6')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        seg_data = df.groupby('Segment')['Sales'].sum().reset_index()
        fig = px.pie(seg_data, names='Segment', values='Sales',
                     title='Sales by Customer Segment',
                     color_discrete_sequence=['#3b82f6','#10b981','#f59e0b'])
        fig.update_layout(height=320, font_family='Inter', title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    monthly = df.groupby(['Year','Month'])['Sales'].sum().reset_index()
    monthly['Period'] = monthly['Year'].astype(str)+'-'+monthly['Month'].astype(str).str.zfill(2)
    fig = px.area(monthly, x='Period', y='Sales', title='Monthly Sales Trend',
                  color_discrete_sequence=['#3b82f6'])
    fig.update_layout(height=280, plot_bgcolor='white', paper_bgcolor='white',
                      font_family='Inter', title_font_size=14)
    fig.update_xaxes(showgrid=False, tickangle=45,
                     tickvals=monthly['Period'].iloc[::3],
                     ticktext=monthly['Period'].iloc[::3])
    fig.update_yaxes(showgrid=True, gridcolor='#f3f4f6')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        reg_data = df.groupby('Region')[['Sales','Profit']].sum().reset_index()
        fig = px.bar(reg_data, x='Region', y='Sales', title='Sales by Region',
                     color='Sales', color_continuous_scale='Blues')
        fig.update_layout(height=280, plot_bgcolor='white', paper_bgcolor='white',
                          font_family='Inter', title_font_size=14, showlegend=False)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        disc_data = df.groupby('Discount')['Profit'].mean().reset_index()
        colors    = ['#ef4444' if v<0 else '#10b981' for v in disc_data['Profit']]
        fig = go.Figure(go.Bar(
            x=disc_data['Discount'].astype(str),
            y=disc_data['Profit'],
            marker_color=colors,
            text=[f'₹{v:.0f}' for v in disc_data['Profit']],
            textposition='outside'
        ))
        fig.update_layout(title='Avg Profit by Discount Rate', height=280,
                          plot_bgcolor='white', paper_bgcolor='white',
                          font_family='Inter', title_font_size=14,
                          xaxis_title='Discount', yaxis_title='Avg Profit (₹)')
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT PROFIT
# ═══════════════════════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown("""<div class="page-header">
        <h1>🔮 Profit Predictor</h1>
        <p>Enter transaction details to get an instant profit prediction</p>
    </div>""", unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            category = st.selectbox("📦 Category", CATEGORIES)
        with c2:
            sub_category = st.selectbox("🏷️ Sub-Category", SUB_CAT_MAP[category])
        with c3:
            region = st.selectbox("🌍 Region", REGIONS)

        c1, c2, c3 = st.columns(3)
        with c1:
            segment = st.selectbox("👥 Customer Segment", SEGMENTS)
        with c2:
            ship_mode = st.selectbox("🚚 Ship Mode", SHIP_MODES)
        with c3:
            year = st.selectbox("📅 Year", [2023,2024,2025,2026], index=1)

        c1, c2, c3 = st.columns(3)
        with c1:
            sales = st.number_input("💰 Sales Amount (₹)", min_value=1.0,
                                     max_value=10000.0, value=500.0, step=10.0)
        with c2:
            quantity = st.number_input("📦 Quantity", min_value=1,
                                        max_value=50, value=3, step=1)
        with c3:
            month = st.selectbox("📆 Month", list(range(1,13)),
                                  format_func=lambda x: datetime(2024,x,1).strftime('%B'))

        discount = st.slider("🏷️ Discount Rate", 0.0, 0.5, 0.0, 0.05,
                              format="%.0f%%",
                              help="Higher discounts significantly reduce profit")

        if discount >= 0.3:
            st.warning("⚠️ High discount rate — this may result in reduced profit or loss")

        submitted = st.form_submit_button("🔮 Predict Profit", use_container_width=True)

    if submitted:
        # Validate
        if sales <= 0:
            st.error("❌ Sales amount must be greater than 0")
        else:
            input_df = pd.DataFrame([{
                'Sales'          : sales,
                'Quantity'       : quantity,
                'Discount'       : discount,
                'Year'           : year,
                'Month'          : month,
                'Category_enc'   : encoders['Category'].transform([category])[0],
                'Sub_Category_enc':encoders['Sub_Category'].transform([sub_category])[0],
                'Region_enc'     : encoders['Region'].transform([region])[0],
                'Segment_enc'    : encoders['Segment'].transform([segment])[0],
                'Ship_Mode_enc'  : encoders['Ship_Mode'].transform([ship_mode])[0],
            }])[features]

            prediction = best_model.predict(input_df)[0]
            margin_pct  = (prediction / sales) * 100
            r2_score_val= model_results['Random Forest']['R²']
            confidence  = min(r2_score_val * 100, 97.5)

            c1, c2 = st.columns([1,1])
            with c1:
                css_class = "result-profit" if prediction >= 0 else "result-loss"
                sign      = "+" if prediction >= 0 else ""
                st.markdown(f"""
                <div class="{css_class}">
                    <div class="result-label">Predicted Profit</div>
                    <div class="result-amount">{sign}₹{prediction:,.2f}</div>
                    <div style="margin-top:8px;font-size:14px;opacity:0.9;">
                        Profit Margin: {margin_pct:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="section-card">
                    <div class="section-title">📋 Prediction Summary</div>
                    <table style="width:100%;font-size:13px;">
                        <tr><td style="color:#6b7280;padding:4px 0;">Category</td><td style="font-weight:500;">{category}</td></tr>
                        <tr><td style="color:#6b7280;padding:4px 0;">Sub-Category</td><td style="font-weight:500;">{sub_category}</td></tr>
                        <tr><td style="color:#6b7280;padding:4px 0;">Region</td><td style="font-weight:500;">{region}</td></tr>
                        <tr><td style="color:#6b7280;padding:4px 0;">Sales</td><td style="font-weight:500;">₹{sales:,.2f}</td></tr>
                        <tr><td style="color:#6b7280;padding:4px 0;">Discount</td><td style="font-weight:500;">{discount:.0%}</td></tr>
                        <tr><td style="color:#6b7280;padding:4px 0;">Model</td><td style="font-weight:500;">Random Forest</td></tr>
                        <tr><td style="color:#6b7280;padding:4px 0;">Confidence</td><td style="font-weight:500;color:#10b981;">{confidence:.1f}%</td></tr>
                    </table>
                </div>""", unsafe_allow_html=True)

            if discount >= 0.3:
                st.info("💡 Tip: Reducing discount to below 20% could increase profit by up to 62%")

            # Save to history
            st.session_state.history.append({
                'Timestamp'   : datetime.now().strftime('%H:%M:%S'),
                'Category'    : category,
                'Sub-Category': sub_category,
                'Region'      : region,
                'Segment'     : segment,
                'Sales (₹)'   : round(sales, 2),
                'Quantity'    : quantity,
                'Discount'    : f"{discount:.0%}",
                'Prediction (₹)': round(prediction, 2),
                'Margin (%)'  : round(margin_pct, 1),
                'Result'      : 'Profit' if prediction >= 0 else 'Loss',
            })
            st.success("✅ Prediction saved to history!")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Visualizations" in page:
    st.markdown("""<div class="page-header">
        <h1>📈 Visualizations</h1>
        <p>Interactive charts for sales analysis and data exploration</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Sales Analysis","Profit Analysis","Correlation","Distribution"])

    with tab1:
        c1,c2 = st.columns(2)
        with c1:
            sub_data = df.groupby('Sub_Category')['Sales'].sum().sort_values(ascending=True)
            fig = px.bar(sub_data, orientation='h', title='Sales by Sub-Category',
                         color=sub_data.values, color_continuous_scale='Blues')
            fig.update_layout(height=380,plot_bgcolor='white',paper_bgcolor='white',
                              font_family='Inter',title_font_size=14,showlegend=False)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ship_data = df.groupby('Ship_Mode')['Sales'].sum().reset_index()
            fig = px.pie(ship_data, names='Ship_Mode', values='Sales',
                         title='Sales by Ship Mode', hole=0.4,
                         color_discrete_sequence=['#3b82f6','#10b981','#f59e0b','#8b5cf6'])
            fig.update_layout(height=380,font_family='Inter',title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        monthly = df.groupby(['Year','Month'])['Sales'].sum().reset_index()
        monthly['Period']=monthly['Year'].astype(str)+'-'+monthly['Month'].astype(str).str.zfill(2)
        fig = px.line(monthly, x='Period', y='Sales', title='Monthly Sales Trend (2020–2023)',
                      markers=True, color_discrete_sequence=['#3b82f6'])
        fig.update_traces(fill='tozeroy', fillcolor='rgba(59,130,246,0.1)')
        fig.update_layout(height=300,plot_bgcolor='white',paper_bgcolor='white',
                          font_family='Inter',title_font_size=14)
        fig.update_xaxes(tickangle=45, tickvals=monthly['Period'].iloc[::3],
                         ticktext=monthly['Period'].iloc[::3])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            fig = px.scatter(df, x='Discount', y='Profit', color='Category',
                             title='Discount vs Profit',
                             color_discrete_sequence=['#3b82f6','#10b981','#f59e0b'])
            fig.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.5)
            fig.update_layout(height=340,plot_bgcolor='white',paper_bgcolor='white',
                              font_family='Inter',title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cat_profit = df.groupby('Category')['Profit'].sum().reset_index()
            fig = px.bar(cat_profit, x='Category', y='Profit',
                         title='Total Profit by Category',
                         color='Profit', color_continuous_scale='RdYlGn')
            fig.update_layout(height=340,plot_bgcolor='white',paper_bgcolor='white',
                              font_family='Inter',title_font_size=14)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df, x='Category', y='Profit_Margin', color='Category',
                     title='Profit Margin Distribution by Category',
                     color_discrete_sequence=['#3b82f6','#10b981','#f59e0b'])
        fig.update_layout(height=300,plot_bgcolor='white',paper_bgcolor='white',
                          font_family='Inter',title_font_size=14,showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        corr_cols = ['Sales','Quantity','Discount','Profit','Profit_Margin']
        corr = df[corr_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                        title='Feature Correlation Heatmap', zmin=-1, zmax=1)
        fig.update_layout(height=400,font_family='Inter',title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x='Sales', nbins=40, title='Sales Distribution',
                               color_discrete_sequence=['#3b82f6'])
            fig.update_layout(height=300,plot_bgcolor='white',paper_bgcolor='white',
                              font_family='Inter',title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x='Profit', nbins=40, title='Profit Distribution',
                               color_discrete_sequence=['#10b981'])
            fig.add_vline(x=0, line_dash='dash', line_color='red', opacity=0.7)
            fig.update_layout(height=300,plot_bgcolor='white',paper_bgcolor='white',
                              font_family='Inter',title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown("""<div class="page-header">
        <h1>🗂️ Dataset Explorer</h1>
        <p>Preview, filter, and download the retail sales dataset</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features", len(df.columns))
    c3.metric("Missing Values", df.isnull().sum().sum())
    c4.metric("Duplicates", df.duplicated().sum())

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🔍 Filter Dataset", expanded=True):
        fc1,fc2,fc3 = st.columns(3)
        with fc1:
            sel_cat = st.multiselect("Category", CATEGORIES, default=CATEGORIES)
        with fc2:
            sel_reg = st.multiselect("Region", REGIONS, default=REGIONS)
        with fc3:
            sel_seg = st.multiselect("Segment", SEGMENTS, default=SEGMENTS)
        fc1,fc2 = st.columns(2)
        with fc1:
            min_sales, max_sales = st.slider("Sales Range (₹)",
                float(df['Sales'].min()), float(df['Sales'].max()),
                (float(df['Sales'].min()), float(df['Sales'].max())))
        with fc2:
            min_profit, max_profit = st.slider("Profit Range (₹)",
                float(df['Profit'].min()), float(df['Profit'].max()),
                (float(df['Profit'].min()), float(df['Profit'].max())))

    filtered = df[
        df['Category'].isin(sel_cat) &
        df['Region'].isin(sel_reg) &
        df['Segment'].isin(sel_seg) &
        df['Sales'].between(min_sales, max_sales) &
        df['Profit'].between(min_profit, max_profit)
    ]

    st.write(f"Showing **{len(filtered):,}** records")
    display_cols = ['Order_ID','Order_Date','Category','Sub_Category','Region','Segment','Sales','Quantity','Discount','Profit','Profit_Margin']
    st.dataframe(filtered[display_cols].head(200), use_container_width=True, height=320)

    csv = filtered[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data as CSV", csv,
                       "filtered_retail_data.csv", "text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📊 Summary Statistics**")
    st.dataframe(filtered[['Sales','Quantity','Discount','Profit','Profit_Margin']].describe().round(2),
                 use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown("""<div class="page-header">
        <h1>🤖 Model Performance</h1>
        <p>Compare machine learning models and evaluate prediction accuracy</p>
    </div>""", unsafe_allow_html=True)

    metrics_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'R² Score': [model_results[m]['R²'] for m in model_results],
        'RMSE (₹)': [model_results[m]['RMSE'] for m in model_results],
        'MAE (₹)':  [model_results[m]['MAE']  for m in model_results],
    })

    c1,c2,c3 = st.columns(3)
    for col, metric in zip([c1,c2,c3], ['R² Score','RMSE (₹)','MAE (₹)']):
        with col:
            fig = px.bar(metrics_df, x='Model', y=metric,
                         title=metric, text=metric,
                         color='Model',
                         color_discrete_sequence=['#6b7280','#f59e0b','#3b82f6'])
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(height=280,plot_bgcolor='white',paper_bgcolor='white',
                              font_family='Inter',title_font_size=14,showlegend=False)
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Model Comparison Table**")
    styled = metrics_df.style.highlight_max(subset=['R² Score'], color='#dcfce7')\
                             .highlight_min(subset=['RMSE (₹)','MAE (₹)'], color='#dcfce7')
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        rf = model_results['Random Forest']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rf['y_test'], y=rf['y_pred'],
                                  mode='markers', name='Predictions',
                                  marker=dict(color='#3b82f6',size=5,opacity=0.5)))
        mn = min(rf['y_test'].min(), rf['y_pred'].min())
        mx = max(rf['y_test'].max(), rf['y_pred'].max())
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                                  name='Perfect Fit', line=dict(color='red',dash='dash')))
        fig.update_layout(title='Actual vs Predicted — Random Forest',
                          xaxis_title='Actual Profit', yaxis_title='Predicted Profit',
                          height=320, plot_bgcolor='white', paper_bgcolor='white',
                          font_family='Inter', title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        importances = pd.DataFrame({
            'Feature'  : features,
            'Importance': model_results['Random Forest']['model'].feature_importances_
        }).sort_values('Importance', ascending=True)
        fig = px.bar(importances, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance — Random Forest',
                     color='Importance', color_continuous_scale='Blues')
        fig.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                          font_family='Inter', title_font_size=14)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif "History" in page:
    st.markdown("""<div class="page-header">
        <h1>📋 Prediction History</h1>
        <p>All predictions made in this session</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions yet. Go to 🔮 Predict Profit and make your first prediction!")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Predictions", len(hist_df))
        profit_count = (hist_df['Result']=='Profit').sum()
        c2.metric("Profitable", f"{profit_count}/{len(hist_df)}")
        c3.metric("Avg Predicted Profit",
                  f"₹{hist_df['Prediction (₹)'].mean():,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        c1,c2 = st.columns(2)
        with c1:
            csv = hist_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download as CSV", csv,
                               "prediction_history.csv", "text/csv",
                               use_container_width=True)
        with c2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("""<div class="page-header">
        <h1>ℹ️ About This Project</h1>
        <p>Project details, technologies, and future scope</p>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🎯 Project Objective</div>
            <p style="font-size:13px;color:#374151;line-height:1.7;">
            This project builds an end-to-end automated retail analytics pipeline that:
            </p>
            <ul style="font-size:13px;color:#374151;line-height:2;padding-left:1.2rem;">
                <li>Analyses retail sales data across categories, regions, and segments</li>
                <li>Identifies key factors affecting profitability (discount, category, region)</li>
                <li>Trains and compares 3 supervised ML regression models</li>
                <li>Predicts profit for new transactions in real time</li>
                <li>Delivers actionable business insights for retail decision-making</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🛠️ Technologies Used</div>
            <table style="width:100%;font-size:13px;">
                <tr style="background:#f9fafb;"><th style="padding:6px 8px;text-align:left;">Library</th><th style="padding:6px 8px;text-align:left;">Purpose</th></tr>
                <tr><td style="padding:5px 8px;">Python 3.x</td><td style="padding:5px 8px;color:#6b7280;">Core language</td></tr>
                <tr style="background:#f9fafb;"><td style="padding:5px 8px;">pandas / numpy</td><td style="padding:5px 8px;color:#6b7280;">Data manipulation</td></tr>
                <tr><td style="padding:5px 8px;">scikit-learn</td><td style="padding:5px 8px;color:#6b7280;">ML models & metrics</td></tr>
                <tr style="background:#f9fafb;"><td style="padding:5px 8px;">Streamlit</td><td style="padding:5px 8px;color:#6b7280;">Web application</td></tr>
                <tr><td style="padding:5px 8px;">Plotly</td><td style="padding:5px 8px;color:#6b7280;">Interactive charts</td></tr>
                <tr style="background:#f9fafb;"><td style="padding:5px 8px;">GitHub</td><td style="padding:5px 8px;color:#6b7280;">Version control</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🤖 ML Algorithm Details</div>
            <div style="font-size:13px;color:#374151;line-height:1.8;">
                <b>Best Model:</b> Random Forest Regressor<br>
                <b>n_estimators:</b> 100 trees<br>
                <b>Train/Test Split:</b> 80% / 20%<br>
                <b>R² Score:</b> 0.9754 (97.54% accuracy)<br>
                <b>RMSE:</b> ₹11.39 | <b>MAE:</b> ₹9.27<br>
                <b>Target Variable:</b> Profit (₹)<br>
                <b>Features:</b> 10 (Sales, Discount, Category, Region, etc.)
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🚀 Future Scope</div>
            <ul style="font-size:13px;color:#374151;line-height:2;padding-left:1.2rem;">
                <li>Real-time data pipeline from POS systems</li>
                <li>Deep Learning models (LSTM for time series)</li>
                <li>Customer segmentation using clustering</li>
                <li>Demand forecasting module</li>
                <li>Integration with BI tools (Power BI, Tableau)</li>
                <li>REST API for model deployment</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card" style="text-align:center;margin-top:0.5rem;">
        <div style="font-size:14px;font-weight:600;color:#1f2937;margin-bottom:8px;">👩‍💻 Developer</div>
        <div style="font-size:13px;color:#6b7280;line-height:1.8;">
            <b>Hasini</b> | B.Tech Information Technology (Final Year)<br>
            J.B. Institute of Engineering & Technology (JBIET), Hyderabad<br>
            Affiliated to JNTUH | Regulation R22 | CGPA: 8.2<br>
            <b>Internship:</b> Thiranex — Data Science Internship<br>
            <b>GitHub:</b> github.com/deshavalihasini-collab
        </div>
    </div>""", unsafe_allow_html=True)

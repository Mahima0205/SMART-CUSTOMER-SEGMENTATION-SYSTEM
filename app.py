import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------
# SmartCart AI | Customer Intelligence & Segmentation Platform
# ------------------------------------------------------------

st.set_page_config(
    page_title="SmartCart AI | Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- ULTRA-MODERN STYLING ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', sans-serif; 
}
.stApp { 
    background: #090d16; 
    color: #f1f5f9;
}
.block-container { 
    padding-top: 1.8rem; 
    padding-bottom: 3rem; 
    max-width: 1550px; 
}

/* Stunning Glassmorphic Hero */
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
  padding: 36px 42px; 
  border-radius: 24px; 
  color: white; 
  margin-bottom: 26px;
  box-shadow: 0 20px 50px -12px rgba(49, 46, 129, 0.35);
  border: 1px solid rgba(255, 255, 255, 0.08);
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: "";
  position: absolute;
  top: -50%;
  right: -20%;
  width: 400px;
  height: 400px;
  background: radial-gradient(circle, rgba(99,102,241,0.2) 0%, rgba(0,0,0,0) 70%);
  border-radius: 50%;
  z-index: 1;
}
.hero > * { position: relative; z-index: 2; }
.hero h1 { 
    font-size: 2.3rem; 
    font-weight: 800; 
    margin: 4px 0 6px 0; 
    color: #ffffff; 
    letter-spacing: -0.02em;
}
.hero p { 
    color: #94a3b8; 
    margin: 0; 
    font-size: 1.05rem; 
}
.eyebrow { 
    color: #818cf8; 
    text-transform: uppercase; 
    letter-spacing: 3px; 
    font-size: 0.72rem; 
    font-weight: 800; 
}

/* Section Headings */
.section-title { 
    font-weight: 800; 
    color: #f8fafc; 
    font-size: 1.35rem; 
    margin: 20px 0 14px 0; 
    letter-spacing: -0.01em;
}

/* Custom Styled Metrics */
div[data-testid="stMetric"] { 
    background: #111827; 
    border: 1px solid #1f2937; 
    padding: 18px 20px; 
    border-radius: 16px; 
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: #4f46e5;
}
div[data-testid="stMetricLabel"] { 
    color: #9ca3af !important; 
    font-weight: 600;
    font-size: 0.83rem;
}
div[data-testid="stMetricValue"] { 
    color: #f3f4f6 !important; 
    font-weight: 800;
    font-size: 1.85rem !important;
}

/* Tab Navigation Styling */
.stTabs [data-baseweb="tab-list"] { 
    gap: 8px; 
    background-color: #111827;
    padding: 6px;
    border-radius: 14px;
    border: 1px solid #1f2937;
}
.stTabs [data-baseweb="tab"] { 
    background: transparent; 
    border-radius: 10px; 
    padding: 10px 20px; 
    font-weight: 600;
    color: #9ca3af;
    border: none !important;
}
.stTabs [aria-selected="true"] { 
    background: #4f46e5 !important; 
    color: #ffffff !important; 
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
}

/* Dataframe Card Polish */
div[data-testid="stDataFrame"] { 
    border-radius: 14px; 
    overflow: hidden; 
    border: 1px solid #1f2937;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

/* Sidebar Dark Theme Refinement */
section[data-testid="stSidebar"] {
    background-color: #0b0f19;
    border-right: 1px solid #1f2937;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #f8fafc;
    font-size: 1.2rem;
    font-weight: 700;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 10px;
}
section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Customer Analytics • Machine Learning • Decision Support</div>
  <h1>🛍️ SmartCart AI</h1>
  <p>Customer Intelligence & Segmentation Platform — turn customer data into actionable retail strategies.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------- helpers ---------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes, file_name):
    import io
    return pd.read_csv(io.BytesIO(file_bytes))

def engineer_features(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Income" in df.columns:
        df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
        df["Income"] = df["Income"].fillna(df["Income"].median())

    if "Year_Birth" in df.columns and "Age" not in df.columns:
        birth = pd.to_numeric(df["Year_Birth"], errors="coerce")
        df["Age"] = pd.Timestamp.today().year - birth

    if "Dt_Customer" in df.columns:
        dt = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
        ref = dt.max()
        if pd.notna(ref):
            df["Customer_Tenure_Days"] = (ref - dt).dt.days

    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    present_spend = [c for c in spend_cols if c in df.columns]
    if present_spend:
        df["Total_Spending"] = df[present_spend].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

    child_cols = [c for c in ["Kidhome", "Teenhome"] if c in df.columns]
    if child_cols:
        df["Total_Children"] = df[child_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

    if "Education" in df.columns:
        df["Education"] = df["Education"].replace({
            "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
            "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
        })
    if "Marital_Status" in df.columns and "Living_With" not in df.columns:
        df["Living_With"] = df["Marital_Status"].replace({
            "Married": "Partner", "Together": "Partner",
            "Single": "Alone", "Divorced": "Alone", "Widow": "Alone",
            "Absurd": "Alone", "YOLO": "Alone"
        })

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df[(df["Age"].isna()) | ((df["Age"] >= 18) & (df["Age"] < 90))]
    if "Income" in df.columns:
        df = df[(df["Income"].isna()) | (df["Income"] < 600_000)]

    drop_cols = ["ID", "Year_Birth", "Marital_Status", "Dt_Customer",
                 "Kidhome", "Teenhome", "Z_CostContact", "Z_Revenue"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df

def make_encoder(numeric_cols, categorical_cols):
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ], remainder="drop")

# --------------------------- sidebar ---------------------------
with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    st.caption("Configure the segmentation run")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    default_path = "smartcart_customers.csv"
    if uploaded is not None:
        try:
            raw_df = load_csv(uploaded.getvalue(), uploaded.name)
            st.success(f"Loaded `{uploaded.name}`")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
    elif os.path.exists(default_path):
        raw_df = pd.read_csv(default_path)
        st.info(f"Using `{default_path}`")
    else:
        st.warning("Upload `smartcart_customers.csv` to begin.")
        st.stop()

    st.divider()
    algorithm = st.selectbox("Clustering algorithm", ["K-Means", "Agglomerative"], index=0)
    k = st.slider("Number of customer segments", min_value=2, max_value=8, value=4)
    pca_dims = st.selectbox("PCA visualization dimensions", [2, 3], index=1)
    st.caption("PCA is used for visualization and clustering, matching the notebook workflow.")

# --------------------------- processing ---------------------------
original_rows = len(raw_df)
df_features = engineer_features(raw_df)
if df_features.empty or len(df_features) < 3:
    st.error("Not enough valid rows after preprocessing. Check your CSV.")
    st.stop()

excluded = [c for c in ["Response", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                         "AcceptedCmp4", "AcceptedCmp5", "Complain", "NumDealsPurchases"]
            if c in df_features.columns]
feature_df = df_features.drop(columns=excluded, errors="ignore")
feature_df = feature_df.dropna(axis=1, how="all")
numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = feature_df.select_dtypes(exclude=np.number).columns.tolist()

try:
    preprocessor = make_encoder(numeric_cols, categorical_cols)
    X_scaled = preprocessor.fit_transform(feature_df)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

n_components = min(3, X_scaled.shape[1], X_scaled.shape[0] - 1)
pca = PCA(n_components=n_components, random_state=42)
X_pca_all = pca.fit_transform(X_scaled)
X_model = X_pca_all[:, :min(pca_dims, X_pca_all.shape[1])]

k = min(k, len(feature_df) - 1)
if algorithm == "K-Means":
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_model)
else:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_model)

results = raw_df.loc[df_features.index].copy()
results = results.loc[feature_df.index].copy()
results["Customer_Segment"] = labels
results["Customer_Segment"] = results["Customer_Segment"].astype(int)

try:
    sil = silhouette_score(X_model, labels) if len(np.unique(labels)) > 1 else 0
except Exception:
    sil = 0
explained = float(pca.explained_variance_ratio_.sum())

# --------------------------- overview ---------------------------
st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers analysed", f"{len(results):,}", f"{len(results)-original_rows:+,} after cleaning" if len(results)!=original_rows else "All rows retained")
c2.metric("Customer segments", f"{len(np.unique(labels))}", algorithm)
c3.metric("Silhouette score", f"{sil:.3f}", "Higher is better")
c4.metric("PCA variance captured", f"{explained:.1%}", f"{n_components} components")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🧩 Segment Explorer", "🧠 Model Diagnostics", "👥 Customer Profiles", "⬇️ Export & Report"
])

# Vibrant modern palette tuned for dark mode
chart_colors = ["#6366f1", "#10b981", "#f59e0b", "#ec4899", "#3b82f6", "#8b5cf6", "#14b8a6", "#ef4444"]

with tab1:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Segment distribution")
        counts = pd.Series(labels).value_counts().sort_index().rename_axis("Segment").reset_index(name="Customers")
        counts["Segment"] = counts["Segment"].astype(str).map(lambda x: f"Segment {x}")
        fig = px.bar(counts, x="Segment", y="Customers", color="Segment",
                     text="Customers", title="Customers in each segment",
                     color_discrete_sequence=chart_colors)
        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=10,r=10,t=45,b=10),
            height=360,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### What this run tells you")
        st.info(
            f"The current run created **{len(np.unique(labels))} segments** using **{algorithm}**. "
            f"The silhouette score is **{sil:.3f}**. Use segment profiles—not this score alone—to decide "
            "whether the groups are useful for business decisions."
        )
        st.markdown("**Suggested retail use cases**")
        st.markdown("- Personalised product recommendations\n- Campaign and loyalty targeting\n- High-value customer retention\n- Discount strategy and cross-selling")
    st.markdown("### Dataset health")
    h1, h2, h3 = st.columns(3)
    missing_pct = raw_df.isna().sum().sum() / max(raw_df.size, 1) * 100
    h1.metric("Source columns", raw_df.shape[1])
    h2.metric("Missing cells", f"{raw_df.isna().sum().sum():,}")
    h3.metric("Missing cell rate", f"{missing_pct:.2f}%")
    with st.expander("Preview source data"):
        st.dataframe(raw_df.head(20), use_container_width=True)

with tab2:
    st.markdown("### Explore the customer map")
    st.caption("Each point represents a customer. Nearby points have similar encoded profiles in the selected PCA space.")
    if X_pca_all.shape[1] >= 3:
        viz_df = pd.DataFrame(X_pca_all[:, :3], columns=["PCA 1", "PCA 2", "PCA 3"], index=feature_df.index)
    else:
        viz_df = pd.DataFrame(X_pca_all[:, :2], columns=["PCA 1", "PCA 2"], index=feature_df.index)
    viz_df["Segment"] = labels.astype(str)
    if "Income" in results.columns:
        viz_df["Income"] = pd.to_numeric(results["Income"], errors="coerce").values
    if "Total_Spending" in results.columns:
        viz_df["Total Spending"] = pd.to_numeric(results["Total_Spending"], errors="coerce").values
    if pca_dims == 3 and "PCA 3" in viz_df.columns:
        fig = px.scatter_3d(viz_df, x="PCA 1", y="PCA 2", z="PCA 3", color="Segment",
                            hover_data=[c for c in ["Income", "Total Spending"] if c in viz_df.columns],
                            opacity=0.82, title="3D customer segmentation map",
                            color_discrete_sequence=chart_colors)
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=45,b=0), paper_bgcolor="rgba(0,0,0,0)")
    else:
        fig = px.scatter(viz_df, x="PCA 1", y="PCA 2", color="Segment",
                         hover_data=[c for c in ["Income", "Total Spending"] if c in viz_df.columns],
                         opacity=0.82, title="2D customer segmentation map",
                         color_discrete_sequence=chart_colors)
        fig.update_layout(template="plotly_dark", height=560, margin=dict(l=0,r=0,t=45,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Clustering quality and PCA")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Explained variance by component**")
        variance_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "Explained variance": pca.explained_variance_ratio_
        })
        fig = px.bar(variance_df, x="Component", y="Explained variance", text=variance_df["Explained variance"].map(lambda x:f"{x:.1%}"), color_discrete_sequence=["#6366f1"])
        fig.update_layout(template="plotly_dark", height=340, yaxis_tickformat=".0%", margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        st.markdown("**Silhouette interpretation**")
        st.write("The silhouette score ranges from -1 to 1. Higher values usually indicate better-separated clusters.")
        st.metric("Current silhouette score", f"{sil:.4f}")
        if sil >= .5:
            st.success("Strong separation in the current feature space.")
        elif sil >= .25:
            st.info("Some useful structure may exist, but inspect the profiles.")
        else:
            st.warning("Segments overlap. Try adjusting cluster count or feature parameters.")
    st.markdown("**Model configuration**")
    st.json({
        "algorithm": algorithm,
        "segments_requested": int(k),
        "encoded_features": int(X_scaled.shape[1]),
        "PCA_components_available": int(n_components),
        "PCA_variance_captured": round(explained, 4),
    })

with tab4:
    st.markdown("### Understand each segment")
    profile = results.copy()
    profile["Customer_Segment"] = profile["Customer_Segment"].astype(str)
    numeric_profile_cols = profile.select_dtypes(include=np.number).columns.tolist()
    numeric_profile_cols = [c for c in numeric_profile_cols if c.lower() not in ["id", "customer_id"]]
    if numeric_profile_cols:
        summary = profile.groupby("Customer_Segment")[numeric_profile_cols].mean(numeric_only=True).round(2)
        st.dataframe(summary, use_container_width=True)
        selected_segment = st.selectbox("Inspect segment", sorted(profile["Customer_Segment"].unique(), key=lambda x:int(x)))
        seg = profile[profile["Customer_Segment"] == selected_segment]
        st.markdown(f"#### Segment {selected_segment} snapshot")
        a, b, c = st.columns(3)
        a.metric("Customers", f"{len(seg):,}", f"{len(seg)/len(profile):.1%} of dataset")
        if "Income" in seg.columns:
            a.metric("Average income", f"{pd.to_numeric(seg['Income'], errors='coerce').mean():,.0f}")
        if "Total_Spending" in seg.columns:
            b.metric("Average total spending", f"{pd.to_numeric(seg['Total_Spending'], errors='coerce').mean():,.0f}")
        if "Age" in seg.columns:
            c.metric("Average age", f"{pd.to_numeric(seg['Age'], errors='coerce').mean():.1f} years")
        
        cat_candidates = [c for c in ["Education", "Living_With", "Marital_Status"] if c in seg.columns]
        if cat_candidates:
            cat_col = st.selectbox("Categorical breakdown", cat_candidates)
            breakdown = seg[cat_col].fillna("Unknown").astype(str).value_counts().reset_index()
            breakdown.columns = [cat_col, "Customers"]
            fig = px.pie(breakdown, names=cat_col, values="Customers", hole=.55,
                         title=f"{cat_col} mix — Segment {selected_segment}", color_discrete_sequence=chart_colors)
            fig.update_layout(template="plotly_dark", height=350, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("### Download results")
    st.write("Export original customer rows with generated segment labels for business implementation.")
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download segmented customers (CSV)", data=csv_bytes,
                       file_name="smartcart_customer_segments.csv", mime="text/csv", type="primary")
    summary_csv = results.groupby("Customer_Segment").size().rename("Customer_Count").reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download segment counts (CSV)", data=summary_csv,
                       file_name="smartcart_segment_summary.csv", mime="text/csv")
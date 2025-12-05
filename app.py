import streamlit as st
import pandas as pd 
import numpy as np 
import joblib
import umap.umap_ as umap
from sklearn.cluster import KMeans

# ------------------------------
# 1. Load data & models
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/final_recommendation_system.csv")
    return df

@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    umap_model = joblib.load("models/umap.pkl")
    kmeans = joblib.load("models/kmeans_umap.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    cluster_names = joblib.load("models/cluster_names.pkl")
    actions = joblib.load("models/actions.pkl")
    cluster_to_learned_action = joblib.load("models/cluster_to_learned_action.pkl")
    return scaler, umap_model, kmeans, feature_columns, cluster_names, actions, cluster_to_learned_action

df = load_data()
scaler, umap_model, kmeans, feature_columns, cluster_names, actions, cluster_to_learned_action = load_models()

feature_columns = [
    c for c in feature_columns
    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
]
# Make sure Cluster_Name exists (in case)
if "Cluster_Name" not in df.columns and "Cluster" in df.columns:
    df["Cluster_Name"] = df["Cluster"].map(cluster_names) 


# Compute UMAP coordinates for all patients (for plotting)
@st.cache_data
def get_umap_embedding(df, feature_columns):
    X = df[feature_columns]
    X_umap = umap_model.transform(X)
    umap_df = pd.DataFrame(X_umap, columns = ["UMAP1", "UMAP2"])
    umap_df["Cluster"] = df["Cluster"]
    umap_df["Cluster_Name"] = df["Cluster_Name"]
    return umap_df

umap_df = get_umap_embedding(df, feature_columns)

# Precompute cluster-level summary
cluster_summary = (
    df.groupby("Cluster_Name").agg({
        "RIDAGEYR": "mean",
        "BMXBMI": "mean",
        "BPXSY1": "mean",
        "BPXDI1": "mean",
        "LBDLDL": "mean",
        "LBXTC": "mean",
        "DR1TKCAL": "mean",
        "DR1TSUGR": "mean",
        "DR1TTFAT": "mean",
        "num_meds": "mean",
        "any_med_use": "mean"
    }).reset_index()
)

cluster_top_action = (
    df.groupby("Cluster_Name")["RL_Recommended_Action"]
      .agg(lambda x: x.value_counts().idxmax())
      .reset_index()
      .rename(columns = {"RL_Recommended_Action": "Top_Recommended_Action"})
)

cluster_summary = cluster_summary.merge(cluster_top_action, on = "Cluster_Name", how = "left")


# For user simulation, we'll use medians as defaults
feature_medians = df[feature_columns].median()

# ------------------------------
# 2. Helper: predict cluster from user input
# ------------------------------
def predict_cluster_from_input(user_inputs: dict):
    """
    user_inputs: dictionary with keys like RIDAGEYR, RIAGENDR, BMXBMI, BPXSY1, BPXDI1, etc.
    """
    row = feature_medians.copy()

    for key, value in user_inputs.items():
        if key in row.index:
            row[key] = value

    X_user = pd.DataFrame([row[feature_columns]])

    X_user_umap = umap_model.transform(X_user)
    cluster_id = int(kmeans.predict(X_user_umap)[0])

    cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
    action_id = cluster_to_learned_action.get(cluster_id, None)
    action_text = actions.get(action_id, "No recommended action")

    return cluster_id, cluster_label, action_text


# ------------------------------
# 3. Streamlit layout
# ------------------------------
st.set_page_config(
    page_title = "Patient Segmentation and Preventive Health Dashboard",
    layout = "wide"
)

st.markdown("""
<style>
/* CSS Variables for Theme Support */
:root {
    --bg-primary: #f5f7fb;
    --bg-secondary: #ffffff;
    --text-primary: #111827;
    --text-secondary: #6b7280;
    --sidebar-bg: linear-gradient(180deg, #020617 0%, #111827 40%, #020617 100%);
    --sidebar-text: #f1f5f9;
    --sidebar-label-text: #f1f5f9;
    --card-shadow: rgba(15, 23, 42, 0.08);
    --metric-text: #ffffff;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --sidebar-bg: linear-gradient(180deg, #0f172a 0%, #1e293b 40%, #0f172a 100%);
        --sidebar-text: #e2e8f0;
        --card-shadow: rgba(0, 0, 0, 0.3);
        --metric-text: #f1f5f9;
    }
}

/* App background */
.stApp {
    background-color: var(--bg-primary);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    padding: 1.25rem 0.75rem;
}

/* Sidebar radio label - white in light mode */
[data-testid="stSidebar"] label {
    color: #ffffff;
    font-weight: 500;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #ffffff;
}

/* Sidebar text elements */
[data-testid="stSidebar"] p {
    color: #e5e7eb;
}

[data-testid="stSidebar"] div {
    color: #e5e7eb;
}

/* Dark mode adjustments for sidebar */
@media (prefers-color-scheme: dark) {
    [data-testid="stSidebar"] label {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] p {
        color: #cbd5e1;
    }
    
    [data-testid="stSidebar"] div {
        color: #cbd5e1;
    }
}

.sidebar-nav-title {
    color: #d1d5db;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.sidebar-nav-help {
    color: #9ca3af;
    font-size: 0.8rem;
    margin-bottom: 1.0rem;
}

@media (prefers-color-scheme: dark) {
    .sidebar-nav-title {
        color: #94a3b8;
    }
    
    .sidebar-nav-help {
        color: #64748b;
    }
}

/* Big title */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary);
}

/* Card-like containers */
.card {
    background-color: var(--bg-secondary);
    padding: 1rem 1.5rem;
    border-radius: 0.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 8px var(--card-shadow);
    color: var(--text-primary);
}

/* Section subtitles */
h3, h4 {
    color: var(--text-primary);
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: var(--metric-text);
    font-weight: 600;
}

/* General text and markdown */
p, div, span {
    color: var(--text-primary);
}
</style>
""", unsafe_allow_html = True)

st.title("🩺 Patient Segmentation System for Preventive Healthcare Planning")
st.markdown("### NHANES-based Unsupervised + Reinforcement Learning Recommendation System")

# Sidebar navigation
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-nav-title">Navigation</div>
        <div class="sidebar-nav-help">
            Explore segments, understand cluster profiles, and get personalized recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "",
        [
            "🏠 Overview",
            "🧬 Cluster Segmentation",
            "📊 Cluster Profiles",
            "👤 Personalized Recommendation",
        ]
    )

# ------------------------------
# Page 1: Overview
# ------------------------------
if page == "🏠 Overview":
    st.subheader("📌 Project Overview")

    st.markdown("""
    This dashboard showcases a **Patient Segmentation System** built using the NHANES dataset.

    **Techniques Used:**
    - Unsupervised Learning: UMAP + KMeans for patient clustering
    - Reinforcement Learning (Q-Learning): to learn the best preventive action per cluster
    - Multi-source data: Demographics, Labs, Examination, Diet, Medications, Questionairre
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Patients", f"{len(df):,}")
    with col2:
        st.metric("Number of Clusters", df["Cluster"].nunique())
    with col3:
        st.metric("Recommendation Types", len(actions))

    st.markdown("### Pipeline")
    st.markdown("""
    `Raw NHANES Data → Cleaning & Feature Engineering → UMAP(Dimensionality Reduction)
    → KMeans Clustering → Q-Learning Policy → RL-based Health Recommendations`
    """)


# ------------------------------
# Page 2: Cluster Segmentation
# ------------------------------
elif page == "🧬 Cluster Segmentation":
    st.subheader("🧬 Patient Segmentation Visualization (UMAP + KMeans)")

    st.markdown("""
    Each point represents a patient, projected into 2D using **UMAP** and 
    colored by their assigned health segment (cluster).
    """)

    all_custer_names = sorted(umap_df["Cluster_Name"].unique())
    selected_clusters = st.multiselect(
        "Filter clusters to display",
        options = all_custer_names,
        default = all_custer_names
    )

    plot_df = umap_df[umap_df["Cluster_Name"].isin(selected_clusters)]

    import matplotlib.pyplot as plt 

    st.markdown('<div class = "card>', unsafe_allow_html = True)
    st.markdown("### 📍 UMAP Projection of Patient Clusters")

    fig, ax = plt.subplots(figsize=(8,6))

    for cname in sorted(plot_df["Cluster_Name"].unique()):
        subset = plot_df[plot_df["Cluster_Name"] == cname]
        ax.scatter(subset["UMAP1"], subset["UMAP2"], s=10, label = cname, alpha = 0.7)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP-based Patient Segmentation")
    ax.legend(markerscale = 2, fontsize = 8, bbox_to_anchor = (1.05,1), loc = "upper left")
    st.pyplot(fig)
    st.markdown("<div>", unsafe_allow_html = True)

    st.markdown('<div class = "card>', unsafe_allow_html = True)
    st.markdown("### 📊 Cluster Size Distribution")

    cluster_counts = (
        df.groupby("Cluster_Name")["Cluster"]
          .count()
          .sort_values(ascending = False)
          .reset_index()
          .rename(columns = {"Cluster": "Count"})
    )
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.bar(cluster_counts["Cluster_Name"], cluster_counts["Count"])
    ax2.set_xticklabels(cluster_counts["Cluster_Name"], rotation = 45, ha = "right")
    ax2.set_ylabel("Number of Patients")
    ax2.set_title("Number of Patients per Cluster")

    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html = True)
# ------------------------------
# Page 3: Cluster Profiles
# ------------------------------
elif page == "📊 Cluster Profiles":
    st.subheader("📊 Cluster Profiles & Preventive Insights")

    cluster_choice = st.selectbox(
        "Select a cluster to inspect",
        sorted(cluster_summary["Cluster_Name"].unique())
    )

    cdata = cluster_summary[cluster_summary["Cluster_Name"] == cluster_choice].iloc[0]

    st.markdown(f"### 🧩 Cluster: **{cluster_choice}**")
    st.write(f"**Top RL Recommended Action:** {cdata['Top_Recommended_Action']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Age", f"{cdata['RIDAGEYR']:.1f} yrs")
        st.metric("Average BMI", f"{cdata['BMXBMI']:.1f}")
    with col2:
        st.metric("Average Systolic BP", f"{cdata['BPXSY1']:.1f} mmHg")
        st.metric("Average Diastolic BP", f"{cdata['BPXDI1']:.1f} mmHg")
    with col3:
        st.metric("Average LDL Cholestrol", f"{cdata['LBDLDL']:.1f} mm/dL")
        st.metric("Average Total Cholestrol", f"{cdata['LBXTC']:.1f} mm/dL")

    st.markdown("### Diet & Medication Patterns")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Average Calories", f"{cdata['DR1TKCAL']:.0f} kcal/day")
    with col5:
        st.metric("Average Sugar Intake", f"{cdata['DR1TSUGR']:.1f} g/day")
    with col6:
        st.metric("Average Fat Intake", f"{cdata['DR1TTFAT']:.1f} g/day")
    
    st.metric("Medication Usage (any_med_use)", f"{cdata['any_med_use'] * 100:.1f}% of patients on meds")

    st.markdown("### Cluster Summary Table")
    st.dataframe(cluster_summary.round(2))

# ------------------------------
# Page 4: Personalized Recommendation
# ------------------------------
elif page == "👤 Personalized Recommendation":
    st.subheader("👤 Personalized Preventive Recommendation")
    
    st.markdown("Fill in the deatils below to estimate your **health segment** and **recommended preventive action.**")

    with st.form("user_input_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", min_value = 1, max_value = 90, value = 40)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI", min_value = 10.0, max_value = 60.0, value = 25.0, step = 0.1)
            sbp = st.number_input("Systolic BP (BPXSY1)", min_value = 80.0, max_value = 220.0, value = 120.0)
        with col2:
            dbp = st.number_input("Diastolic BP (BPXDI1)", min_value = 80.0, max_value = 130.0, value = 80.0)
            calories = st.number_input("Daily calories (DR1TKCAL)", min_value = 500.0, max_value = 6000.0, value = 2000.0)
            sugar = st.number_input("Daily Sugar (DR1TSUGR, g)", min_value = 0.0, max_value = 500.0, value = 80.0)
            alcohol = st.number_input("Alcohol Intake (DR1TALCO, g/day)", min_value = 0.0, max_value = 300.0, value = 0.0)
        
        submitted = st.form_submit_button("Get Recommendation")

    if submitted:
        if "RIAGENDR" in feature_columns:
            if gender == "Male":
                gender_val = 0
            else:
                gender_val = 1
        else:
            gender_val = None

        user_inputs = {
            "RIDAGEYR": age,
            "RIAGENDR": gender_val,
            "BMXBMI": bmi,
            "BPXSY1": sbp,
            "BPXDI1": dbp,
            "DR1TKCAL": calories,
            "DR1TSUGR": sugar,
            "DR1TALCO": alcohol,
        }

        cluster_id, cluster_label, action_text = predict_cluster_from_input(user_inputs)

        st.markdown("### 🧪 Results")
        st.write(f"**Closest Cluster:** {cluster_label} (ID: {cluster_id})")
        st.write(f"**RL Recommended Preventive Action:** {action_text}")

        st.markdown("### Patients in the same Cluster (sample)")
        sample = df[df["Cluster"] == cluster_id].sample(min(5, (df["Cluster"] == cluster_id).sum()))
        st.dataframe(sample[["RIDAGEYR", "BMXBMI", "BPXSY1", "BPXDI1", "Cluster_Name", "RL_Recommended_Action"]])
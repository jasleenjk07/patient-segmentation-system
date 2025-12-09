import streamlit as st
import pandas as pd 
import numpy as np 
import joblib
import umap.umap_ as umap
from sklearn.cluster import KMeans
from fpdf import FPDF
import io

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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --accent: #14b8a6;
    --accent-hover: #0d9488;
    --border-soft: rgba(148, 163, 184, 0.4);
}

/* Default body font */
.stApp, body {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}

/* Light / Dark aware background using prefers-color-scheme */
@media (prefers-color-scheme: light) {
    .stApp {
        background-color: #f8fafc;
        background-image:
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.12) 0px, transparent 40%),
            radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.10) 0px, transparent 40%);
    }
    .custom-card,
    [data-testid="stForm"],
    [data-testid="stDataFrame"],
    [data-testid="stImage"],
    [data-testid="stArrowChart"] {
        background: rgba(255, 255, 255, 0.9);
        color: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.4);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }
    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid rgba(148, 163, 184, 0.4);
    }
}

@media (prefers-color-scheme: dark) {
    .stApp {
        background-color: #020617;
        background-image:
            radial-gradient(at 0% 0%, rgba(20, 184, 166, 0.18) 0px, transparent 40%),
            radial-gradient(at 100% 100%, rgba(129, 140, 248, 0.14) 0px, transparent 40%);
    }
    .custom-card,
    [data-testid="stForm"],
    [data-testid="stDataFrame"],
    [data-testid="stImage"],
    [data-testid="stArrowChart"] {
        background: rgba(15, 23, 42, 0.92);
        color: #e5e7eb;
        border: 1px solid rgba(51, 65, 85, 0.9);
        box-shadow: 0 16px 35px rgba(0, 0, 0, 0.55);
    }
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid rgba(30, 41, 59, 0.9);
    }
}

/* Shared card styling */
.custom-card,
[data-testid="stForm"],
[data-testid="stDataFrame"],
[data-testid="stImage"],
[data-testid="stArrowChart"] {
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

/* Pipeline cards – stronger borders for light theme */
.pipeline-card {
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
}

@media (prefers-color-scheme: light) {
    .pipeline-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(148, 163, 184, 0.8); /* visible on light bg */
    }
}

@media (prefers-color-scheme: dark) {
    .pipeline-card {
        background: rgba(15, 23, 42, 0.96);
        border: 1px solid rgba(148, 163, 184, 0.6);
    }
}

/* Sidebar radio styling */
[data-testid="stSidebar"] div[role="radiogroup"] {
    background: transparent;
    border: none;
    flex-direction: column;
    gap: 12px;
    padding: 0;
}

[data-testid="stSidebar"] label {
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.4);
    border-radius: 12px;
    padding: 10px 14px;
    width: 100%;
    transition: all 0.15s ease;
    cursor: pointer;
    margin: 0 !important;
}

[data-testid="stSidebar"] label:hover {
    background-color: rgba(20, 184, 166, 0.16) !important;
    border-color: rgba(20, 184, 166, 0.5);
    transform: translateX(4px);
}

[data-testid="stSidebar"] label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 14px !important;
    font-weight: 500 !important;
    margin: 0;
}

/* Inputs */
.stTextInput > div > div > input, 
.stNumberInput > div > div > input, 
.stSelectbox > div > div > div {
    background-color: rgba(148, 163, 184, 0.06) !important;
    border-radius: 10px;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stTextInput > div > div > input:focus, 
.stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.25) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: all 0.18s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(20, 184, 166, 0.35);
}

/* Metric text */
[data-testid="stMetricLabel"] {
    color: var(--accent) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem;
}

/* Hide fullscreen button clutter on charts */
button[title="View fullscreen"] {
    display: none;
}
</style>
""", unsafe_allow_html = True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧭 Menu")
    page = st.radio(
        "",
        [
            "🏠 Overview",
            "🧬 Cluster Segmentation",
            "📊 Cluster Profiles",
            "🎛️ What-If Simulator",
            "📂 Batch Processing for Clinicians",
            "📄 PDF Medical Report",
            "👤 Personalized Recommendation"
        ],
        label_visibility="collapsed"
    )

st.title("🩺 Patient Segmentation System for Preventive Healthcare Planning")
st.markdown("### NHANES-based Unsupervised + Reinforcement Learning Recommendation System")
st.markdown("---")

# ------------------------------
# Page 1: Overview
# ------------------------------
if page == "🏠 Overview":
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 60px 0;">
        <div style="font-size: 64px; margin-bottom: 20px; animation: float 6s ease-in-out infinite;">🩺</div>
        <h1 style="background: linear-gradient(135deg, #2dd4bf 0%, #ccfbf1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 56px; font-weight: 800; margin-bottom: 24px; letter-spacing: -0.02em;">
            Patient Intelligence
        </h1>
        <p style="font-size: 20px; max-width: 700px; margin: 0 auto 40px; line-height: 1.6;">
            Advanced patient segmentation system leveraging <strong>Unsupervised Learning</strong> to transform complex NHANES data into actionable preventive healthcare strategies.
        </p>
    </div>

    <!-- Features Grid -->
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-bottom: 60px;">
        <div class="custom-card" style="text-align: center; padding: 32px 24px;">
            <div style="font-size: 40px; margin-bottom: 16px;">🧬</div>
            <h3 style="font-weight: 700; margin-bottom: 12px;">Smart Clustering</h3>
            <p style="font-size: 14px; line-height: 1.5;">
                Powered by <strong>UMAP</strong> and <strong>KMeans</strong> to identify distinct, clinically relevant patient subgroups from high-dimensional data.
            </p>
        </div>
        <div class="custom-card" style="text-align: center; padding: 32px 24px;">
            <div style="font-size: 40px; margin-bottom: 16px;">📊</div>
            <h3 style="font-weight: 700; margin-bottom: 12px;">Visual Insights</h3>
            <p style="font-size: 14px; line-height: 1.5;">
                Interactive <strong>cluster profiles</strong> and distributions help uncover medication usage patterns and risk factors.
            </p>
        </div>
        <div class="custom-card" style="text-align: center; padding: 32px 24px;">
            <div style="font-size: 40px; margin-bottom: 16px;">🎯</div>
            <h3 style="font-weight: 700; margin-bottom: 12px;">Actionable AI</h3>
            <p style="font-size: 14px; line-height: 1.5;">
                Reinforcement Learning inspired rules engine provides <strong>personalized preventive recommendations</strong>.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; margin-bottom: 32px;'>System Status</h3>", unsafe_allow_html=True)

    # Custom Stat Cards (Data Ticker)
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-bottom: 32px;">
        <div class="custom-card">
            <div style="font-size: 14px; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em;">Total Patients</div>
            <div style="font-size: 32px; font-weight: 700; font-family: 'Outfit', sans-serif;">{len(df):,}</div>
            <div style="font-size: 12px; color: #14b8a6;">In Data Warehouse</div>
        </div>
        <div class="custom-card">
            <div style="font-size: 14px; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em;">Active Clusters</div>
            <div style="font-size: 32px; font-weight: 700; font-family: 'Outfit', sans-serif;">{df["Cluster"].nunique()}</div>
            <div style="font-size: 12px; color: #8b5cf6;">Segmentation Profiles</div>
        </div>
        <div class="custom-card">
            <div style="font-size: 14px; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em;">Strategies</div>
            <div style="font-size: 32px; font-weight: 700; font-family: 'Outfit', sans-serif;">{len(actions)}</div>
            <div style="font-size: 12px; color: #f59e0b;">Preventive Pathways</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3 style="text-align: center; margin-bottom: 24px;">Processing Pipeline</h3>
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 16px; align-items: center; margin-bottom: 40px;">
    <!-- Step 1 -->
    <div class="pipeline-card">
    <div style="font-size: 24px; margin-bottom: 8px;">📂</div>
    <div style="font-size: 14px; font-weight: 600;">NHANES Data</div>
    </div>
    <div style="color: #64748b; font-size: 24px;">→</div>
    <!-- Step 2 -->
    <div class="pipeline-card">
    <div style="font-size: 24px; margin-bottom: 8px;">🧹</div>
    <div style="font-size: 14px; font-weight: 600;">Preprocessing</div>
    </div>
    <div style="color: #64748b; font-size: 24px;">→</div>
    <!-- Step 3 -->
    <div class="pipeline-card">
    <div style="font-size: 24px; margin-bottom: 8px;">📉</div>
    <div style="font-size: 14px; font-weight: 600;">UMAP Reduction</div>
    </div>
    <div style="color: #64748b; font-size: 24px;">→</div>
    <!-- Step 4 -->
    <div class="pipeline-card">
    <div style="font-size: 24px; margin-bottom: 8px;">🧬</div>
    <div style="font-size: 14px; font-weight: 600;">KMeans Clustering</div>
    </div>
    <div style="color: #64748b; font-size: 24px;">→</div>
    <!-- Step 5 -->
    <div class="pipeline-card">
    <div style="font-size: 24px; margin-bottom: 8px;">🤖</div>
    <div style="font-size: 14px; font-weight: 600;">RL Agent</div>
    </div>
    </div>
    """, unsafe_allow_html=True)


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

    st.markdown("### 📍 UMAP Projection of Patient Clusters")

    fig, ax = plt.subplots(figsize=(10,7))
    # Dark mode plotting
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Custom color cycle if needed, but default is okay. 
    # Let's use a spectral map or just iterate.
    cmap = plt.get_cmap("tab10")

    for i, cname in enumerate(sorted(plot_df["Cluster_Name"].unique())):
        subset = plot_df[plot_df["Cluster_Name"] == cname]
        ax.scatter(subset["UMAP1"], subset["UMAP2"], s=15, label=cname, alpha=0.8, edgecolors='white', linewidths=0.1)

    # Axis styling
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.spines['left'].set_color('#cbd5e1') 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', colors='#cbd5e1')
    ax.tick_params(axis='y', colors='#cbd5e1')
    ax.yaxis.label.set_color('#cbd5e1')
    ax.xaxis.label.set_color('#cbd5e1')
    ax.title.set_color('white')

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    # Legend styled
    leg = ax.legend(markerscale=2, fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    for text in leg.get_texts():
        text.set_color("#cbd5e1")

    st.pyplot(fig)

    st.markdown("### 📊 Cluster Size Distribution")

    cluster_counts = (
        df.groupby("Cluster_Name")["Cluster"]
          .count()
          .sort_values(ascending = False)
          .reset_index()
          .rename(columns = {"Cluster": "Count"})
    )
    
    fig2, ax2 = plt.subplots(figsize=(10,5))
    fig2.patch.set_facecolor('none')
    ax2.set_facecolor('none')
    
    bars = ax2.bar(cluster_counts["Cluster_Name"], cluster_counts["Count"], color='#14b8a6')
    
    # Axis styling
    ax2.spines['bottom'].set_color('#cbd5e1')
    ax2.spines['left'].set_color('#cbd5e1')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', colors='#cbd5e1', rotation=45)
    ax2.tick_params(axis='y', colors='#cbd5e1')
    ax2.yaxis.label.set_color('#cbd5e1')
    
    ax2.set_ylabel("Number of Patients")
    
    st.pyplot(fig2)
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
 # Page 4: What-If Health Simulator
 # ------------------------------
elif page == "🎛️ What-If Simulator":
    st.subheader("🎛️ What-If Health Scenario Simulator")

    st.markdown(
        """
        Explore how changing key health parameters (like BMI, blood pressure, diet)
        might move a profile between different **risk clusters** and how the 
        **recommended preventive action** changes.
        """
    )

    # Layout: left = baseline info, right = sliders
    left_col, right_col = st.columns(2)

    with left_col:
        base_cluster = st.selectbox(
            "Start from a cluster profile",
            sorted(cluster_summary["Cluster_Name"].unique()),
            help="Choose a cluster whose average profile will be used as the baseline."
        )
        base_row = cluster_summary[cluster_summary["Cluster_Name"] == base_cluster].iloc[0]

        st.markdown("#### Baseline (Cluster Averages)")
        st.write(f"- **Age:** {base_row['RIDAGEYR']:.1f} years")
        st.write(f"- **BMI:** {base_row['BMXBMI']:.1f}")
        st.write(f"- **Systolic BP:** {base_row['BPXSY1']:.1f} mmHg")
        st.write(f"- **Diastolic BP:** {base_row['BPXDI1']:.1f} mmHg")
        st.write(f"- **Calories:** {base_row['DR1TKCAL']:.0f} kcal/day")
        st.write(f"- **Sugar:** {base_row['DR1TSUGR']:.1f} g/day")
        st.write(f"- **Fat:** {base_row['DR1TTFAT']:.1f} g/day")
        st.write(f"- **On meds (%):** {base_row['any_med_use']*100:.1f}%")

        st.info(
            "You can adjust the sliders on the right to create a 'what-if' scenario. "
            "The system will re-compute the closest cluster and the RL-based recommendation."
        )

    with right_col:
        st.markdown("#### Adjust Your Scenario")

        # Sliders start from cluster averages
        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=90,
            value=int(round(base_row["RIDAGEYR"])),
        )
        bmi = st.slider(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=float(round(base_row["BMXBMI"], 1)),
            step=0.1,
        )
        sbp = st.slider(
            "Systolic BP (BPXSY1)",
            min_value=80,
            max_value=220,
            value=int(round(base_row["BPXSY1"])),
        )
        dbp = st.slider(
            "Diastolic BP (BPXDI1)",
            min_value=40,
            max_value=130,
            value=int(round(base_row["BPXDI1"])),
        )
        calories = st.slider(
            "Daily calories (DR1TKCAL)",
            min_value=500,
            max_value=6000,
            value=int(round(base_row["DR1TKCAL"])),
            step=50,
        )
        sugar = st.slider(
            "Daily sugar (DR1TSUGR, g)",
            min_value=0,
            max_value=500,
            value=int(round(base_row["DR1TSUGR"])),
        )

        # DR1TALCO is not in cluster_summary, so use dataset median as a reasonable default
        default_alcohol = 0
        if "DR1TALCO" in df.columns:
            default_alcohol = int(round(df["DR1TALCO"].median()))
        alcohol = st.slider(
            "Alcohol intake (DR1TALCO, g/day)",
            min_value=0,
            max_value=300,
            value=default_alcohol,
        )

        simulate = st.button("Simulate What-If Scenario")

    if simulate:
        # We don't ask for gender here; assume a default encoding if needed
        if "RIAGENDR" in feature_columns:
            gender_val = 0  # e.g., default male encoding (0) as used in training
        else:
            gender_val = None

        user_inputs = {
            "RIDAGEYR": age,
            "RIAGENDR": gender_val,
            "BMXBMI": bmi,
            "BPXSY1": sbp,
            "BPXDI1": dbp,
            "DR1TKCAL": calories,
            "DR1TSUGR": float(sugar),
            "DR1TALCO": float(alcohol),
        }

        sim_cluster_id, sim_cluster_label, sim_action_text = predict_cluster_from_input(user_inputs)

        st.markdown("### 🔍 Simulation Result")
        st.write(f"- **Baseline cluster:** {base_cluster}")
        st.write(f"- **Scenario assigned to cluster:** {sim_cluster_label} (ID: {sim_cluster_id})")
        st.write(f"- **RL Recommended Preventive Action:** {sim_action_text}")

        # Show a few similar patients in that simulated cluster
        st.markdown("#### Similar patients in this simulated cluster (sample)")
        same_cluster = df[df["Cluster"] == sim_cluster_id]
        if not same_cluster.empty:
            sample = same_cluster.sample(min(5, len(same_cluster)))
            st.dataframe(
                sample[
                    ["RIDAGEYR", "BMXBMI", "BPXSY1", "BPXDI1", "Cluster_Name", "RL_Recommended_Action"]
                ]
            )
        else:
            st.write("No patients found in this cluster (unexpected for this dataset).")

# ------------------------------
# Page: Batch Processing for Clinicians
# ------------------------------
elif page == "📂 Batch Processing for Clinicians":
    st.subheader("📂 Batch Processing for Clinicians")

    st.markdown(
        """
        Upload a **CSV file** containing multiple patient records.

        For each row, the system will:
        - Build a full feature vector (filling missing fields with dataset medians)
        - Project it with **UMAP**
        - Assign a **cluster** using KMeans
        - Attach the **RL-based recommended preventive action**

        This is designed for clinicians or researchers who want to score many patients at once.
        """
    )

    st.markdown("#### 1️⃣ Upload CSV")
    st.markdown(
        """Minimal recommended columns (case-sensitive):

        - `RIDAGEYR`  (Age in years)
        - `BMXBMI`    (Body Mass Index)
        - `BPXSY1`    (Systolic BP)
        - `BPXDI1`    (Diastolic BP)
        - `DR1TKCAL`  (Daily calories)
        - `DR1TSUGR`  (Daily sugar, g)
        - `DR1TALCO`  (Alcohol intake, g/day)

        Extra numeric columns that match the training features will also be used if present.
        """
    )

    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
        else:
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())

            # Identify which feature columns we can actually use from this file
            usable_cols = [c for c in feature_columns if c in batch_df.columns]

            if len(usable_cols) == 0:
                st.error("None of the required feature columns are present in the uploaded file.")
            else:
                st.markdown("#### 2️⃣ Process & Assign Clusters")

                # Start from median-based baseline for each row, then overwrite with uploaded values
                base_matrix = pd.DataFrame([feature_medians] * len(batch_df))

                # Overwrite any columns that appear both in the uploaded CSV and feature set
                for col in batch_df.columns:
                    if col in base_matrix.columns:
                        base_matrix[col] = batch_df[col]

                # Keep only the feature_columns used during training
                X_batch = base_matrix[feature_columns]

                # Apply UMAP and KMeans
                X_batch_umap = umap_model.transform(X_batch)
                batch_clusters = kmeans.predict(X_batch_umap)

                # Map to cluster names and RL actions
                cluster_labels = [cluster_names.get(int(c), f"Cluster {int(c)}") for c in batch_clusters]
                action_ids = [cluster_to_learned_action.get(int(c), None) for c in batch_clusters]
                action_texts = [actions.get(a, "No recommendation found") for a in action_ids]

                # Build results DataFrame (original data + outputs)
                result_df = batch_df.copy()
                result_df["Assigned_Cluster"] = batch_clusters
                result_df["Cluster_Name"] = cluster_labels
                result_df["RL_Recommended_Action"] = action_texts

                st.success(f"Successfully processed {len(result_df)} patients.")

                st.markdown("#### 3️⃣ Batch Results (first 20 rows)")
                st.dataframe(result_df.head(20))

                # Offer CSV download of full results
                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download full batch results as CSV",
                    data=csv_bytes,
                    file_name="batch_patient_recommendations.csv",
                    mime="text/csv",
                )
    else:
        st.info("Upload a CSV file to run batch recommendations.")

elif page == "📄 PDF Medical Report":
    st.subheader("📄 Downloadable Medical Segment Report")

    st.markdown(
        """
        Generate a **PDF summary report** for any patient segment (cluster). This can be used in
        presentations, documentation, or as a simple handout showing the key characteristics and
        the recommended preventive strategy for that segment.
        """
    )

    # Let user choose which cluster to generate report for
    cluster_choice = st.selectbox(
        "Select a cluster to generate report for",
        sorted(cluster_summary["Cluster_Name"].unique())
    )

    cdata = cluster_summary[cluster_summary["Cluster_Name"] == cluster_choice].iloc[0]

    st.markdown("### Preview of Report Content")
    st.write(f"**Cluster:** {cluster_choice}")
    st.write(f"**Top RL Recommended Action:** {cdata['Top_Recommended_Action']}")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- Average Age: {cdata['RIDAGEYR']:.1f} years")
        st.write(f"- Average BMI: {cdata['BMXBMI']:.1f}")
        st.write(f"- Systolic BP: {cdata['BPXSY1']:.1f} mmHg")
        st.write(f"- Diastolic BP: {cdata['BPXDI1']:.1f} mmHg")
    with col2:
        st.write(f"- LDL Cholesterol: {cdata['LBDLDL']:.1f} mg/dL")
        st.write(f"- Total Cholesterol: {cdata['LBXTC']:.1f} mg/dL")
        st.write(f"- Calories: {cdata['DR1TKCAL']:.0f} kcal/day")
        st.write(f"- Sugar: {cdata['DR1TSUGR']:.1f} g/day")
        st.write(f"- Fat: {cdata['DR1TTFAT']:.1f} g/day")

    st.write(f"- Patients on meds (any_med_use): {cdata['any_med_use']*100:.1f}%")

    st.markdown("---")
    st.markdown("#### Generate PDF Report")

    if st.button("📄 Create PDF Report"):
        # Build a simple one-page PDF using FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Patient Segment Medical Report", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.ln(4)
        pdf.multi_cell(0, 8, f"Cluster Name: {cluster_choice}")
        pdf.multi_cell(0, 8, f"Top Recommended Action: {cdata['Top_Recommended_Action']}")

        pdf.ln(4)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Clinical Profile", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, f"Average Age: {cdata['RIDAGEYR']:.1f} years")
        pdf.multi_cell(0, 7, f"Average BMI: {cdata['BMXBMI']:.1f}")
        pdf.multi_cell(0, 7, f"Systolic BP: {cdata['BPXSY1']:.1f} mmHg")
        pdf.multi_cell(0, 7, f"Diastolic BP: {cdata['BPXDI1']:.1f} mmHg")
        pdf.multi_cell(0, 7, f"LDL Cholesterol: {cdata['LBDLDL']:.1f} mg/dL")
        pdf.multi_cell(0, 7, f"Total Cholesterol: {cdata['LBXTC']:.1f} mg/dL")

        pdf.ln(3)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Diet & Lifestyle", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, f"Average Calories: {cdata['DR1TKCAL']:.0f} kcal/day")
        pdf.multi_cell(0, 7, f"Average Sugar Intake: {cdata['DR1TSUGR']:.1f} g/day")
        pdf.multi_cell(0, 7, f"Average Fat Intake: {cdata['DR1TTFAT']:.1f} g/day")
        pdf.multi_cell(0, 7, f"Patients on Medications: {cdata['any_med_use']*100:.1f}%")

        pdf.ln(3)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Preventive Strategy", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, f"Recommended Action: {cdata['Top_Recommended_Action']}")

        # Export PDF to bytes
        pdf_bytes = pdf.output(dest="S").encode("latin-1")

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"medical_report_{cluster_choice.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

# ------------------------------
# Page 5: Personalized Recommendation
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
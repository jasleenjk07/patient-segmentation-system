# 🩺 Patient Segmentation System Using Unsupervised & Reinforcement Learning

A **Preventive Healthcare Recommendation System** built using the **NHANES medical dataset**, applying **UMAP + KMeans clustering** to group patients into health-based categories and **Q-Learning Reinforcement Learning** to determine the best preventive health action for each cluster.

The system also includes an **interactive Streamlit dashboard** that allows users to:
- Explore cluster-level insights
- Visualize patient distribution
- Enter personal health parameters
- Receive personalized preventive recommendations

---

## 🚀 Project Objectives

- Identify meaningful patient segments based on medical, dietary, lab, and lifestyle variables
- Predict the most beneficial preventive action per segment using reinforcement learning
- Provide **cluster-level and personalized recommendations**
- Help clinicians understand metabolic & lifestyle-related health risks

---

## 📦 Tech Stack

| Component | Technology |
|----------|------------|
| **Programming Language** | Python |
| **Unsupervised Clustering** | UMAP + KMeans |
| **Reinforcement Learning** | Q-Learning |
| **Dashboard** | Streamlit |
| **Web UI (optional)** | Flask + TailwindCSS |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Dataset** | NHANES (CDC Public Dataset) |

---

## 📁 Project Structure
patient-segmentation-system/
│
├── data/
│   └── processed/
│       └── final_recommendation_system.csv
│
├── models/
│   ├── scaler.pkl
│   ├── umap.pkl
│   ├── kmeans_umap.pkl
│   ├── feature_columns.pkl
│   ├── cluster_names.pkl
│   ├── actions.pkl
│   └── cluster_to_learned_action.pkl
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── clustering_umap_kmeans.ipynb
│   ├── reinforcement_learning.ipynb
│   └── visualization.ipynb
│
├── app.py                           # Streamlit Dashboard
├── flask_app.py                     # Flask + Tailwind demo UI
└── README.md

---

## 🧠 Machine Learning Workflow

### **1. Data Preparation**
- Merged multiple NHANES segments (demographics, dietary, labs, medications, exam)
- Cleaned missing values and standardized features
- Created engineered risk features

### **2. Unsupervised Learning (Clustering)**
- Dimensionality reduction using **UMAP (n_components = 2)**
- Cluster formation using **KMeans (k = 9)**
- Manual labeling of clusters for interpretability

Example cluster names:

| Cluster | Name |
|---------|------|
| 0 | Middle-aged Metabolic Risk |
| 1 | Healthy Youth |
| 2 | Young Lifestyle Shift |
| 3 | Fit Adults |
| 4 | Elderly Chronic Condition |
| 5 | High Sugar/Fat Consumers |
| 6 | Unhealthy Youth |
| 7 | Alcohol-associated Metabolic Risk |
| 8 | Moderate-Risk Adults |

---

### **3. Reinforcement Learning — Q-Learning**
- **State** = Cluster ID
- **Action** = Preventive recommendation (e.g., reduce alcohol, increase exercise)
- **Reward** = Simulated improvement in health outcomes
- Learned optimal action for each cluster using Q-Table updates

Example RL Output:
Cluster 0 → Quit smoking & nutritional coaching
Cluster 1 → Maintain exercise & healthy habits
Cluster 7 → Reduce alcohol consumption
Cluster 4 → Regular doctor follow-up

---

## 📊 Dashboard Features (Streamlit)

| Page | Description |
|--------|-------------|
| 🏠 **Overview** | Summary of problem, methods, dataset statistics |
| 🧬 **Cluster Segmentation** | UMAP scatter plot + cluster distribution bar chart |
| 📊 **Cluster Profiles** | Mean health indicators and top actions |
| 👤 **Personalized Recommendation** | User form → cluster prediction → RL recommendation |

---

## ▶️ Run the Dashboard

### **Install dependencies**
```bash
pip install -r requirements.txt
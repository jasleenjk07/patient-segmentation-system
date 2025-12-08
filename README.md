# 🩺 Patient Segmentation System (Premium AI Edition)

A high-performance **Preventive Healthcare Recommendation System** leveraging **Unsupervised Learning (UMAP + KMeans)** for patient segmentation and **Reinforcement Learning (Q-Learning)** for personalized health interventions.

---

## 🚀 Key Features

### 1. 🧠 Advanced ML Pipeline
- **Dataset**: Built on **NHANES** (CDC) data including physiological, dietary, and lab metrics.
- **Clustering**: **UMAP** dimensionality reduction + **KMeans** clustering identifies 9 distinct patient profiles (e.g., "Middle-Aged Metabolic Risk", "Healthy Youth").
- **Recommendation Engine**: **Q-Learning Policy** determines the optimal lifestyle action for each cluster to maximize long-term health rewards.

### 2. � Premium Web Dashboard
A modern, dark-themed Flask application featuring:
- **Interactive Visualizations**: 
    - **Radar Charts**: Compare your vitals (BMI, BP, Calories) against your cluster average.
    - **Cluster Galaxy Map**: See your exact position in the patient universe (Scatter Plot).
- **"What-If" Health Simulator**:
    - Gamified controls (Sliders) to simulate weight loss or diet changes.
    - Instantly updates prediction and recommended actions.
- **Batch Processing**:
    - Upload CSV files to analyze hundreds of patients at once.
- **PDF Reporting**:
    - Generate professional medical reports for clinical use.

---

## 📦 Tech Stack

| Component | Technology |
|----------|------------|
| **Backend** | Python, Flask |
| **ML Libraries** | Scikit-Learn, UMAP-Learn, Pandas, NumPy |
| **Frontend** | HTML5, TailwindCSS (CDN), Chart.js |
| **Reporting** | ReportLab (PDF) |
| **Persistence** | Joblib |

---

## 🛠️ Usage Guide

### Prerequisities
```bash
pip install -r requirements.txt
pip install reportlab umap-learn flask pandas numpy scikit-learn
```

### 1. Run the Application
```bash
python flask_app.py
```
Visit **http://127.0.0.1:5000/** in your browser.

### 2. Dashboard Workflow
- **Input Vitals**: Enter Age, Gender, BMI, BP, and Calories.
- **Run Analysis**: Get your **Identified Segment** and **AI Recommendation**.
- **Visuals**: Check the **Radar Chart** to see which metric is driving your risk.
- **Simulation**: Scroll down to the **Simulator** to see how lowering your BMI affects your segment.
- **Report**: Click **Download Medical Report** to save a PDF.

### 3. Batch Analysis
- Scroll to the **Batch Analysis** section.
- Upload a CSV with columns: `Age`, `Gender`, `BMI`, `SBP`, `DBP`, `Calories`.
- Download the processed results with appended predictions.

---

## 📂 Project Structure
```
patient-segmentation-system/
├── flask_app.py           # Main Web Application (Flask)
├── app.py                 # (Legacy) Streamlit Dashboard
├── assets/                # CSS/Images
├── utils/                 # Helper functions
├── models/                # Pre-trained .pkl models
│   ├── scaler.pkl
│   ├── umap.pkl
│   ├── kmeans_umap.pkl
│   ├── feature_columns.pkl
│   ├── cluster.pkl
│   └── policy.pkl
└── README.md              # Documentation
```

---

## 🧠 Model Details

### Unsupervised Learning (Clustering)
- **UMAP (n_components=2)**: Reduces 40+ medical features to 2 dimensions for visualization and density-based clustering.
- **KMeans (k=9)**: Groups patients into actionable segments.

### Reinforcement Learning
- **State Space**: The 9 Cluster IDs.
- **Action Space**: 30+ standardized health interventions (e.g., "Increase cardio", "Reduce sodium").
- **Reward Signal**: Based on improvement in cardiovascular risk scores.

---

*Built for the Advanced Agentic Coding Project.*
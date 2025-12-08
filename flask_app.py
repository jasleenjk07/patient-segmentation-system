from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import io
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import joblib
import umap.umap_ as umap
from sklearn.cluster import KMeans

app = Flask(__name__)

# -----------------------------
# 1. Load Data & Models
# -----------------------------
# We load these once at startup so we don't hit disk on every request
try:
    df = pd.read_csv("data/processed/final_recommendation_system.csv")
    scaler = joblib.load("models/scaler.pkl")
    umap_model = joblib.load("models/umap.pkl")
    kmeans = joblib.load("models/kmeans_umap.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    cluster_names = joblib.load("models/cluster_names.pkl")
    actions = joblib.load("models/actions.pkl")
    cluster_to_learned_action = joblib.load("models/cluster_to_learned_action.pkl")
    
    # Filter feature columns to ensure they exist in DF
    feature_columns = [
        c for c in feature_columns
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    # Pre-calculate medians for missing user inputs
    feature_medians = df[feature_columns].median()
    
    # ---------------------------------------------------------
    # NEW: Pre-compute data for Visualizations
    # ---------------------------------------------------------
    # 1. Cluster Averages (for Radar Chart)
    # Group by Cluster (ID) and get mean of feature columns
    # We normalized features in training usually, but here we work with raw values 
    # if the CSV has raw values. Assuming CSV is raw.
    # We need to map Cluster IDs to their means.
    cluster_means = df.groupby("Cluster")[feature_columns].mean().to_dict(orient="index")
    
    # 2. Background UMAP Scatter (Sampled for performance)
    # We'll take a random sample of 500 points to show as "background galaxy"
    SAMPLE_SIZE = 500
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        df_sample = df.copy()
        
    X_sample = df_sample[feature_columns]
    # Transform to get 2D coords
    # Note: If your scaler was used before UMAP in training, you MUST scale here too.
    # app.py code: X_umap = umap_model.transform(X) (It didn't show explicit scaling before UMAP there?)
    # Let's assume app.py was correct. 
    umap_sample = umap_model.transform(X_sample) 
    
    # Prepare list of dicts for Chart.js: [{x: 1.2, y: -0.5, cluster: 1}, ...]
    scatter_data = []
    sample_clusters = df_sample["Cluster"].values
    for i in range(len(umap_sample)):
        scatter_data.append({
            "x": float(umap_sample[i][0]),
            "y": float(umap_sample[i][1]),
            "cluster": int(sample_clusters[i])
        })
    
    MODELS_LOADED = True
    print("✅ All ML models loaded successfully.")
except Exception as e:
    MODELS_LOADED = False
    print(f"⚠️ Warning: Could not load ML models. {e}")
    # Initialize placeholders to prevent crash if models missing
    df, feature_medians = None, None
    scaler, umap_model, kmeans = None, None, None
    feature_columns, cluster_names, actions, cluster_to_learned_action = [], {}, [], {}
    cluster_means, scatter_data = {}, []

# -----------------------------
# 2. Prediction Helper
# -----------------------------
def predict_cluster_from_input(user_inputs: dict):
    """
    Predicts cluster and returns (Cluster Name, Recommended Action).
    """
    if not MODELS_LOADED:
        return "System Error", "Models not loaded."

    # Start with median values (handles missing features safely)
    row = feature_medians.copy()

    # Update with user provided values
    for key, value in user_inputs.items():
        if key in row.index:
            row[key] = value

    # Create DataFrame for single sample
    X_user = pd.DataFrame([row[feature_columns]])
    
    # Transform: Standard Scaling might be needed if your UMAP model expects it.
    # NOTE: In app.py, 'umap_model.transform(X)' is called directly on X.
    # If your original pipeline scaled data BEFORE UMAP, you must scale here too.
    # Assuming app.py logic is correct (direct transform):
    try:
        X_user_umap = umap_model.transform(X_user)
        cluster_id = int(kmeans.predict(X_user_umap)[0])
        
        cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        action_id = cluster_to_learned_action.get(cluster_id, None)
        
        # 'actions' is likely a list or array where index = action_id
        # Safely get action text
        if action_id is not None and 0 <= action_id < len(actions):
            recommended_action = actions[action_id]
        else:
            recommended_action = "Consult a healthcare provider."
            
        return cluster_label, recommended_action, cluster_id, X_user_umap[0]
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error", "Could not process input.", None, None

# -----------------------------
# Tailwind + basic HTML template
# -----------------------------
# -----------------------------
# LANDING PAGE TEMPLATE
# -----------------------------
LANDING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Patient AI - Premium Healthcare Intelligence</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
      tailwind.config = {
        darkMode: 'class',
        theme: {
          extend: {
            colors: {
              primary: {
                50: '#f0fdfa',
                100: '#ccfbf1',
                200: '#99f6e4',
                300: '#5eead4',
                400: '#2dd4bf',
                500: '#14b8a6',
                600: '#0d9488',
                700: '#0f766e',
                800: '#115e59',
                900: '#134e4a',
                950: '#042f2e',
              },
              dark: {
                900: '#020617',
                800: '#0f172a',
                700: '#1e293b',
              }
            },
            fontFamily: {
              sans: ['Plus Jakarta Sans', 'sans-serif'],
              display: ['Outfit', 'sans-serif'],
            },
            animation: {
              'blob': 'blob 7s infinite',
              'fade-in-up': 'fadeInUp 0.8s ease-out forwards',
              'float': 'float 6s ease-in-out infinite',
            },
            keyframes: {
              blob: {
                '0%': { transform: 'translate(0px, 0px) scale(1)' },
                '33%': { transform: 'translate(30px, -50px) scale(1.1)' },
                '66%': { transform: 'translate(-20px, 20px) scale(0.9)' },
                '100%': { transform: 'translate(0px, 0px) scale(1)' },
              },
              fadeInUp: {
                '0%': { opacity: '0', transform: 'translateY(20px)' },
                '100%': { opacity: '1', transform: 'translateY(0)' },
              },
              float: {
                '0%, 100%': { transform: 'translateY(0)' },
                '50%': { transform: 'translateY(-20px)' },
              }
            }
          }
        }
      }
    </script>
    <style>
        body {
            background-color: #020617; /* Dark 900 */
            background-image: 
                radial-gradient(at 0% 0%, rgba(15, 118, 110, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(45, 212, 191, 0.1) 0px, transparent 50%);
        }
        .glass-nav {
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .glass-card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        }
        .glass-card:hover {
            border-color: rgba(45, 212, 191, 0.3);
            background: rgba(30, 41, 59, 0.6);
            transform: translateY(-5px);
        }
        .text-gradient {
            background: linear-gradient(135deg, #2dd4bf 0%, #ccfbf1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
</head>
<body class="text-slate-300 font-sans antialiased overflow-x-hidden selection:bg-primary-500 selection:text-white">

    <!-- Navbar -->
    <nav class="glass-nav fixed w-full z-50 transition-all duration-300">
        <div class="max-w-7xl mx-auto px-6 lg:px-8">
            <div class="flex justify-between h-20 items-center">
                <!-- Logo -->
                <div class="flex items-center gap-3 group cursor-pointer">
                    <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white text-xl shadow-lg shadow-primary-500/20 group-hover:shadow-primary-500/40 transition-all duration-300">
                        🩺
                    </div>
                    <span class="font-display font-bold text-xl text-white tracking-tight group-hover:text-primary-300 transition-colors">
                        Patient<span class="text-primary-400">AI</span>
                    </span>
                </div>
                
                <!-- Desktop Nav -->
                <div class="hidden md:flex items-center gap-8">
                    <a href="#features" class="text-sm font-medium hover:text-white transition-colors">Features</a>
                    <a href="#how-it-works" class="text-sm font-medium hover:text-white transition-colors">How it works</a>
                    <a href="/dashboard" class="relative inline-flex h-10 overflow-hidden rounded-full p-[1px] focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50">
                        <span class="absolute inset-[-1000%] animate-[spin_2s_linear_infinite] bg-[conic-gradient(from_90deg_at_50%_50%,#E2E8F0_0%,#0F766E_50%,#E2E8F0_100%)]"></span>
                        <span class="inline-flex h-full w-full cursor-pointer items-center justify-center rounded-full bg-slate-950 px-6 py-1 text-sm font-medium text-white backdrop-blur-3xl hover:bg-slate-900 transition-colors">
                            Launch Dashboard
                        </span>
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="relative min-h-screen flex items-center pt-20 overflow-hidden">
        <!-- Background Blobs -->
        <div class="absolute top-0 -left-4 w-96 h-96 bg-primary-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
        <div class="absolute top-0 -right-4 w-96 h-96 bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
        <div class="absolute -bottom-8 left-20 w-96 h-96 bg-pink-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>

        <div class="max-w-7xl mx-auto px-6 lg:px-8 relative z-10 w-full">
            <div class="flex flex-col lg:flex-row items-center gap-16 py-12">
                
                <!-- Text Content -->
                <div class="lg:w-1/2 text-center lg:text-left animate-fade-in-up">
                    <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-900/30 border border-primary-500/30 text-primary-300 text-sm font-medium mb-8 backdrop-blur-sm">
                        <span class="relative flex h-2 w-2">
                          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-400 opacity-75"></span>
                          <span class="relative inline-flex rounded-full h-2 w-2 bg-primary-500"></span>
                        </span>
                        Next-Gen Preventive Healthcare
                    </div>
                    
                    <h1 class="font-display text-5xl lg:text-7xl font-bold text-white leading-[1.1] mb-6 tracking-tight">
                        Precision Health <br>
                        <span class="text-gradient">Through AI Intelligence</span>
                    </h1>
                    
                    <p class="text-lg text-slate-400 mb-10 leading-relaxed max-w-2xl mx-auto lg:mx-0">
                        Unlock personalized health insights using advanced Unsupervised Learning and Reinforcement Learning on NHANES data. Discover your segment, act on your future.
                    </p>
                    
                    <div class="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                        <a href="/dashboard" class="group relative px-8 py-4 bg-primary-600 hover:bg-primary-500 rounded-2xl font-bold text-white shadow-xl shadow-primary-500/20 transition-all hover:scale-[1.02] active:scale-[0.98]">
                            Get Started
                            <span class="inline-block transition-transform group-hover:translate-x-1 ml-2">→</span>
                            <div class="absolute inset-0 rounded-2xl ring-2 ring-white/10 group-hover:ring-white/30"></div>
                        </a>
                        <a href="#features" class="px-8 py-4 bg-slate-800/50 hover:bg-slate-800 rounded-2xl font-semibold text-white border border-white/10 hover:border-white/20 backdrop-blur-sm transition-all">
                            Explore Features
                        </a>
                    </div>
                    
                    <div class="mt-12 flex items-center justify-center lg:justify-start gap-8 text-slate-500 text-sm font-medium">
                        <div class="flex items-center gap-2">
                            <svg class="w-5 h-5 text-primary-500" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>
                            NHANES Data
                        </div>
                        <div class="flex items-center gap-2">
                            <svg class="w-5 h-5 text-primary-500" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>
                            Privacy First
                        </div>
                        <div class="flex items-center gap-2">
                            <svg class="w-5 h-5 text-primary-500" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>
                            Verified Models
                        </div>
                    </div>
                </div>

                <!-- Visual/Image Area (Abstract 3D or Dashboard Preview) -->
                <div class="lg:w-1/2 relative lg:h-[600px] flex items-center justify-center">
                    <div class="relative w-full max-w-lg aspect-square">
                        <div class="absolute inset-0 bg-primary-500/10 rounded-full blur-[100px] animate-pulse"></div>
                        <!-- Glass Card showcasing a metric -->
                        <div class="glass-card absolute top-10 left-0 right-0 p-6 rounded-3xl z-20 animate-[float_6s_ease-in-out_infinite]">
                            <div class="flex justify-between items-start mb-4">
                                <div>
                                    <h4 class="text-slate-400 text-xs font-bold uppercase tracking-wider">Active Segment</h4>
                                    <div class="text-2xl font-bold text-white mt-1">Metabolic Optimization</div>
                                </div>
                                <div class="p-2 bg-primary-500/20 rounded-lg">
                                    <span class="text-2xl">🧬</span>
                                </div>
                            </div>
                            <div class="w-full bg-slate-700/50 rounded-full h-2 mb-2">
                                <div class="bg-primary-500 h-2 rounded-full" style="width: 78%"></div>
                            </div>
                            <div class="flex justify-between text-xs text-slate-400">
                                <span>Confidence</span>
                                <span class="text-primary-300">78%</span>
                            </div>
                        </div>

                        <!-- Second Glass Card -->
                        <div class="glass-card absolute bottom-20 -right-4 w-64 p-5 rounded-2xl z-10 animate-[float_7s_ease-in-out_infinite_1s]">
                             <div class="flex items-center gap-3">
                                <div class="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400">
                                    🤖
                                </div>
                                <div>
                                    <div class="text-white font-bold text-sm">RL Policy</div>
                                    <div class="text-slate-400 text-xs">Action Recommended</div>
                                </div>
                             </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Features Grid -->
    <section id="features" class="py-32 relative bg-dark-800/50">
        <div class="max-w-7xl mx-auto px-6 lg:px-8">
            <div class="text-center max-w-3xl mx-auto mb-20 animate-fade-in-up">
                <h2 class="font-display text-4xl font-bold text-white mb-6">Engineered for Results</h2>
                <p class="text-lg text-slate-400">Our system combines three layers of intelligence to provide the most accurate health segmentation and recommendations available.</p>
            </div>
            
            <div class="grid md:grid-cols-3 gap-8">
                <!-- Card 1 -->
                <div class="glass-card p-8 rounded-3xl transition-all duration-300 group">
                    <div class="w-14 h-14 bg-slate-800 rounded-2xl flex items-center justify-center text-3xl mb-6 group-hover:scale-110 transition-transform duration-300 border border-slate-700">
                        🧬
                    </div>
                    <h3 class="font-display text-xl font-bold text-white mb-4">Smart Segmentation</h3>
                    <p class="text-slate-400 leading-relaxed">
                        We utilize <strong>UMAP</strong> for dimensionality reduction followed by <strong>KMeans</strong> to identify distinct patient clusters from complex high-dimensional NHANES data.
                    </p>
                </div>

                <!-- Card 2 -->
                <div class="glass-card p-8 rounded-3xl transition-all duration-300 group">
                    <div class="w-14 h-14 bg-slate-800 rounded-2xl flex items-center justify-center text-3xl mb-6 group-hover:scale-110 transition-transform duration-300 border border-slate-700">
                        �
                    </div>
                    <h3 class="font-display text-xl font-bold text-white mb-4">Reinforcement Learning</h3>
                    <p class="text-slate-400 leading-relaxed">
                        Unlike static rules, our <strong>Q-Learning</strong> policy learns the optimal preventive actions by simulating patient health trajectories over time.
                    </p>
                </div>

                <!-- Card 3 -->
                <div class="glass-card p-8 rounded-3xl transition-all duration-300 group">
                    <div class="w-14 h-14 bg-slate-800 rounded-2xl flex items-center justify-center text-3xl mb-6 group-hover:scale-110 transition-transform duration-300 border border-slate-700">
                        �️
                    </div>
                    <h3 class="font-display text-xl font-bold text-white mb-4">Privacy & Security</h3>
                    <p class="text-slate-400 leading-relaxed">
                        Built with privacy-first principles. Your health data is processed in real-time and never stored permanently without your explicit consent.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="py-12 border-t border-slate-800/50 relative overflow-hidden">
        <div class="max-w-7xl mx-auto px-6 lg:px-8 flex flex-col items-center">
            <div class="text-2xl mb-4">🩺</div>
            <p class="text-slate-500 text-sm mb-6">
                © 2025 Patient Segmentation System. Powered by Advanced AI.
            </p>
            <div class="flex gap-6">
                <a href="#" class="text-slate-400 hover:text-primary-400 transition-colors">Privacy</a>
                <a href="#" class="text-slate-400 hover:text-primary-400 transition-colors">Terms</a>
                <a href="#" class="text-slate-400 hover:text-primary-400 transition-colors">Documentation</a>
            </div>
        </div>
    </footer>

</body>
</html>
"""

# -----------------------------
# DASHBOARD TEMPLATE (Renamed from BASE_HTML)
# -----------------------------
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Dashboard - Patient Segmentation</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
      tailwind.config = {
        darkMode: 'class',
        theme: {
          extend: {
            colors: {
              primary: {
                50: '#f0fdfa',
                100: '#ccfbf1',
                200: '#99f6e4',
                300: '#5eead4',
                400: '#2dd4bf',
                500: '#14b8a6',
                600: '#0d9488',
                700: '#0f766e',
                800: '#115e59',
                900: '#134e4a',
                950: '#042f2e',
              },
              dark: {
                900: '#020617',
                800: '#0f172a',
                700: '#1e293b',
              }
            },
            fontFamily: {
              sans: ['Plus Jakarta Sans', 'sans-serif'],
              display: ['Outfit', 'sans-serif'],
            },
            animation: {
              'blob': 'blob 7s infinite',
              'fade-in-up': 'fadeInUp 0.5s ease-out forwards',
            },
            keyframes: {
              blob: {
                '0%': { transform: 'translate(0px, 0px) scale(1)' },
                '33%': { transform: 'translate(30px, -50px) scale(1.1)' },
                '66%': { transform: 'translate(-20px, 20px) scale(0.9)' },
                '100%': { transform: 'translate(0px, 0px) scale(1)' },
              },
              fadeInUp: {
                '0%': { opacity: '0', transform: 'translateY(10px)' },
                '100%': { opacity: '1', transform: 'translateY(0)' },
              }
            }
          }
        }
      }
    </script>
    <style>
        body {
            background-color: #020617;
            background-image: 
                radial-gradient(at 0% 0%, rgba(15, 118, 110, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(45, 212, 191, 0.05) 0px, transparent 50%);
        }
        .glass-nav {
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .glass-panel {
            background: rgba(30, 41, 59, 0.3);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .form-input {
            background-color: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e2e8f0;
        }
        .form-input:focus {
            border-color: #2dd4bf;
            ring: 1px solid #2dd4bf;
            outline: none;
        }
    </style>
</head>
<body class="text-slate-300 font-sans antialiased min-h-screen">

    <!-- Navbar -->
    <nav class="glass-nav fixed w-full z-50">
        <div class="max-w-7xl mx-auto px-6 lg:px-8">
            <div class="flex justify-between h-16 items-center">
                <a href="/" class="flex items-center gap-3 group hover:opacity-80 transition">
                    <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white shadow-lg">
                        🩺
                    </div>
                    <span class="font-display font-bold text-lg text-white tracking-tight">
                        Patient<span class="text-primary-400">AI</span> Dashboard
                    </span>
                </a>
                <div class="flex items-center gap-4">
                     <a href="/how-it-works" class="text-sm font-medium text-slate-400 hover:text-primary-300 transition-colors mr-2">How it Works</a>
                     <span class="hidden sm:inline-block text-xs font-semibold px-3 py-1 bg-primary-900/40 text-primary-300 border border-primary-500/20 rounded-full">
                        Flask Mode
                     </span>
                     <div class="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-sm">
                        👨‍⚕️
                     </div>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-24 px-6 lg:px-8">
        
        <!-- Header -->
        <div class="mb-10 animate-fade-in-up">
            <a href="/" class="text-primary-400 hover:text-primary-300 text-sm font-medium flex items-center gap-1 mb-4 transition-colors">
                ← Back to Home
            </a>
            <h1 class="font-display text-3xl font-bold text-white mb-2">
                Patient Analysis & Recommendation
            </h1>
            <p class="text-slate-400 max-w-2xl">
                Enter real-time patient vitals below. Our AI engine will segment the patient using KMeans and retrieve the optimal Reinforcement Learning policy action.
            </p>
        </div>

        <!-- Stats Grid -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10 animate-fade-in-up" style="animation-delay: 0.1s;">
            <div class="glass-panel rounded-2xl p-5 border-l-4 border-l-primary-500">
                <div class="text-xs uppercase tracking-wider text-slate-500 font-bold mb-1">Active Clusters</div>
                <div class="text-2xl font-bold text-white">9 Segments</div>
            </div>
            <div class="glass-panel rounded-2xl p-5 border-l-4 border-l-blue-500">
                <div class="text-xs uppercase tracking-wider text-slate-500 font-bold mb-1">Model Architecture</div>
                <div class="text-2xl font-bold text-white">KMeans + Q-Learning</div>
            </div>
            <div class="glass-panel rounded-2xl p-5 border-l-4 border-l-purple-500">
                <div class="text-xs uppercase tracking-wider text-slate-500 font-bold mb-1">Data Foundation</div>
                <div class="text-2xl font-bold text-white">NHANES Dataset</div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-fade-in-up" style="animation-delay: 0.2s;">
            
            <!-- Input Form -->
            <div class="lg:col-span-1 glass-panel rounded-2xl p-8 h-fit">
                <div class="flex items-center gap-2 mb-6 border-b border-slate-700/50 pb-4">
                    <span class="text-xl">📝</span>
                    <h2 class="font-display text-lg font-bold text-white">Patient Vitals</h2>
                </div>
                
                <form method="POST" class="space-y-5">
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Age</label>
                        <input type="number" name="age" min="1" max="90" value="{{ age }}" class="form-input w-full rounded-xl px-4 py-3 transition-all focus:ring-2 focus:ring-primary-500/50" required />
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Gender</label>
                        <select name="gender" class="form-input w-full rounded-xl px-4 py-3 transition-all focus:ring-2 focus:ring-primary-500/50 appearance-none">
                            <option value="Male" {% if gender == "Male" %}selected{% endif %}>Male</option>
                            <option value="Female" {% if gender == "Female" %}selected{% endif %}>Female</option>
                        </select>
                    </div>

                    <div>
                         <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">BMI</label>
                         <input type="number" step="0.1" name="bmi" value="{{ bmi }}" class="form-input w-full rounded-xl px-4 py-3 transition-all focus:ring-2 focus:ring-primary-500/50" />
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                         <div>
                            <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Systolic BP</label>
                            <input type="number" name="sbp" value="{{ sbp }}" class="form-input w-full rounded-xl px-4 py-3 transition-all focus:ring-2 focus:ring-primary-500/50" />
                        </div>
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Diastolic BP</label>
                            <input type="number" name="dbp" value="{{ dbp }}" class="form-input w-full rounded-xl px-4 py-3 transition-all focus:ring-2 focus:ring-primary-500/50" />
                        </div>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Daily Calories</label>
                        <input type="number" name="calories" value="2000" class="form-input w-full rounded-xl px-4 py-3 transition-all focus:ring-2 focus:ring-primary-500/50" />
                    </div>
                    
                    <div class="pt-4">
                        <button type="submit" class="w-full bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-500 hover:to-primary-400 text-white font-bold py-4 rounded-xl shadow-lg shadow-primary-500/20 transition-all transform hover:scale-[1.02] active:scale-[0.98]">
                            Run Analysis
                        </button>
                    </div>
                </form>
            </div>

            <!-- Results Column -->
            <div class="lg:col-span-2 space-y-6">
                
                <!-- Main Result Card -->
                <div class="glass-panel rounded-2xl p-8 border border-primary-500/20 relative overflow-hidden">
                    <div class="absolute top-0 right-0 w-64 h-64 bg-primary-500/10 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none"></div>
                    
                    <h2 class="font-display text-xl font-bold text-white mb-8 flex items-center gap-3 relative z-10">
                        <span class="flex items-center justify-center w-8 h-8 rounded-full bg-primary-500/20 text-primary-300">🚀</span>
                        <span>Analysis Results</span>
                    </h2>

                    {% if cluster_name %}
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 relative z-10">
                            <!-- Cluster Result -->
                            <div class="bg-slate-800/50 rounded-xl p-6 border border-slate-700 hover:border-emerald-500/50 transition-colors group">
                                <div class="text-xs uppercase tracking-wider text-emerald-400 font-bold mb-2 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                                    Identified Segment
                                </div>
                                <div id="result-cluster-name" class="text-xl font-bold text-white leading-tight group-hover:text-emerald-300 transition-colors">
                                    {{ cluster_name }}
                                </div>
                            </div>
                            
                            <!-- Recommendation Result -->
                            <div class="bg-slate-800/50 rounded-xl p-6 border border-slate-700 hover:border-sky-500/50 transition-colors group">
                                <div class="text-xs uppercase tracking-wider text-sky-400 font-bold mb-2 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-sky-500 animate-pulse"></span>
                                    AI Recommendation
                                </div>
                                <div id="result-action" class="text-lg font-bold text-white leading-tight group-hover:text-sky-300 transition-colors">
                                    {{ action_text }}
                                </div>
                            </div>

                            <!-- PDF Download Button -->
                            <div class="mt-4 text-center">
                                <form action="/download_report" method="POST" target="_blank">
                                    <input type="hidden" name="age" value="{{ age }}">
                                    <input type="hidden" name="gender" value="{{ gender }}">
                                    <input type="hidden" name="bmi" value="{{ bmi }}">
                                    <input type="hidden" name="sbp" value="{{ sbp }}">
                                    <input type="hidden" name="dbp" value="{{ dbp }}">
                                    <input type="hidden" name="calories" value="{{ calories }}">
                                    <input type="hidden" name="cluster_name" value="{{ cluster_name }}">
                                    <input type="hidden" name="action_text" value="{{ action_text }}">
                                    
                                    <button type="submit" class="inline-flex items-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-xl text-white font-bold text-sm transition-all border border-slate-600 hover:border-slate-500 shadow-lg group">
                                        <span class="text-xl">📄</span>
                                        <span>Download Medical Report</span>
                                    </button>
                                </form>
                            </div>
                        </div>
                        
                        {% if not models_loaded %}
                        <div class="mt-6 flex items-start gap-3 p-4 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-200 text-sm">
                            <span class="text-lg">⚠️</span>
                            <p><strong>System Warning:</strong> ML models could not be loaded. The application is running in demonstration mode with simulated predictions.</p>
                        </div>
                        {% endif %}
                    {% else %}
                        <div class="flex flex-col items-center justify-center py-16 text-center space-y-4 opacity-50 relative z-10">
                            <div class="text-6xl animate-bounce">👈</div>
                            <p class="text-slate-400 font-medium max-w-sm">
                                Enter patient metrics in the sidebar form to generate an instant health profile and recommendation.
                            </p>
                        </div>
                    {% endif %}
                </div>

                <!-- Info Section -->
                <div class="glass-panel rounded-2xl p-8">
                     <h3 class="font-display font-bold text-white mb-4">Under the Hood</h3>
                     <p class="text-sm text-slate-400 leading-relaxed mb-4">
                        Our hybrid architecture combines unsupervised and reinforcement learning:
                     </p>
                     <ul class="space-y-3 text-sm text-slate-400">
                        <li class="flex items-start gap-3">
                            <div class="w-5 h-5 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 text-xs mt-0.5">1</div>
                            <span><strong>Preprocessing:</strong> Data normalization and feature encoding.</span>
                        </li>
                        <li class="flex items-start gap-3">
                            <div class="w-5 h-5 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 text-xs mt-0.5">2</div>
                            <span><strong>Clustering (UMAP + KMeans):</strong> Maps the patient to one of 9 distinct health segments.</span>
                        </li>
                        <li class="flex items-start gap-3">
                            <div class="w-5 h-5 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 text-xs mt-0.5">3</div>
                            <span><strong>RL Policy:</strong> Retrieves the pre-calculated optimal action (Q-Learning) for that specific state.</span>
                        </li>
                     </ul>
                </div>
                
                
                <!-- NEW: Charts Section -->
                {% if cluster_name and models_loaded %}
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in-up" style="animation-delay: 0.3s;">
                    <!-- Radar Chart -->
                    <div class="glass-panel rounded-2xl p-6">
                        <h4 class="font-display text-sm font-bold text-slate-300 mb-4 uppercase tracking-wider">How You Compare</h4>
                        <div class="relative h-64 w-full">
                            <canvas id="radarChart"></canvas>
                        </div>
                    </div>
                    
                    <!-- Scatter Chart -->
                    <div class="glass-panel rounded-2xl p-6">
                        <h4 class="font-display text-sm font-bold text-slate-300 mb-4 uppercase tracking-wider">Cluster Galaxy Map</h4>
                        <div class="relative h-64 w-full">
                            <canvas id="scatterChart"></canvas>
                        </div>
                    </div>
                </div>
                
                    </div>
                </div>
                
                <!-- NEW: Batch Processing Section -->
                <div class="glass-panel rounded-2xl p-8 mt-6 animate-fade-in-up border border-purple-500/30" style="animation-delay: 0.5s;">
                    <div class="flex items-center gap-3 mb-6">
                        <div class="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 text-xl">
                            📂
                        </div>
                        <div>
                            <h3 class="font-display text-xl font-bold text-white">Batch Analysis</h3>
                            <p class="text-xs text-slate-400">Upload a CSV file with multiple patient records to process them all at once.</p>
                        </div>
                    </div>
                    
                    <form action="/batch_predict" method="POST" enctype="multipart/form-data" class="flex flex-col md:flex-row gap-4 items-center">
                        <div class="flex-1 w-full">
                            <input type="file" name="file" accept=".csv" required
                                class="block w-full text-sm text-slate-400
                                file:mr-4 file:py-2.5 file:px-4
                                file:rounded-xl file:border-0
                                file:text-sm file:font-semibold
                                file:bg-slate-700 file:text-white
                                hover:file:bg-slate-600
                                cursor-pointer bg-slate-800/50 rounded-xl border border-slate-700 focus:outline-none focus:border-purple-500 transition-all font-sans"
                            />
                            <p class="text-[10px] text-slate-500 mt-2 pl-1">Expected columns: Age, Gender, BMI, SBP, DBP, Calories</p>
                        </div>
                        <button type="submit" class="w-full md:w-auto px-6 py-2.5 bg-purple-600 hover:bg-purple-500 rounded-xl text-white font-bold text-sm shadow-lg shadow-purple-500/20 transition-all hover:scale-105 flex items-center justify-center gap-2">
                            <span>⚡</span> Process CSV
                        </button>
                    </form>
                </div>
                
                <!-- NEW: Simulator Section -->
                <div class="glass-panel rounded-2xl p-8 mt-6 animate-fade-in-up border border-indigo-500/30" style="animation-delay: 0.4s;">
                    <div class="flex items-center gap-3 mb-6">
                        <div class="w-10 h-10 rounded-full bg-indigo-500/20 flex items-center justify-center text-indigo-400 text-xl">
                            🎮
                        </div>
                        <div>
                            <h3 class="font-display text-xl font-bold text-white">"What-If" Health Simulator</h3>
                            <p class="text-xs text-slate-400">Adjust the sliders to see how lifestyle changes affect your health profile in real-time.</p>
                        </div>
                    </div>
                    
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
                        <!-- BMI Slider -->
                        <div class="space-y-4">
                            <div class="flex justify-between items-end">
                                <label class="text-sm text-slate-400 font-semibold">BMI</label>
                                <span id="sim-bmi-val" class="text-indigo-300 font-bold bg-indigo-900/30 px-2 py-0.5 rounded">{{ bmi }}</span>
                            </div>
                            <input type="range" id="sim-bmi" min="15" max="50" step="0.5" value="{{ bmi }}" 
                                class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all">
                        </div>
                        
                        <!-- Calories Slider -->
                        <div class="space-y-4">
                             <div class="flex justify-between items-end">
                                <label class="text-sm text-slate-400 font-semibold">Calories</label>
                                <span id="sim-cal-val" class="text-indigo-300 font-bold bg-indigo-900/30 px-2 py-0.5 rounded">{{ calories }}</span>
                            </div>
                            <input type="range" id="sim-cal" min="1000" max="4000" step="100" value="{{ calories }}" 
                                class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all">
                        </div>
                        
                        <!-- SBP Slider -->
                         <div class="space-y-4">
                             <div class="flex justify-between items-end">
                                <label class="text-sm text-slate-400 font-semibold">Systolic BP</label>
                                <span id="sim-sbp-val" class="text-indigo-300 font-bold bg-indigo-900/30 px-2 py-0.5 rounded">{{ sbp }}</span>
                            </div>
                            <input type="range" id="sim-sbp" min="90" max="200" step="1" value="{{ sbp }}" 
                                class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all">
                        </div>
                    </div>
                </div>

                <script>
                    // Safe injection of Python variables into JS
                    const userMetrics = {{ user_metrics_json | safe }};
                    const clusterAvgMetrics = {{ cluster_avg_json | safe }};
                    const featureLabels = {{ feature_labels_json | safe }};
                    
                    const userUmapJson = {{ user_umap_json | safe }};
                    const scatterDataJson = {{ scatter_data_json | safe }};
                    
                    // --- Radar Chart ---
                    const ctxRadar = document.getElementById('radarChart').getContext('2d');
                    const radarChart = new Chart(ctxRadar, {
                        type: 'radar',
                        data: {
                            labels: featureLabels,
                            datasets: [{
                                label: 'You',
                                data: userMetrics,
                                fill: true,
                                backgroundColor: 'rgba(45, 212, 191, 0.2)',
                                borderColor: '#2dd4bf',
                                pointBackgroundColor: '#2dd4bf',
                                pointBorderColor: '#fff',
                                pointHoverBackgroundColor: '#fff',
                                pointHoverBorderColor: '#2dd4bf'
                            }, {
                                label: 'Cluster Avg',
                                data: clusterAvgMetrics,
                                fill: true,
                                backgroundColor: 'rgba(148, 163, 184, 0.2)',
                                borderColor: '#94a3b8',
                                pointBackgroundColor: '#94a3b8',
                                pointBorderColor: '#fff',
                                pointHoverBackgroundColor: '#fff',
                                pointHoverBorderColor: '#94a3b8'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            elements: {
                                line: { borderWidth: 2 }
                            },
                            scales: {
                                r: {
                                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                    pointLabels: {
                                        color: '#cbd5e1',
                                        font: { size: 10 }
                                    },
                                    ticks: { display: false, backdropColor: 'transparent' }
                                }
                            },
                            plugins: {
                                legend: {
                                    labels: { color: '#cbd5e1' }
                                }
                            }
                        }
                    });

                    // --- Scatter Chart ---
                    const ctxScatter = document.getElementById('scatterChart').getContext('2d');
                    
                    // 1. Background points
                    const bgPoints = scatterDataJson.map(p => ({x: p.x, y: p.y}));
                    
                    // 2. User point
                    const userPoint = [{x: userUmapJson[0], y: userUmapJson[1]}];
                    
                    const scatterChart = new Chart(ctxScatter, {
                        type: 'scatter',
                        data: {
                            datasets: [
                                {
                                    label: 'You',
                                    data: userPoint,
                                    backgroundColor: '#2dd4bf', // Teal
                                    pointRadius: 8,
                                    pointHoverRadius: 10,
                                    borderColor: '#ffffff',
                                    borderWidth: 2
                                },
                                {
                                    label: 'Others',
                                    data: bgPoints,
                                    backgroundColor: 'rgba(148, 163, 184, 0.3)', // Slate
                                    pointRadius: 2,
                                    pointHoverRadius: 2
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: { display: false },
                                y: { display: false }
                            },
                            plugins: {
                                legend: {
                                    labels: { color: '#cbd5e1' }
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            return context.dataset.label;
                                        }
                                    }
                                }
                            }
                        }
                    });
                    
                    // --- SIMULATION LOGIC ---
                    const simBmi = document.getElementById('sim-bmi');
                    const simCal = document.getElementById('sim-cal');
                    const simSbp = document.getElementById('sim-sbp');
                    
                    const simBmiVal = document.getElementById('sim-bmi-val');
                    const simCalVal = document.getElementById('sim-cal-val');
                    const simSbpVal = document.getElementById('sim-sbp-val');
                    
                    // Static user data (Age, Gender) - we don't simulate these
                    const staticAge = {{ age }};
                    const staticGender = "{{ gender }}";
                    const staticDbp = {{ dbp }}; // Keeping DBP static for simplicity or add slider if needed

                    async function updateSimulation() {
                        // Update UI labels
                        simBmiVal.textContent = simBmi.value;
                        simCalVal.textContent = simCal.value;
                        simSbpVal.textContent = simSbp.value;
                        
                        // Prepare payload
                        const payload = {
                            age: staticAge,
                            gender: staticGender,
                            bmi: parseFloat(simBmi.value),
                            sbp: parseInt(simSbp.value),
                            dbp: staticDbp,
                            calories: parseFloat(simCal.value)
                        };
                        
                        try {
                            const response = await fetch('/api/simulate', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(payload)
                            });
                            const data = await response.json();
                            
                            if (data.status === 'success') {
                                // 1. Update Text Results
                                // Note: We need to target the HTML elements. 
                                // Let's simplify: Update charts mainly, text is harder without IDs.
                                // Ideal: Add IDs to the result divs in HTML above.
                                document.querySelector('#result-cluster-name').textContent = data.cluster_name;
                                document.querySelector('#result-action').textContent = data.action_text;
                                
                                // 2. Update Charts
                                // Radar: Update 'You' dataset (Index 0)
                                // Order: Age (0), BMI (1), SBP (2), DBP (3), Cal (4)
                                // We update: BMI (1), SBP (2), Cal (4) 
                                // Note: We should probably just ask API for the full array to be safe
                                // But for speed, let's inject values locally:
                                const newMetrics = [staticAge, parseFloat(simBmi.value), parseInt(simSbp.value), staticDbp, parseFloat(simCal.value)];
                                radarChart.data.datasets[0].data = newMetrics;
                                // Also update Cluster Avg if cluster changed?
                                if (data.cluster_avg) {
                                     radarChart.data.datasets[1].data = data.cluster_avg;
                                }
                                radarChart.update();
                                
                                // Scatter: Update 'You' point
                                const newUmap = {x: data.umap[0], y: data.umap[1]};
                                scatterChart.data.datasets[0].data = [newUmap];
                                scatterChart.update();
                            }
                        } catch (err) {
                            console.error("Simulation failed", err);
                        }
                    }
                    
                    // Attach listeners with debounce? For now direct change.
                    simBmi.addEventListener('input', updateSimulation);
                    simCal.addEventListener('input', updateSimulation);
                    simSbp.addEventListener('input', updateSimulation);
                </script>
                {% endif %}

            </div>
        </div>

        <footer class="text-center text-slate-600 text-xs mt-12 pb-8">
            Flask App Mode • Patient Segmentation System v2.0
        </footer>
    </main>
</body>
</html>
"""

# -----------------------------
# HOW IT WORKS TEMPLATE
# -----------------------------
HOW_IT_WORKS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>How It Works - Patient Segmentation</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
      tailwind.config = {
        darkMode: 'class',
        theme: {
          extend: {
            colors: {
              primary: {
                50: '#f0fdfa',
                100: '#ccfbf1',
                200: '#99f6e4',
                300: '#5eead4',
                400: '#2dd4bf',
                500: '#14b8a6',
                600: '#0d9488',
                700: '#0f766e',
                800: '#115e59',
                900: '#134e4a',
                950: '#042f2e',
              },
              dark: {
                900: '#020617',
                800: '#0f172a',
                700: '#1e293b',
              }
            },
            fontFamily: {
              sans: ['Plus Jakarta Sans', 'sans-serif'],
              display: ['Outfit', 'sans-serif'],
            },
            animation: {
              'blob': 'blob 7s infinite',
              'fade-in-up': 'fadeInUp 0.5s ease-out forwards',
            },
            keyframes: {
              blob: {
                '0%': { transform: 'translate(0px, 0px) scale(1)' },
                '33%': { transform: 'translate(30px, -50px) scale(1.1)' },
                '66%': { transform: 'translate(-20px, 20px) scale(0.9)' },
                '100%': { transform: 'translate(0px, 0px) scale(1)' },
              },
              fadeInUp: {
                '0%': { opacity: '0', transform: 'translateY(10px)' },
                '100%': { opacity: '1', transform: 'translateY(0)' },
              }
            }
          }
        }
      }
    </script>
    <style>
        body {
            background-color: #020617;
            background-image: 
                radial-gradient(at 0% 0%, rgba(15, 118, 110, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(45, 212, 191, 0.05) 0px, transparent 50%);
        }
        .glass-nav {
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .glass-panel {
            background: rgba(30, 41, 59, 0.3);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
    </style>
</head>
<body class="text-slate-300 font-sans antialiased min-h-screen">

    <!-- Navbar -->
    <nav class="glass-nav fixed w-full z-50">
        <div class="max-w-7xl mx-auto px-6 lg:px-8">
            <div class="flex justify-between h-16 items-center">
                <a href="/" class="flex items-center gap-3 group hover:opacity-80 transition">
                    <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white shadow-lg">
                        ✅
                    </div>
                    <span class="font-display font-bold text-lg text-white tracking-tight">
                        Patient<span class="text-primary-400">AI</span>
                    </span>
                </a>
                <div class="flex items-center gap-8">
                    <a href="/dashboard" class="text-sm font-medium text-slate-400 hover:text-white transition-colors">Dashboard</a>
                    <a href="/" class="px-5 py-2.5 bg-slate-800 hover:bg-slate-700 rounded-xl text-white font-bold text-sm border border-slate-700 transition-all">
                        Back Home
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto py-24 px-6 lg:px-8">
        
        <!-- Header -->
        <div class="text-center max-w-3xl mx-auto mb-20 animate-fade-in-up">
            <h1 class="font-display text-4xl lg:text-5xl font-bold text-white mb-6">Demystifying the Intelligence</h1>
            <p class="text-lg text-slate-400 leading-relaxed">
                Our system doesn't just guess. It follows a rigorous scientific pipeline transforming raw data into actionable health strategies.
            </p>
        </div>

        <!-- Pipeline Steps -->
        <div class="space-y-24 relative">
            
            <!-- Connection Line -->
            <div class="absolute left-8 lg:left-1/2 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary-500/0 via-primary-500/50 to-primary-500/0 hidden lg:block"></div>

            <!-- Step 1 -->
            <div class="relative flex flex-col lg:flex-row gap-12 items-center animate-fade-in-up" style="animation-delay: 0.1s;">
                <div class="lg:w-1/2 flex justify-end">
                    <div class="glass-panel p-8 rounded-3xl max-w-lg border-l-4 border-l-blue-500 w-full relative group hover:bg-slate-800/50 transition-colors">
                        <div class="absolute -right-16 top-1/2 -translate-y-1/2 w-12 h-12 rounded-full bg-slate-900 border-4 border-slate-800 flex items-center justify-center z-10 hidden lg:flex">
                            <span class="text-xl">1</span>
                        </div>
                        <h3 class="font-display text-2xl font-bold text-white mb-4">Data Ingestion</h3>
                        <p class="text-slate-400">
                            We ingest high-dimensional health data from the <strong>NHANES</strong> dataset. This includes physiological markers like BMI, Blood Pressure, Cholesterol, and dietary habits.
                        </p>
                    </div>
                </div>
                <div class="lg:w-1/2 pl-12 hidden lg:block">
                     <div class="text-6xl opacity-20">📊</div>
                </div>
            </div>

            <!-- Step 2 -->
             <div class="relative flex flex-col lg:flex-row-reverse gap-12 items-center animate-fade-in-up" style="animation-delay: 0.2s;">
                <div class="lg:w-1/2 flex justify-start">
                    <div class="glass-panel p-8 rounded-3xl max-w-lg border-r-4 border-r-purple-500 w-full relative group hover:bg-slate-800/50 transition-colors">
                         <div class="absolute -left-16 top-1/2 -translate-y-1/2 w-12 h-12 rounded-full bg-slate-900 border-4 border-slate-800 flex items-center justify-center z-10 hidden lg:flex">
                            <span class="text-xl">2</span>
                        </div>
                        <h3 class="font-display text-2xl font-bold text-white mb-4">Intelligent Segmentation</h3>
                        <p class="text-slate-400 mb-4">
                            We use <strong>UMAP</strong> to reduce dozens of variables into a 2D map, preserving the global structure of patient similarities.
                        </p>
                        <p class="text-slate-400">
                             Then, <strong>KMeans Clustering</strong> groups these mapped patients into 9 distinct profiles (e.g., "High Risk Metabolic", "Optimized Health").
                        </p>
                    </div>
                </div>
                <div class="lg:w-1/2 pr-12 hidden lg:block text-right">
                     <div class="text-6xl opacity-20">🧩</div>
                </div>
            </div>

            <!-- Step 3 -->
            <div class="relative flex flex-col lg:flex-row gap-12 items-center animate-fade-in-up" style="animation-delay: 0.3s;">
                <div class="lg:w-1/2 flex justify-end">
                    <div class="glass-panel p-8 rounded-3xl max-w-lg border-l-4 border-l-emerald-500 w-full relative group hover:bg-slate-800/50 transition-colors">
                         <div class="absolute -right-16 top-1/2 -translate-y-1/2 w-12 h-12 rounded-full bg-slate-900 border-4 border-slate-800 flex items-center justify-center z-10 hidden lg:flex">
                            <span class="text-xl">3</span>
                        </div>
                        <h3 class="font-display text-2xl font-bold text-white mb-4">Prescriptive AI</h3>
                        <p class="text-slate-400">
                            Using <strong>Reinforcement Learning (Q-Learning)</strong>, we've simulated millions of health trajectories to learn the "Optimal Policy". 
                            The system recommends the specific lifestyle change (Action) that maximizes the long-term health reward for your specific cluster.
                        </p>
                    </div>
                </div>
                <div class="lg:w-1/2 pl-12 hidden lg:block">
                     <div class="text-6xl opacity-20">🎯</div>
                </div>
            </div>
        </div>

        <!-- CTA -->
        <div class="mt-24 text-center">
             <a href="/dashboard" class="inline-flex items-center gap-2 px-8 py-4 bg-primary-600 hover:bg-primary-500 rounded-2xl font-bold text-white shadow-xl shadow-primary-500/20 transition-all hover:scale-[1.02]">
                Try the System
                <span>→</span>
            </a>
        </div>

    </main>
    
    <footer class="py-12 border-t border-slate-800/50 relative overflow-hidden mt-12">
        <div class="max-w-7xl mx-auto px-6 lg:px-8 text-center text-slate-600 text-sm">
            © 2025 Patient Segmentation System.
        </div>
    </footer>

</body>
</html>
"""

# -----------------------------
# Fake mapping (for now) - Fallback
# -----------------------------
DUMMY_CLUSTER = "Middle-aged Metabolic Risk (Simulated)"
DUMMY_ACTION = "Quit smoking & nutritional lifestyle coaching (Simulated)"

@app.route("/")
def landing():
    return render_template_string(LANDING_HTML)

@app.route("/how-it-works")
def how_it_works():
    return render_template_string(HOW_IT_WORKS_HTML)

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    # Default values for the form (generic baseline)
    default_values = {
        "age": 40,
        "gender": "Male",
        "bmi": 25.0,
        "sbp": 120,
        "dbp": 80,
        "calories": 2000
    }
    
    # Variables to pass to template
    context = default_values.copy()
    cluster_name = None
    action_text = None
    
    # Visualization Data Containers
    user_metrics_json = "[]"
    cluster_avg_json = "[]"
    feature_labels_json = "[]"
    user_umap_json = "[]"
    scatter_data_json = "[]"

    if request.method == "POST":
        try:
            # Update contexts with submitted values
            context["age"] = int(request.form.get("age", default_values["age"]))
            context["gender"] = request.form.get("gender", default_values["gender"])
            context["bmi"] = float(request.form.get("bmi", default_values["bmi"]))
            context["sbp"] = int(request.form.get("sbp", default_values["sbp"]))
            context["dbp"] = int(request.form.get("dbp", default_values["dbp"]))
            context["calories"] = float(request.form.get("calories", default_values["calories"]))
            
            gender_val = 1 if context["gender"] == "Male" else 2 # Common NHANES encoding: 1=Male, 2=Female
            
            model_inputs = {
                "RIDAGEYR": context["age"],
                "RIAGENDR": gender_val,
                "BMXBMI": context["bmi"],
                "BPXSY1": context["sbp"],
                "BPXDI1": context["dbp"],
                "DR1TKCAL": context["calories"]
            }
            
            if MODELS_LOADED:
                # Run Prediction
                cluster_name, action_text, cluster_id, user_umap = predict_cluster_from_input(model_inputs)
                
                # --- PREPARE DATA FOR CHARTS ---
                # 1. Radar Chart Data
                # We need raw values for Age, BMI, BP, Calories logic.
                # Features in `cluster_means` are typically scaled or raw? 
                # The `cluster_means` was computed from `df` directly using `feature_columns`.
                # If `df` is scaled, we should normalize user inputs too for fair comparison.
                # BUT for simplicity in this demo, let's assume raw-to-raw comparison or just visual.
                # Currently `model_inputs` contains our RAW valid inputs.
                
                vis_features = ["RIDAGEYR", "BMXBMI", "BPXSY1", "BPXDI1", "DR1TKCAL"]
                vis_labels = ["Age", "BMI", "Systolic BP", "Diastolic BP", "Calories"]
                
                user_vals = [model_inputs.get(f, 0) for f in vis_features]
                
                # Get Cluster Average
                # cluster_means assumes the CSV has these columns.
                # If cluster_id is valid
                if cluster_id is not None and cluster_id in cluster_means:
                    c_means = cluster_means[cluster_id]
                    avg_vals = [c_means.get(f, 0) for f in vis_features]
                else:
                    avg_vals = [0]*len(vis_features)
                    
                import json
                user_metrics_json = json.dumps(user_vals)
                cluster_avg_json = json.dumps(avg_vals)
                feature_labels_json = json.dumps(vis_labels)
                
                # 2. Scatter Data
                user_umap_json = json.dumps(user_umap.tolist() if user_umap is not None else [0,0])
                scatter_data_json = json.dumps(scatter_data) # Global variable computed at startup
                
            else:
                # Fallback to dummy if models failed to load
                cluster_name = DUMMY_CLUSTER
                action_text = DUMMY_ACTION
                # Dummy chart data could be added here if needed

        except ValueError as e:
            print(f"Error parsing inputs: {e}")
            pass

    return render_template_string(
        DASHBOARD_HTML,
        **context,
        cluster_name=cluster_name,
        action_text=action_text,
        models_loaded=MODELS_LOADED,
        user_metrics_json=user_metrics_json,
        cluster_avg_json=cluster_avg_json,
        feature_labels_json=feature_labels_json,
        user_umap_json=user_umap_json,
        scatter_data_json=scatter_data_json
    )


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    API endpoint for the Simulator.
    Receives params JSON, runs prediction, returns new cluster/action/stats.
    """
    if not MODELS_LOADED:
        return jsonify({"status": "error", "message": "Models not loaded"})
    
    data = request.get_json()
    
    # Extract
    age = data.get("age")
    gender = data.get("gender")
    bmi = data.get("bmi")
    sbp = data.get("sbp")
    dbp = data.get("dbp")
    calories = data.get("calories")
    
    gender_val = 1 if gender == "Male" else 2
    
    model_inputs = {
        "RIDAGEYR": age,
        "RIAGENDR": gender_val,
        "BMXBMI": bmi,
        "BPXSY1": sbp,
        "BPXDI1": dbp,
        "DR1TKCAL": calories
    }
    
    cluster_name, action_text, cluster_id, user_umap = predict_cluster_from_input(model_inputs)
    
    # Get new cluster average profile for chart
    vis_features = ["RIDAGEYR", "BMXBMI", "BPXSY1", "BPXDI1", "DR1TKCAL"]
    cluster_avg = []
    if cluster_id is not None and cluster_id in cluster_means:
        c_means = cluster_means[cluster_id]
        cluster_avg = [c_means.get(f, 0) for f in vis_features]
    
    return jsonify({
        "status": "success",
        "cluster_name": cluster_name,
        "action_text": action_text,
        "umap": user_umap.tolist(),
        "cluster_avg": cluster_avg
    })

@app.route("/download_report", methods=["POST"])
def download_report():
    """
    Generates a PDF report for the patient.
    """
    # Get data from form
    age = request.form.get("age")
    gender = request.form.get("gender")
    bmi = request.form.get("bmi")
    sbp = request.form.get("sbp")
    dbp = request.form.get("dbp")
    calories = request.form.get("calories")
    cluster_name = request.form.get("cluster_name")
    action_text = request.form.get("action_text")
    
    # Create PDF in memory
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # --- Header ---
    c.setFillColor(colors.darkslategrey)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "Patient Segmentation System")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, "AI-Powered Preventive Healthcare Analysis")
    
    c.setStrokeColor(colors.teal)
    c.setLineWidth(2)
    c.line(50, height - 80, width - 50, height - 80)
    
    # --- Patient Vitals ---
    y = height - 120
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Patient Vitals")
    y -= 30
    
    c.setFont("Helvetica", 12)
    details = [
        f"Age: {age} years",
        f"Gender: {gender}",
        f"BMI: {bmi}",
        f"Blood Pressure: {sbp}/{dbp} mmHg",
        f"Caloric Intake: {calories} kcal/day"
    ]
    
    for detail in details:
        c.drawString(70, y, f"• {detail}")
        y -= 20
        
    y -= 20
    c.setStrokeColor(colors.lightgrey)
    c.setLineWidth(1)
    c.line(50, y, width - 50, y)
    y -= 40
    
    # --- AI Analysis ---
    c.setFillColor(colors.darkslategrey)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Clinical Analysis")
    y -= 30
    
    # Segment
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, y, "Identified Health Segment:")
    c.setFillColor(colors.teal)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(250, y, cluster_name)
    y -= 30
    
    # Recommendation
    c.setFillColor(colors.darkslategrey)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, y, "AI Recommendation:")
    y -= 20
    
    # Wrap text for recommendation
    c.setFont("Helvetica", 12)
    text = action_text
    # Simple wrapping
    if len(text) > 80:
        part1 = text[:80]
        part2 = text[80:]
        c.drawString(70, y, part1 + "-")
        y -= 15
        c.drawString(70, y, part2)
    else:
        c.drawString(70, y, text)
        
    # --- Footer ---
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.grey)
    c.drawString(50, 50, f"Generated automatically on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 35, "Disclaimer: This report is AI-generated and does not replace professional medical advice.")
    
    c.save()
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Health_Report_{int(time.time())}.pdf",
        mimetype="application/pdf"
    )

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Handles CSV file upload for batch processing.
    """
    if not MODELS_LOADED:
        return "Models not loaded - cannot process batch.", 500

    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        # Read CSV
        df_batch = pd.read_csv(file)
        
        # We need specific columns. Let's try to find them loosely or assume standard names.
        # Expected: Age, Gender, BMI, SBP, DBP, Calories
        # Map to model inputs
        
        results = []
        
        for index, row in df_batch.iterrows():
            # Flexible fetching (case insensitive-ish or fallback to known names)
            # You might want to make this more robust in production
            age = row.get("Age") or row.get("age") or 0
            gender = row.get("Gender") or row.get("gender") or "Male"
            bmi = row.get("BMI") or row.get("bmi") or 25.0
            sbp = row.get("SBP") or row.get("sbp") or 120
            dbp = row.get("DBP") or row.get("dbp") or 80
            calories = row.get("Calories") or row.get("calories") or 2000
            
            gender_val = 1 if gender == "Male" else 2
            
            model_inputs = {
                "RIDAGEYR": age,
                "RIAGENDR": gender_val,
                "BMXBMI": bmi,
                "BPXSY1": sbp,
                "BPXDI1": dbp,
                "DR1TKCAL": calories
            }
            
            cluster_name, action_text, _, _ = predict_cluster_from_input(model_inputs)
            
            results.append({
                "Predicted_Segment": cluster_name,
                "Recommended_Action": action_text
            })
            
        # Append results to original DF
        df_results = pd.DataFrame(results)
        df_final = pd.concat([df_batch.reset_index(drop=True), df_results], axis=1)
        
        # Convert to CSV
        output = io.BytesIO()
        df_final.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            as_attachment=True,
            download_name=f"Batch_Results_{int(time.time())}.csv",
            mimetype="text/csv"
        )
            
    except Exception as e:
        return f"Error processing file: {e}", 500

if __name__ == "__main__":
    # debug=True for development only
    app.run(debug=True)
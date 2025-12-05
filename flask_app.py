from flask import Flask, render_template_string, request

app = Flask(__name__)

# -----------------------------
# Tailwind + basic HTML template
# -----------------------------
BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Patient Segmentation – Flask Demo</title>
    <!-- TailwindCSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>

    <!-- Optional: Custom Tailwind config -->
    <script>
      tailwind.config = {
        theme: {
          extend: {
            colors: {
              primary: "#0f766e",
              primaryDark: "#0f172a",
            }
          }
        }
      }
    </script>
</head>
<body class="bg-slate-100 min-h-screen">
    <!-- Top navbar -->
    <nav class="bg-primaryDark text-slate-100 px-6 py-4 flex justify-between items-center shadow">
        <div class="flex items-center gap-2">
            <span class="text-xl">🩺</span>
            <span class="font-semibold tracking-wide">Patient Segmentation System</span>
        </div>
        <div class="text-sm text-slate-300">
            Flask + Tailwind Demo
        </div>
    </nav>

    <!-- Main content -->
    <main class="max-w-5xl mx-auto py-10 px-4">
        <div class="bg-white rounded-xl shadow-md p-6 mb-6">
            <h1 class="text-2xl font-bold text-slate-800 mb-2">
                NHANES-based Preventive Health Dashboard (Flask UI)
            </h1>
            <p class="text-slate-600 text-sm">
                This is a <span class="font-semibold">Flask</span> frontend using 
                <span class="font-semibold">TailwindCSS</span>. 
                You can later connect this to your clustering + RL models.
            </p>
        </div>

        <!-- Quick stats / highlights -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div class="bg-white rounded-xl shadow-sm p-4 flex flex-col gap-1 transition hover:shadow-lg hover:scale-[1.02]">
                <div class="text-xs uppercase tracking-wide text-slate-500 font-semibold">
                    Clusters (Demo)
                </div>
                <div class="text-2xl font-bold text-primaryDark">
                    9
                </div>
                <p class="text-xs text-slate-500">
                    Example health segments such as “Healthy Youth” and “Metabolic Risk”.
                </p>
            </div>
            <div class="bg-white rounded-xl shadow-sm p-4 flex flex-col gap-1 transition hover:shadow-lg hover:scale-[1.02]">
                <div class="text-xs uppercase tracking-wide text-slate-500 font-semibold">
                    Data Sources
                </div>
                <div class="text-2xl font-bold text-primaryDark">
                    5+
                </div>
                <p class="text-xs text-slate-500">
                    Demographics, labs, diet, examination, medications, questionnaire.
                </p>
            </div>
            <div class="bg-white rounded-xl shadow-sm p-4 flex flex-col gap-1 transition hover:shadow-lg hover:scale-[1.02]">
                <div class="text-xs uppercase tracking-wide text-slate-500 font-semibold">
                    ML Techniques
                </div>
                <div class="text-2xl font-bold text-primaryDark">
                    UMAP + KMeans + RL
                </div>
                <p class="text-xs text-slate-500">
                    Unsupervised clustering plus Q-learning for preventive actions.
                </p>
            </div>
        </div>

        <!-- Pipeline chips -->
        <div class="flex flex-wrap gap-2 mb-8 text-xs">
            <span class="px-3 py-1 rounded-full bg-slate-200 text-slate-700">Data Cleaning</span>
            <span class="px-3 py-1 rounded-full bg-slate-200 text-slate-700">Feature Engineering</span>
            <span class="px-3 py-1 rounded-full bg-slate-200 text-slate-700">UMAP Dimensionality Reduction</span>
            <span class="px-3 py-1 rounded-full bg-slate-200 text-slate-700">KMeans Clustering</span>
            <span class="px-3 py-1 rounded-full bg-slate-200 text-slate-700">Q-Learning Policy</span>
            <span class="px-3 py-1 rounded-full bg-slate-200 text-slate-700">Health Recommendation</span>
        </div>

        <!-- Two-column layout -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <!-- Left: form -->
            <div class="md:col-span-1 bg-white rounded-xl shadow p-5">
                <h2 class="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
                    <span>👤</span> 
                    <span>Enter Patient Snapshot</span>
                </h2>
                <form method="POST" class="space-y-3">
                    <div>
                        <label class="block text-sm text-slate-600 mb-1">Age (years)</label>
                        <div class="relative">
                            <span class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400 text-xs">
                                👤
                            </span>
                            <input 
                                type="number" name="age" min="1" max="90" value="{{ age }}"
                                class="w-full rounded border border-slate-300 outline outline-1 outline-slate-400 focus:outline-primary focus:ring-2 focus:ring-primary text-sm pl-8 pr-3 py-2 bg-white"
                                required
                            />
                        </div>
                    </div>

                    <div>
                        <label class="block text-sm text-slate-600 mb-1">Gender</label>
                        <div class="relative">
                            <span class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400 text-xs">
                                ⚧
                            </span>
                            <select 
                                name="gender"
                                class="w-full rounded border border-slate-300 outline outline-1 outline-slate-400 focus:outline-primary focus:ring-2 focus:ring-primary text-sm pl-8 pr-3 py-2 bg-white"
                            >
                                <option value="Male" {% if gender == "Male" %}selected{% endif %}>Male</option>
                                <option value="Female" {% if gender == "Female" %}selected{% endif %}>Female</option>
                            </select>
                        </div>
                    </div>

                    <div>
                        <label class="block text-sm text-slate-600 mb-1">BMI</label>
                        <div class="relative">
                            <span class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400 text-xs">
                                ⚖️
                            </span>
                            <input 
                                type="number" step="0.1" min="10" max="60" name="bmi" value="{{ bmi }}"
                                class="w-full rounded border border-slate-300 outline outline-1 outline-slate-400 focus:outline-primary focus:ring-2 focus:ring-primary text-sm pl-8 pr-3 py-2 bg-white"
                            />
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-sm text-slate-600 mb-1">Systolic BP</label>
                            <div class="relative">
                                <span class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400 text-xs">
                                    💓
                                </span>
                                <input 
                                    type="number" min="80" max="220" name="sbp" value="{{ sbp }}"
                                    class="w-full rounded border border-slate-300 outline outline-1 outline-slate-400 focus:outline-primary focus:ring-2 focus:ring-primary text-sm pl-8 pr-3 py-2 bg-white"
                                />
                            </div>
                            <p class="text-[10px] text-red-500 mt-1">Value must be between 80–220</p>
                        </div>
                        <div>
                            <label class="block text-sm text-slate-600 mb-1">Diastolic BP</label>
                            <div class="relative">
                                <span class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400 text-xs">
                                    💗
                                </span>
                                <input 
                                    type="number" min="40" max="130" name="dbp" value="{{ dbp }}"
                                    class="w-full rounded border border-slate-300 outline outline-1 outline-slate-400 focus:outline-primary focus:ring-2 focus:ring-primary text-sm pl-8 pr-3 py-2 bg-white"
                                />
                            </div>
                            <p class="text-[10px] text-red-500 mt-1">Value must be between 40–130</p>
                        </div>
                    </div>

                    <div>
                      <label class="block text-sm text-slate-600 mb-1">Daily Calories</label>
                      <div class="relative">
                          <span class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400 text-xs">
                              🔥
                          </span>
                          <input 
                              type="number" min="500" max="6000" name="calories"
                              class="w-full rounded border border-slate-300 outline outline-1 outline-slate-400 focus:outline-primary focus:ring-2 focus:ring-primary text-sm pl-8 pr-3 py-2 bg-white"
                          />
                      </div>
                      <p class="text-[10px] text-red-500 mt-1">Recommended range: 500–6000 kcal/day</p>
                    </div>

                    <div>
                      <label class="block text-sm text-slate-600 mb-1">Sugar Intake (g/day)</label>
                      <div class="relative">
                          <span class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400 text-xs">
                              🍭
                          </span>
                          <input 
                              type="number" min="0" max="500" name="sugar"
                              class="w-full rounded border border-slate-300 outline outline-1 outline-slate-400 focus:outline-primary focus:ring-2 focus:ring-primary text-sm pl-8 pr-3 py-2 bg-white"
                          />
                      </div>
                      <p class="text-[10px] text-red-500 mt-1">Typical range: 0–500 g/day</p>
                    </div>

                    <div>
                      <label class="block text-sm text-slate-600 mb-1">Alcohol Intake (g/day)</label>
                      <div class="relative">
                          <span class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400 text-xs">
                              🍺
                          </span>
                          <input 
                              type="number" min="0" max="300" name="alcohol"
                              class="w-full rounded border border-slate-300 outline outline-1 outline-slate-400 focus:outline-primary focus:ring-2 focus:ring-primary text-sm pl-8 pr-3 py-2 bg-white"
                          />
                      </div>
                      <p class="text-[10px] text-red-500 mt-1">Range: 0–300 g/day</p>
                    </div>

                    <button 
                        type="submit"
                        class="w-full mt-3 bg-primary hover:bg-emerald-700 text-white text-sm font-medium py-2.5 rounded-lg shadow-sm transition"
                    >
                        Get Dummy Recommendation
                    </button>
                </form>
            </div>

            <!-- Right: results -->
            <div class="md:col-span-2 space-y-4">
                <div class="bg-white rounded-xl shadow p-5">
                    <h2 class="text-lg font-semibold text-slate-800 mb-3 flex items-center gap-2">
                        <span>📌</span>
                        <span>Predicted Cluster (Demo)</span>
                    </h2>

                    {% if cluster_name %}
                        <p class="text-sm text-slate-700 mb-2">
                            This is a <span class="font-semibold">dummy</span> result for now. 
                            Later, you will replace this logic with your **UMAP + KMeans + RL** pipeline.
                        </p>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                            <div class="bg-emerald-50 border border-emerald-100 rounded-lg p-3">
                                <div class="text-xs uppercase tracking-wide text-emerald-700 font-semibold">
                                    Cluster
                                </div>
                                <div class="text-sm mt-1 font-medium text-emerald-900">
                                    {{ cluster_name }}
                                </div>
                            </div>
                            <div class="bg-sky-50 border border-sky-100 rounded-lg p-3">
                                <div class="text-xs uppercase tracking-wide text-sky-700 font-semibold">
                                    Recommended Action
                                </div>
                                <div class="text-sm mt-1 font-medium text-sky-900">
                                    {{ action_text }}
                                </div>
                            </div>
                        </div>
                    {% else %}
                        <p class="text-sm text-slate-600">
                            Submit the form on the left to see a sample cluster label and recommendation.
                        </p>
                    {% endif %}
                </div>

                <div class="bg-white rounded-xl shadow p-5">
                    <h2 class="text-lg font-semibold text-slate-800 mb-2 flex items-center gap-2">
                        <span>🧠</span>
                        <span>How this will connect to your ML pipeline</span>
                    </h2>
                    <ul class="list-disc list-inside text-sm text-slate-700 space-y-1">
                        <li>Take the user inputs and build a feature row (Age, BMI, BP, diet, etc.).</li>
                        <li>Apply the same preprocessing (encoding, scaling) you used in your notebooks.</li>
                        <li>Use your saved <code>umap_model</code> and <code>kmeans</code> to get the cluster.</li>
                        <li>Look up RL policy: <code>cluster_to_learned_action[cluster_id]</code> → text from <code>actions</code>.</li>
                        <li>Render the final recommendation here instead of the dummy one.</li>
                    </ul>
                </div>

                <div class="bg-white rounded-xl shadow p-5">
                    <h2 class="text-lg font-semibold text-slate-800 mb-2 flex items-center gap-2">
                        <span>📊</span>
                        <span>Example Cluster Profiles (Static Demo)</span>
                    </h2>
                    <p class="text-sm text-slate-600 mb-3">
                        Below are sample descriptions of a few patient segments you might discover with your clustering:
                    </p>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                        <div class="border rounded-lg p-3 bg-slate-50">
                            <div class="font-semibold text-slate-800 mb-1">Healthy Youth</div>
                            <p class="text-slate-600 text-xs">
                                Young, normal BMI, good blood pressure, low medication usage. Focus on maintaining healthy habits.
                            </p>
                        </div>
                        <div class="border rounded-lg p-3 bg-slate-50">
                            <div class="font-semibold text-slate-800 mb-1">Middle-aged Metabolic Risk</div>
                            <p class="text-slate-600 text-xs">
                                Higher BMI and BP, elevated cholesterol. Emphasis on weight management and sodium reduction.
                            </p>
                        </div>
                        <div class="border rounded-lg p-3 bg-slate-50">
                            <div class="font-semibold text-slate-800 mb-1">Alcohol-associated Risk</div>
                            <p class="text-slate-600 text-xs">
                                Moderate-to-high alcohol intake with metabolic risk factors. Key action is structured alcohol reduction.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer class="text-xs text-slate-500 text-center mt-6">
            Built with Flask + TailwindCSS • Backed by NHANES-based clustering & Q-learning policy.
        </footer>
    </main>
</body>
</html>
"""

# -----------------------------
# Fake mapping (for now)
# Later you'll import real ones from joblib like in Streamlit app
# -----------------------------
DUMMY_CLUSTER = "Middle-aged Metabolic Risk"
DUMMY_ACTION = "Quit smoking & nutritional lifestyle coaching"


@app.route("/", methods=["GET", "POST"])
def index():
    # Default values for the form
    age = 40
    gender = "Male"
    bmi = 25.0
    sbp = 120
    dbp = 80

    cluster_name = None
    action_text = None

    if request.method == "POST":
        try:
            age = int(request.form.get("age", age))
            gender = request.form.get("gender", gender)
            bmi = float(request.form.get("bmi", bmi))
            sbp = int(request.form.get("sbp", sbp))
            dbp = int(request.form.get("dbp", dbp))
        except ValueError:
            # just keep defaults if parse fails
            pass

        # ❗️For now we just hardcode output.
        # Later: plug in your UMAP + KMeans + RL pipeline here.
        cluster_name = DUMMY_CLUSTER
        action_text = DUMMY_ACTION

    return render_template_string(
        BASE_HTML,
        age=age,
        gender=gender,
        bmi=bmi,
        sbp=sbp,
        dbp=dbp,
        cluster_name=cluster_name,
        action_text=action_text,
    )


if __name__ == "__main__":
    # debug=True for development only
    app.run(debug=True)
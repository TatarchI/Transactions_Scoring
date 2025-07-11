"""
📦 Module4_tune_isoforest_contamination.py
Author: Ivan Tatarchuk

Description:
- Loads the final cleaned treasury transactions dataset.
- Applies RobustScaler for robust normalization across all features.
- Prepares data for contamination (outlier proportion) tuning in Isolation Forest.
- Sets up a clean, reproducible structure for systematic anomaly threshold calibration.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
import os

# ----------------------------------------
# 📌 Output directory preparation
# ----------------------------------------
output_dir = 'check_files/isoforest_tuning'
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------
# 📌 Load dataset
# ----------------------------------------
print("✅ Loading dataset for tuning...")
df = pd.read_csv('check_files/transactions_dataset_final_cleaned.csv.gz', compression='gzip')

# Drop 'ID' and 'SUMPAY' (we decided to exclude these from modeling)
df_features = df.drop(columns=['ID', 'SUMPAY'])

# Apply RobustScaler (robust to outliers) to all features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_features)
print("✅ Data normalized using RobustScaler.")

# ----------------------------------------
# 📌 Why we use subsampling:
# ----------------------------------------
# Silhouette Score requires O(N^2) memory and time on large datasets (737,000+ rows).
# To speed up tuning without losing quality, we use a random subsample of 10,000 rows
# (taken from X_scaled and corresponding preds) when computing silhouette_score.
# The model is trained on the entire dataset; subsampling is only for metric calculation.

sample_size = 10000  # increase if sufficient RAM is available
random_state = 42

# ----------------------------------------
# 📌 Parameter grid for tuning
# ----------------------------------------
n_estimators_list = [50, 100, 150]
contamination_list = [0.0175, 0.02, 0.0225, 0.025]
max_samples_list = [0.8, 0.5]
max_features_list = [1.0, 0.8, 0.5]

results = []

total_combinations = (
    len(n_estimators_list)
    * len(contamination_list)
    * len(max_samples_list)
    * len(max_features_list)
)
current_idx = 0

print(f"\n🔍 Total parameter combinations: {total_combinations}. Starting grid search...\n")

# ----------------------------------------
# 📌 Grid Search loop
# ----------------------------------------
for n_est in n_estimators_list:
    for cont in contamination_list:
        for max_samp in max_samples_list:
            for max_feat in max_features_list:
                current_idx += 1
                print(
                    f"🔄 Progress: {current_idx}/{total_combinations} | "
                    f"n_estimators={n_est}, contamination={cont}, max_samples={max_samp}, max_features={max_feat}"
                )

                try:
                    # Train Isolation Forest on the full dataset
                    model = IsolationForest(
                        n_estimators=n_est,
                        contamination=cont,
                        max_samples=max_samp,
                        max_features=max_feat,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    model.fit(X_scaled)
                    preds = model.predict(X_scaled)
                    is_outlier = (preds == -1).astype(int)
                    anomaly_score = model.decision_function(X_scaled)

                    # Basic metrics
                    anomaly_ratio = is_outlier.mean()
                    avg_score = anomaly_score.mean()
                    std_score = anomaly_score.std()

                    # ⚡ Fast subsample of 10,000 rows for silhouette_score
                    X_sample, preds_sample = resample(
                        X_scaled,
                        preds,
                        n_samples=sample_size,
                        random_state=random_state,
                        stratify=None,
                    )
                    sil_score = silhouette_score(X_sample, preds_sample)

                    results.append(
                        {
                            'n_estimators': n_est,
                            'contamination': cont,
                            'max_samples': max_samp,
                            'max_features': max_feat,
                            'anomaly_ratio': anomaly_ratio,
                            'avg_score': avg_score,
                            'std_score': std_score,
                            'silhouette_score': sil_score,
                        }
                    )

                    print(
                        f"✅ Metrics: anomaly_ratio={anomaly_ratio:.4f}, avg_score={avg_score:.4f}, "
                        f"std_score={std_score:.4f}, silhouette_score={sil_score:.4f}"
                    )

                except Exception as e:
                    print(f"⚠️ Error at combination {current_idx}/{total_combinations}: {e}")
                    continue

# ----------------------------------------
# 📌 Save results to Excel
# ----------------------------------------
results_df = pd.DataFrame(results)
output_path = f'{output_dir}/isoforest_tuning_results.xlsx'
results_df.to_excel(output_path, index=False)

print(f"\n✅ Tuning completed. Results saved to: {output_path}")
print("\n🔹 Top 10 by silhouette_score:")
print(results_df.sort_values(by='silhouette_score', ascending=False).head(10))

"""
🔹 Intermediate reasoning:
This module systematically tunes the contamination (expected anomaly proportion) and other hyperparameters
of the Isolation Forest on the full treasury transactions dataset.

Key outcomes:
- Enables objective, metric-based calibration of anomaly detection thresholds.
- Uses **silhouette_score** on a robust subsample to evaluate cluster separability and anomaly quality.
- Explores multiple combinations of:
    ▸ n_estimators
    ▸ contamination rates
    ▸ max_samples fractions
    ▸ max_features fractions
- Saves all grid search results for further analysis and reporting.

Recommended:
Keep these tuning results (`isoforest_tuning_results.xlsx`) for documentation
and for selecting stable parameters when building final production pipelines.
"""
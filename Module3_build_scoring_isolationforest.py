"""
📦 Module3_build_scoring_isolationforest.py
Author: Ivan Tatarchuk

Description:
- Loads the final cleaned treasury transactions dataset.
- Applies RobustScaler for robust, outlier-resistant normalization.
- Builds an Isolation Forest for anomaly scoring and transaction correctness analysis.
- Outputs:
    • anomaly_score (from decision_function)
    • is_outlier flag (from predict)
- Saves:
    • Normalized dataset for manual inspection
    • Final scoring outputs
    • Top-100 anomalies with all features for investigation
    • All outputs organized under check_files/isoforest for clarity and pipeline order.
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import os

# ----------------------------------------
# 📌 Prepare output directory
# ----------------------------------------
output_dir = 'check_files/isoforest'
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------
# 📌 Load the final cleaned dataset
# ----------------------------------------
def load_final_dataset(path: str):
    """
    Loads the final cleaned dataset with aggregated features for anomaly scoring.

    Displays:
    - Dataset shape
    - Column names for structure verification
    - Sample of the first 5 rows for sanity check
    - Descriptive statistics including detailed percentiles for distribution review
    """
    print(f"✅ Loading dataset: {path}")
    df = pd.read_csv(path, compression='gzip')
    print(f"🔹 Shape: {df.shape}")
    print(f"🔹 Columns: {df.columns.tolist()}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)
    print(df.head(5))

    print("\n📊 Descriptive statistics with detailed percentiles:")
    pd.set_option('display.float_format', '{:,.1f}'.format)
    percentiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    print(df.describe(percentiles=percentiles).transpose())

    return df

# ----------------------------------------
# 📌 Normalization using RobustScaler
# ----------------------------------------
def normalize_features(df_features: pd.DataFrame):
    """
    Applies RobustScaler to the feature set for outlier-resistant normalization.

    Returns:
    - features_scaled: The normalized feature matrix.
    - scaler: The fitted RobustScaler instance for potential inverse transforms or further analysis.
    """
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(df_features)
    print("✅ Normalization using RobustScaler completed.")
    return features_scaled, scaler

# ----------------------------------------
# 📌 Build Isolation Forest
# ----------------------------------------
def build_isolation_forest(X_scaled):
    """
    Trains an Isolation Forest using hyperparameters optimized via prior Grid Search.

    Selected hyperparameters (based on silhouette_score = 0.84 for stability and quality):
    - n_estimators = 150 (number of trees)
    - contamination = 0.02 (expecting ~2% anomalies)
    - max_samples = 0.5 (for speed and stability)
    - max_features = 1.0 (use all features)
    - random_state = 42 (for reproducibility)
    - n_jobs = -1 (utilize all CPU cores)

    Returns:
    - model: The trained IsolationForest model ready for scoring.
    """
    model = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        max_samples=0.5,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)
    print("✅ Isolation Forest model trained with optimized parameters.")
    return model

# ----------------------------------------
# 📌 Main Execution Block
# ----------------------------------------
if __name__ == '__main__':
    # 1️⃣ Load the final cleaned and feature-engineered dataset
    input_path = 'check_files/transactions_dataset_final_cleaned.csv.gz'
    df = load_final_dataset(input_path)

    # 2️⃣ Separate IDs and select features for modeling
    ids = df['ID']
    # We drop 'SUMPAY' (Winsorized) from the model features,
    # using 'SUMPAY_RAW' instead to retain true extreme values
    # important for fraud and anomaly detection.
    df_features = df.drop(columns=['ID', 'SUMPAY'])

    # 3️⃣ Normalize features using RobustScaler
    X_scaled, scaler = normalize_features(df_features)

    # Save the normalized dataset for manual inspection if needed
    df_scaled = pd.DataFrame(X_scaled, columns=df_features.columns)
    df_scaled.insert(0, 'ID', ids)
    df_scaled.to_csv(f'{output_dir}/transactions_dataset_normalized.csv.gz',
                     index=False, compression='gzip', encoding='utf-8-sig')
    print("✅ Normalized dataset saved for manual review.")

    # 4️⃣ Train the Isolation Forest model
    model = build_isolation_forest(X_scaled)

    # 5️⃣ Generate anomaly_score and is_outlier predictions
    # anomaly_score from decision_function: lower = higher anomaly likelihood
    anomaly_score = model.decision_function(X_scaled)

    # predict returns -1 (outlier) or 1 (inlier)
    # we convert to 1 (outlier) and 0 (normal) for clarity
    is_outlier = (model.predict(X_scaled) == -1).astype(int)

    # 🔍 Display cluster distribution
    normal_count = (is_outlier == 0).sum()
    anomaly_count = (is_outlier == 1).sum()
    total_count = len(is_outlier)

    print(f"\n📊 Cluster distribution:")
    print(f"   ▸ Normal (0): {normal_count:,} transactions ({normal_count / total_count:.3%})")
    print(f"   ▸ Anomaly (1): {anomaly_count:,} transactions ({anomaly_count / total_count:.3%})")

    # 6️⃣ Create the final scoring DataFrame
    df_result = pd.DataFrame({
        'ID': ids,
        'anomaly_score': anomaly_score,
        'is_outlier': is_outlier
    })

    # 7️⃣ Display basic stats
    print("\n✅ Final shape:", df_result.shape)
    print(f"🔹 Anomaly proportion in dataset: {df_result['is_outlier'].mean():.4f}")

    # 8️⃣ Save final scoring results
    df_result.to_csv(f'{output_dir}/scoring_output_isoforest.csv.gz',
                     index=False, compression='gzip', encoding='utf-8-sig')
    print(f"✅ Final scoring results saved in {output_dir}")

    # 9️⃣ Save the top 300 anomalies with all features for manual investigation
    df_with_all_features = df.copy()
    df_with_all_features['anomaly_score'] = anomaly_score
    df_with_all_features['is_outlier'] = is_outlier

    top_300_anomalies = df_with_all_features.sort_values(by='anomaly_score', ascending=True).head(300)
    top_300_anomalies.to_excel(f'{output_dir}/top_300_anomalies.xlsx', index=False)
    print("✅ Top-300 anomalies saved for manual review.")

"""
🔹 Intermediate reasoning:
This module transitions from clean aggregated transaction data to actionable anomaly detection scores.
By applying RobustScaler, we ensure stability in the presence of extreme values, preserving useful signals 
for fraud detection.
Using Isolation Forest allows scalable, unsupervised anomaly scoring without manual labeling, automatically flagging
transactions that deviate from normal patterns.

Key outcomes:
- Generates 'anomaly_score' and 'is_outlier' flags for each transaction.
- Saves a normalized dataset for validation and inspection.
- Provides a ranked list of the top-300 anomalies for manual investigation.
- Establishes a fully automated anomaly scoring pipeline ready for integration into downstream fraud detection 
or reporting dashboards.
"""
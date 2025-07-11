"""
📦 Module5_build_scoring_multi_criteria.py
Author: Ivan Tatarchuk

Description:
- Loads the cleaned treasury transactions dataset.
- Loads feature weights from an Excel file.
- Performs robust normalization (robust z-score) across features.
- Calculates anomaly scoring using a multi-criteria weighted model.
- Displays quantile distribution of the scores.
- Shows the Top-10 transactions with detailed feature contributions.
- Saves:
    • Normalized dataset for review.
    • Dataset with scores and binary anomaly labels.
    • Top-300 anomalies with full feature context.
    • All outputs organized under check_files/multi_criteria for clarity.
"""

import pandas as pd
import os

# ----------------------------------------
# 📌 Paths
# ----------------------------------------
input_data_path = 'check_files/transactions_dataset_final_cleaned.csv.gz'
weights_path = 'check_files/multi_criteria/name_weights.xlsx'
output_dir = 'check_files/multi_criteria'
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------
# 📌 Load data and weights
# ----------------------------------------
print("✅ Loading transaction dataset...")
df = pd.read_csv(input_data_path, compression='gzip')
ids = df['ID']

print("✅ Loading feature weights...")
weights_df = pd.read_excel(weights_path)
weights_dict = dict(zip(weights_df['Field_in_data'], weights_df['Weights']))

# ----------------------------------------
# 📌 Feature selection
# ----------------------------------------
features = list(weights_dict.keys())
df_features = df[features].copy()

# ----------------------------------------
# 📌 Robust normalization
# ----------------------------------------
print("✅ Performing robust normalization using robust z-score scaling...")
medians = df_features.median()
iqr = df_features.quantile(0.75) - df_features.quantile(0.25)
df_normalized = (df_features - medians) / iqr.replace(0, 1)
# Robust normalization is applied specifically for non-normal distributions and outlier presence,
# ensuring appropriate scaling without distortion from outliers.

# ----------------------------------------
# 📌 Scoring calculation using feature weights
# ----------------------------------------
print("✅ Calculating weighted multi-criteria scores...")

# Apply weights to each feature
for col in df_normalized.columns:
    df_normalized[col] *= weights_dict[col]

# Use abs() to capture deviations in both directions (high or low = bad),
# ensuring the score remains positive and interpretable:
# higher = more suspicious.
df_normalized['multi_criteria_score'] = df_normalized.abs().sum(axis=1)

# Move 'multi_criteria_score' to the first column for easier inspection
cols = df_normalized.columns.tolist()
cols.insert(0, cols.pop(cols.index('multi_criteria_score')))
df_normalized = df_normalized[cols]

# ----------------------------------------
# 📌 Quantile distribution of scores
# ----------------------------------------
quantiles = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]
print("\n📊 Quantile distribution of multi_criteria_score:")
quantile_values = df_normalized['multi_criteria_score'].quantile(quantiles)
print(quantile_values.apply(lambda x: f"{x:,.1f}"))

# ----------------------------------------
# 📌 Top-20 transactions with detailed feature contributions
# ----------------------------------------
print("\n📌 Top-20 transactions by score with feature breakdown:")
top_20 = df_normalized.sort_values(by='multi_criteria_score', ascending=False).head(20)
top_20['multi_criteria_score'] = top_20['multi_criteria_score'].apply(lambda x: f"{x:.1f}")
print(top_20.to_string(index=False, max_cols=None))

# ----------------------------------------
# 📌 Save normalized dataset
# ----------------------------------------
df_normalized.insert(0, 'ID', ids)
print("\n💾 Saving normalized dataset to disk. This may take 20-30 seconds...")
df_normalized.to_csv(
    f'{output_dir}/transactions_dataset_normalized.csv.gz',
    index=False,
    compression='gzip',
    encoding='utf-8-sig'
)
print(f"\n✅ Normalized dataset saved in {output_dir}")

# ----------------------------------------
# 📌 Final scoring and binary labeling
# ----------------------------------------
threshold = df_normalized['multi_criteria_score'].quantile(0.98)
df_scoring = df_normalized[['ID', 'multi_criteria_score']].copy()
df_scoring['is_outlier'] = (df_scoring['multi_criteria_score'] >= threshold).astype(int)

df_scoring.to_csv(
    f'{output_dir}/scoring_output_multi_criteria.csv.gz',
    index=False,
    compression='gzip',
    encoding='utf-8-sig'
)
print(f"✅ Final scoring outputs saved in {output_dir}")

# ----------------------------------------
# 📌 Save Top-300 anomalies for manual analysis
# ----------------------------------------
top_300 = df_normalized.sort_values(by='multi_criteria_score', ascending=False).head(300)
top_300.to_excel(
    f'{output_dir}/top_300_anomalies_multi_criteria.xlsx',
    index=False
)
print("✅ Top-300 anomalies saved for manual inspection.")

print("\n✅ Module completed. Model ready for comparison with IsoForest scoring.")

# ----------------------------------------
# 📌 Display top-3 anomalies per feature where feature contribution > SUMPAY_RAW
# ----------------------------------------
print("\n📌 Top-3 anomalies per feature (where contribution exceeds SUMPAY_RAW):")

# Select relevant features excluding SUMPAY_RAW and multi_criteria_score
columns_of_interest = [
    col for col in df_normalized.columns
    if col not in ['ID', 'SUMPAY_RAW', 'multi_criteria_score']
]

anom_df = df_normalized[df_normalized['multi_criteria_score'] >= threshold].copy()
results = []

for col in columns_of_interest:
    # Identify where abs(col) > abs(SUMPAY_RAW)
    mask = anom_df[col].abs() > anom_df['SUMPAY_RAW'].abs()
    filtered = anom_df.loc[mask, ['ID', 'multi_criteria_score', col, 'SUMPAY_RAW']]
    filtered = filtered.rename(columns={col: 'feature_value', 'SUMPAY_RAW': 'SUMPAY_RAW_value'})
    filtered['feature'] = col
    filtered = filtered.reindex(columns=['ID', 'multi_criteria_score', 'feature', 'feature_value', 'SUMPAY_RAW_value'])
    filtered = filtered.iloc[(filtered['feature_value'].abs().argsort())[::-1]].head(3)
    if not filtered.empty:
        results.append(filtered)

if results:
    display_df = pd.concat(results, ignore_index=True)
    pd.set_option('display.float_format', '{:,.2f}'.format)
    print(display_df.to_string(index=False))

    # 💾 Save for client reporting
    display_df.to_excel(
        f'{output_dir}/top_features_contribution_anomalies.xlsx',
        index=False
    )
    print(f"\n✅ Saved to {output_dir}/top_features_contribution_anomalies.xlsx for review.")
else:
    print("ℹ️ No transactions found where other features contributed more than SUMPAY_RAW.")

# ----------------------------------------
# ℹ️ Approach justification:
# Initially, Isolation Forest was used for anomaly detection on large-scale transactional data,
# achieving high silhouette scores (~0.84) but with low interpretability regarding feature contributions.
#
# ⚖️ To improve interpretability, a manual multi-criteria model was added:
# - Uses robust deviation calculations across each feature, summed with weights.
# - Produces a transparent score showing which features contribute most to suspicion.
# - Allows rapid Top-N review, sensitivity adjustment, and clear explanations for stakeholders.
#
# ❌ Min-max scaling was rejected due to instability on real-world data with outliers.
# ✅ Robust normalization is used for stability under extreme values.
#
# Feature weights are integrated (currently set to 1) for future flexibility without code changes,
# allowing Excel-driven weight adjustments tailored to project or client needs.
#
# This multi-criteria model offers a fast, interpretable scoring alternative
# for identifying suspicious transactions with clear, explainable logic.

"""
🔹 Intermediate reasoning:
This module implements a **fast, transparent multi-criteria scoring pipeline** on treasury transactions:

- Applies **robust normalization (robust z-score)** to handle outliers reliably.
- Uses a **weighted sum of absolute deviations**, allowing each feature to contribute proportionally 
to the suspicion score.
- Produces a **fully interpretable scoring**:
    • Higher scores indicate higher suspicion.
    • Clear feature contributions are preserved for manual inspection.
    • Top-300 transactions and per-feature anomaly analysis are automatically saved.
- Enables **stakeholder-friendly reporting** without requiring black-box methods.
- Lays the groundwork for comparing explainable multi-criteria scoring with **IsoForest scoring** for optimal 
pipeline integration.
"""
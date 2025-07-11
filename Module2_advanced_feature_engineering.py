"""
📦 Module2_advanced_feature_engineering.py
Author: Ivan Tatarchuk

Description:
- Loads the cleaned dataset prepared by Module1.
- Displays shape and column list for quick inspection.
- Shows the number of unique values for each feature (excluding ID, SUMPAY).
- Prepares the dataset for advanced feature engineering prior to normalization.
"""

import pandas as pd

# ----------------------------------------
# 📌 Function: Load cleaned dataset
# ----------------------------------------
def load_cleaned_dataset(path: str):
    """
    Loads the cleaned .csv.gz dataset prepared by the previous module.

    Displays:
    - Dataset shape (rows, columns)
    - Column names for quick structure inspection.
    """
    print(f"✅ Loading dataset: {path}")
    df = pd.read_csv(path, compression='gzip')
    print(f"🔹 Shape: {df.shape}")
    print(f"🔹 Columns: {df.columns.tolist()}")
    return df

# ----------------------------------------
# 📌 Function: Display unique value counts for each column
# ----------------------------------------
def show_unique_counts(df: pd.DataFrame):
    """
    Prints the count of unique values for each feature in the dataset,
    excluding 'ID' and 'SUMPAY' which are not informative for this check.

    Helps:
    ✅ Identify columns with low/high cardinality.
    ✅ Spot potential categorical or constant columns for feature engineering.
    """
    cols_to_check = [col for col in df.columns if col not in ['ID', 'SUMPAY']]
    print("\n📊 Unique value counts per column:")
    for col in cols_to_check:
        print(f"{col}: {df[col].nunique(dropna=False)}")
    print("\n✅ Completed. Ready for advanced feature engineering.")

# ----------------------------------------
# 📌 Function: Generate aggregated features
# ----------------------------------------
def generate_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    📌 Generates advanced aggregated numeric features from categorical columns:

    Uses:
    - EMP, REASONID, ORGINN
    - FILINPUT, EXPENSESUM_TYPE_COUNT, PURPOSE_CODE_5DIGIT, HAS_DUPLICATE_ID
    - DIFF_OPERDAY_DOCDATE (via IS_LATE flag)

    Before aggregation, applies Winsorization on SUMPAY (>99%) to stabilize the aggregation statistics.
    After feature generation, drops the original categorical columns.

    Returns:
    - Updated DataFrame with advanced aggregated features for modeling and anomaly detection.
    """

    # ----------------------------------------
    # 📌 Winsorization on SUMPAY (capping at 99th percentile)
    # ----------------------------------------
    # Explanation:
    # In treasury transaction scoring, rows cannot be removed,
    # but extreme SUMPAY outliers can distort aggregation (mean per EMP/ORGINN),
    # making them unreliable for clustering and anomaly detection.
    # Winsorization (capping values above the 99th percentile)
    # preserves row count while stabilizing statistics.
    df['SUMPAY_RAW'] = df['SUMPAY']
    sumpay_cap = df['SUMPAY'].quantile(0.99)
    df['SUMPAY'] = df['SUMPAY'].clip(upper=sumpay_cap)
    print(f"\n✅ Winsorization applied: upper SUMPAY capped at {sumpay_cap:,.0f}")

    # ----------------------------------------
    # 📌 Create IS_LATE flag based on DIFF_OPERDAY_DOCDATE
    # ----------------------------------------
    df['IS_LATE'] = (df['DIFF_OPERDAY_DOCDATE'] > 10).astype(int)

    # ----------------------------------------
    # 📌 1) EMP aggregations
    # ----------------------------------------
    emp_month = df.groupby(['EMP', 'OPERDAY_MONTH']).agg(
        EMP_transaction_count_month=('ID', 'count'),
        EMP_avg_SUMPAY_month=('SUMPAY', 'mean')
    ).reset_index()

    emp_dayweek = df.groupby(['EMP', 'OPERDAY_WEEKDAY']).agg(
        EMP_transaction_count_dayweek=('ID', 'count'),
        EMP_avg_SUMPAY_dayweek=('SUMPAY', 'mean')
    ).reset_index()

    emp_orginn_month = df.groupby(['EMP', 'ORGINN', 'OPERDAY_MONTH']).size().reset_index(name='EMP_ORGINN_count_month')
    emp_orginn_dayweek = df.groupby(['EMP', 'ORGINN', 'OPERDAY_WEEKDAY']).size().reset_index(name='EMP_ORGINN_count_dayweek')

    # ----------------------------------------
    # 📌 2) REASONID aggregations
    # ----------------------------------------
    reasonid_month = df.groupby(['REASONID', 'OPERDAY_MONTH']).agg(
        REASONID_transaction_count_month=('ID', 'count'),
        REASONID_avg_SUMPAY_month=('SUMPAY', 'mean')
    ).reset_index()

    # ----------------------------------------
    # 📌 3) ORGINN aggregations
    # ----------------------------------------
    orginn_month = df.groupby(['ORGINN', 'OPERDAY_MONTH']).agg(
        ORGINN_transaction_count_month=('ID', 'count'),
        ORGINN_avg_SUMPAY_month=('SUMPAY', 'mean')
    ).reset_index()

    # ----------------------------------------
    # 📌 4) FILINPUT aggregations
    # ----------------------------------------
    filinput_month = df.groupby(['FILINPUT', 'OPERDAY_MONTH']).agg(
        FILINPUT_transaction_count_month=('ID', 'count'),
        FILINPUT_avg_SUMPAY_month=('SUMPAY', 'mean')
    ).reset_index()

    # ----------------------------------------
    # 📌 5) EXPENSESUM_TYPE_COUNT aggregations
    # ----------------------------------------
    exp_type_month = df.groupby(['EXPENSESUM_TYPE_COUNT', 'OPERDAY_MONTH']).agg(
        EXP_TYPE_transaction_count_month=('ID', 'count'),
        EXP_TYPE_avg_SUMPAY_month=('SUMPAY', 'mean')
    ).reset_index()

    # ----------------------------------------
    # 📌 6) PURPOSE_CODE_5DIGIT aggregations
    # ----------------------------------------
    purpose_code_month = df.groupby(['PURPOSE_CODE_5DIGIT', 'OPERDAY_MONTH']).agg(
        PUPROSE_CODE_transaction_count_month=('ID', 'count'),
        PUPROSE_CODE_avg_SUMPAY_month=('SUMPAY', 'mean')
    ).reset_index()

    # ----------------------------------------
    # 📌 7) HAS_DUPLICATE_ID aggregations
    # ----------------------------------------
    duplicate_count_emp = df[df['HAS_DUPLICATE_ID'] == 1].groupby(['EMP', 'OPERDAY_MONTH']).size().reset_index(
        name='DUPLICATE_COUNT_EMP_month')

    duplicate_count_orginn = df[df['HAS_DUPLICATE_ID'] == 1].groupby(['ORGINN', 'OPERDAY_MONTH']).size().reset_index(
        name='DUPLICATE_COUNT_ORGINN_month')

    # ----------------------------------------
    # 📌 8) IS_LATE aggregations
    # ----------------------------------------
    late_count_emp = df[df['IS_LATE'] == 1].groupby(['EMP', 'OPERDAY_MONTH']).size().reset_index(
        name='LATE_COUNT_EMP_month')

    late_count_orginn = df[df['IS_LATE'] == 1].groupby(['ORGINN', 'OPERDAY_MONTH']).size().reset_index(
        name='LATE_COUNT_ORGINN_month')

    # ----------------------------------------
    # 📌 Merge aggregated features back into the main DataFrame
    # ----------------------------------------
    df = df.merge(emp_month, on=['EMP', 'OPERDAY_MONTH'], how='left')
    df = df.merge(emp_dayweek, on=['EMP', 'OPERDAY_WEEKDAY'], how='left')
    df = df.merge(emp_orginn_month, on=['EMP', 'ORGINN', 'OPERDAY_MONTH'], how='left')
    df = df.merge(emp_orginn_dayweek, on=['EMP', 'ORGINN', 'OPERDAY_WEEKDAY'], how='left')
    df = df.merge(reasonid_month, on=['REASONID', 'OPERDAY_MONTH'], how='left')
    df = df.merge(orginn_month, on=['ORGINN', 'OPERDAY_MONTH'], how='left')
    df = df.merge(filinput_month, on=['FILINPUT', 'OPERDAY_MONTH'], how='left')
    df = df.merge(exp_type_month, on=['EXPENSESUM_TYPE_COUNT', 'OPERDAY_MONTH'], how='left')
    df = df.merge(purpose_code_month, on=['PURPOSE_CODE_5DIGIT', 'OPERDAY_MONTH'], how='left')
    df = df.merge(duplicate_count_emp, on=['EMP', 'OPERDAY_MONTH'], how='left')
    df = df.merge(duplicate_count_orginn, on=['ORGINN', 'OPERDAY_MONTH'], how='left')
    df = df.merge(late_count_emp, on=['EMP', 'OPERDAY_MONTH'], how='left')
    df = df.merge(late_count_orginn, on=['ORGINN', 'OPERDAY_MONTH'], how='left')

    df.fillna(0, inplace=True)

    # ----------------------------------------
    # 📌 Drop original categorical columns post-aggregation
    # ----------------------------------------
    cols_to_drop = [
        'EMP', 'REASONID', 'ORGINN',
        'FILINPUT', 'EXPENSESUM_TYPE_COUNT', 'PURPOSE_CODE_5DIGIT',
        'HAS_DUPLICATE_ID', 'DIFF_OPERDAY_DOCDATE', 'IS_LATE',
        'OPERDAY_MONTH', 'OPERDAY_WEEKDAY'
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    # ----------------------------------------
    # 📌 Display summary of the final DataFrame
    # ----------------------------------------
    print("\n🔍 Top 10 rows preview:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)
    print(df.head(10))

    print(f"\n✅ Final shape: {df.shape}")
    print("\nℹ️ Data types:")
    print(df.dtypes)

    print("\n📊 Descriptive statistics:")
    pd.set_option('display.float_format', '{:,.1f}'.format)
    print(df.describe(include='all').transpose())

    return df

# ----------------------------------------
# 📌 Execution Block
# ----------------------------------------
if __name__ == '__main__':
    input_path = 'check_files/transactions_dataset_pre_cleaned.csv.gz'

    # Load the cleaned dataset from Module 1 output
    df = load_cleaned_dataset(input_path)

    # Display the count of unique values per column
    show_unique_counts(df)

    # Generate advanced aggregated features for modeling
    df = generate_aggregated_features(df)

    # Save the final cleaned and feature-engineered dataset as compressed CSV (gzip)
    output_path = 'check_files/transactions_dataset_final_cleaned.csv.gz'
    df.to_csv(output_path, index=False, encoding='utf-8-sig', compression='gzip')

    print(f"\n✅ Final dataset successfully saved to {output_path}")

"""
🔹 Intermediate reasoning:
In this module, we performed advanced feature engineering by generating aggregation-based features across employee, 
reason, organization, and transaction attributes, using both temporal (month, weekday) and categorical splits.

Key steps:
✅ Applied Winsorization on SUMPAY to handle extreme outliers while preserving row count.
✅ Created aggregated features for EMP, REASONID, ORGINN, FILINPUT, EXPENSESUM_TYPE_COUNT, and PURPOSE_CODE_5DIGIT.
✅ Added late transaction flags and duplicate ID counts as fraud-related signals.
✅ Dropped original categorical columns post-aggregation to reduce dimensionality.

Outcome:
We now have a numerically stable, feature-rich dataset, preserving row-level granularity while enriching it with 
aggregated behavioral patterns, ready for normalization, clustering, and anomaly scoring in the next pipeline stages.
"""
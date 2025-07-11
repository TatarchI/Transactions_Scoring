"""
📦 Module: Module1_parsing_preprocessing.py
Author: Ivan Tatarchuk

Description:
- Loads compressed CSV (.csv.gz) with treasury transactions.
- Displays column formats, nulls, and shape.
- Performs initial hypothesis checks for duplicate or redundant columns.
- Cleans and converts dates, numbers, and parses EXPENSESUM and PURPOSE for feature generation.
- Saves a pre-cleaned dataset for further feature engineering and anomaly detection.

This is the **essential data parsing and EDA module** that prepares the raw treasury transactions into a consistent,
analyzable structure for the downstream scoring models.
"""

import pandas as pd

# ----------------------------------------
# 📌 Function 1: Load and check dataset
# ----------------------------------------
def load_and_check(input_file: str):
    """
    Loads .csv.gz and performs a basic check:
    - Displays column formats.
    - Shows null counts per column.
    - Displays dataset shape.
    - Prints a preview of the first 5 rows for sanity check.
    """

    print("✅ Step 1: Loading .csv.gz")
    df = pd.read_csv(input_file, compression='gzip')

    print(f"\n🔹 Shape: {df.shape}\n")

    print("🔹 Column data types:")
    print(df.dtypes)

    print("\n🔹 Null values per column:")
    print(df.isnull().sum())

    print("\n🔹 First 5 rows for visual inspection:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)
    print(df.head())

    return df

# ----------------------------------------
# 📌 Dropping Non-Informative Columns
# ----------------------------------------

# ⚠️ Reasoning:
# - ACC: A 27-digit identifier also embedded within PURPOSE.
#   It likely carries no ML value as it is purely an ID.
#   Can be restored later if needed for feature engineering.

# - DOCNUMB and DOCID: Document identifiers likely used for database keys,
#   offering no predictive power for modeling, similar to ACC.

# - EMPNAME: Text name of the employee, while we already have EMP (numeric ID).
#   We retain EMP and drop EMPNAME.

# - REASONIDNAME: Text description of reason, while REASONID (numeric) is present.
#   We keep REASONID and drop REASONIDNAME.

# - ORGNAME: Organization name, while ORGINN (organization tax ID) is present.
#   We keep ORGINN and drop ORGNAME.

# These removals help:
# ✅ Reduce dataset dimensionality.
# ✅ Prevent potential data leakage via identifiers.
# ✅ Preserve future reversibility if feature engineering requires these fields later.

# ----------------------------------------
# 📌 Function 2: Date Conversion and Feature Generation
# ----------------------------------------
def convert_dates(df):
    """
    📌 Converts date columns to datetime and calculates the difference in seconds between DATEEXEC and DATEENTER.

    Steps:
    1) Converts DATEENTER and DATEEXEC to datetime while preserving time.
    2) Creates the column SECONDS_DIFF_DATEEXEC_DATEENTER.
    3) Converts all 4 date-related columns to date-only format (YYYY-MM-DD) for consistent downstream comparisons.
    """

    print("\n✅ Converting date columns to datetime")

    # 1️⃣ Preserve datetime with time for DATEENTER and DATEEXEC
    df['DATEENTER'] = pd.to_datetime(df['DATEENTER'], errors='coerce')
    df['DATEEXEC'] = pd.to_datetime(df['DATEEXEC'], errors='coerce')

    # 2️⃣ Calculate the difference in seconds between DATEEXEC and DATEENTER
    df['SECONDS_DIFF_DATEEXEC_DATEENTER'] = (
        (df['DATEEXEC'] - df['DATEENTER']).dt.total_seconds()
    )

    # 3️⃣ Normalize all date columns to date-only (YYYY-MM-DD) while preserving dtype as datetime64[ns]
    # Using .dt.normalize() to strip time while retaining date type
    for col in ['OPERDAY', 'DOCDATE']:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.normalize()

    df['DATEENTER'] = df['DATEENTER'].dt.normalize()
    df['DATEEXEC'] = df['DATEEXEC'].dt.normalize()

    print("✅ Date columns processed and SECONDS_DIFF_DATEEXEC_DATEENTER feature added.")
    return df

# ----------------------------------------
# 📌 Function 3: Numeric Columns Conversion
# ----------------------------------------
def convert_numbers(df):
    """
    📌 Converts numeric columns to float where applicable.

    SUMPAY:
        - Attempts conversion to numeric (float).
        - Values containing invalid characters (e.g., '#') will remain as-is.
        - These invalid values are intended to be fixed later using EXPENSESUM in the pipeline.

    SUMPAYFRM:
        - Removes thousand separators ','.
        - Converts to numeric (float).
    """

    print("\n✅ Converting numeric columns to numeric (float) format")

    # SUMPAY: convert to numeric if possible, retain non-numeric values for later repair
    df['SUMPAY'] = df['SUMPAY'].apply(lambda x: pd.to_numeric(x, errors='ignore'))

    # SUMPAYFRM: remove commas before converting to numeric
    df['SUMPAYFRM'] = df['SUMPAYFRM'].str.replace(',', '', regex=False)
    df['SUMPAYFRM'] = df['SUMPAYFRM'].apply(lambda x: pd.to_numeric(x, errors='ignore'))

    print("✅ Conversion completed.")
    return df

# ----------------------------------------
# 📌 Universal function for auditing non-numeric 'object' values in numeric columns
# ----------------------------------------
def check_non_numeric_objects(df, columns_to_check):
    """
    Checks specified numeric columns for unexpected string (object) values,
    prints the count and lists unique string values for manual audit.
    """
    for col in columns_to_check:
        # Create a mask for rows where the value is still a string
        mask = df[col].apply(lambda x: isinstance(x, str))
        unique_objects = df.loc[mask, col].unique()
        count_objects = mask.sum()

        print(f"\n🔹 Number of rows with non-converted string values in {col}: {count_objects} out of {len(df)}")
        print(f"🔹 Unique string values found in {col}:")
        print(unique_objects)

# ----------------------------------------
# 📌 Function 4: Extracting data from EXPENSESUM and PURPOSE
# ----------------------------------------

def extract_expensesum_and_purpose(df):
    """
    📌 Simplified extraction of structured information from EXPENSESUM and PURPOSE for feature engineering.

    EXPENSESUM:
    - Split by '~' (ignoring the trailing one if present).
    - Count the number of operation types (odd positions) -> EXPENSESUM_TYPE_COUNT.
    - Calculate the total sum across all types (even positions) -> EXPENSESUM_TOTAL_PARSED.

    PURPOSE:
    - Split by '~'.
    - Hypothesis: PURPOSE contains strings in three possible formats:
        1) A 27-digit code (duplicate of the removed ACC), then ORGINN, then text.
        2) A 5-digit code (likely a treasury department code), then as above.
        3) A 5-digit code followed by a 25-digit unique ID, then as above.
      These codes carry unique identifiers for features:
        - 5-digit code: potential department or transaction code,
        - 25-digit code: potential additional transaction classifier.
      Extraction logic:
        - First found 5-digit code -> PURPOSE_CODE_5DIGIT,
        - First found 25-digit code -> PURPOSE_ID_25DIGIT.
      If not found, the value is set to "No".

    After execution, the function prints the dtypes for verification.
    """

    print("\n✅ Processing EXPENSESUM")

    # Remove trailing '~' if it exists
    df['EXPENSESUM'] = df['EXPENSESUM'].str.rstrip('~')

    # Split the string
    expense_split = df['EXPENSESUM'].str.split('~', expand=True)

    # Convert to numeric where possible
    expense_split = expense_split.apply(pd.to_numeric, errors='coerce')

    # Count the number of operation types:
    # Count non-null values in odd-position columns (0-based index)
    type_cols = [col for idx, col in enumerate(expense_split.columns) if idx % 2 == 0]
    df['EXPENSESUM_TYPE_COUNT'] = expense_split[type_cols].notnull().sum(axis=1)

    # Calculate total sums:
    # Sum values in even-position columns (actual sums)
    sum_cols = [col for idx, col in enumerate(expense_split.columns) if idx % 2 == 1]
    df['EXPENSESUM_TOTAL_PARSED'] = expense_split[sum_cols].sum(axis=1)
    # Divide by 100 to convert from pennies to base currency
    df['EXPENSESUM_TOTAL_PARSED'] = df['EXPENSESUM_TOTAL_PARSED'] / 100

    print("\n✅ Processing PURPOSE")

    # Split the string
    purpose_split = df['PURPOSE'].str.split('~', expand=True)

    # Initialize target columns
    df['PURPOSE_CODE_5DIGIT'] = 'No'
    df['PURPOSE_ID_25DIGIT'] = 'No'

    # Iterate through rows to find and extract 5-digit and 25-digit codes
    for idx, row in purpose_split.iterrows():
        found_code = False
        found_id = False
        for item in row.dropna():
            item_str = str(item).strip()
            if not found_code and item_str.isdigit() and len(item_str) == 5:
                df.at[idx, 'PURPOSE_CODE_5DIGIT'] = item_str
                found_code = True
            if not found_id and item_str.isdigit() and len(item_str) == 25:
                df.at[idx, 'PURPOSE_ID_25DIGIT'] = item_str
                found_id = True
            if found_code and found_id:
                break

    print("\n✅ Simplified extraction from EXPENSESUM and PURPOSE completed.")

    # Display data types for verification
    print("\n🔹 Data types after EXPENSESUM and PURPOSE extraction:")
    print(df.dtypes)

    return df

# ----------------------------------------
# 📌 Function 5: Column hypothesis validation
# ----------------------------------------

def verify_column_hypotheses(df):
    """
    📌 Validates hypotheses about potential duplicates and column matches:

    1️⃣ Date matching:
        - Base column: OPERDAY
        - Compared with DOCDATE, DATEENTER, DATEEXEC
        - Prints match counts for each and overall.

    2️⃣ Amount matching:
        - Base column: SUMPAY
        - Compared with SUMPAYFRM and EXPENSESUM_TOTAL_PARSED
        - Replaces '#' entries in SUMPAY and SUMPAYFRM
          with EXPENSESUM_TOTAL_PARSED (observed in 4 cases) for cleaning.
        - Prints match counts for each and overall.

    3️⃣ FILIAL vs FILINPUT:
        - Checks only rows where FILIAL != 0
        - Base column: FILINPUT
        - Prints the count of matches.
    """

    # ---------------------- 1️⃣ DATE MATCHING ----------------------
    print("\n✅ Check 1: Date matching (OPERDAY vs others)")

    comparisons = {}
    base_col = 'OPERDAY'
    date_cols = ['DOCDATE', 'DATEENTER', 'DATEEXEC']

    for col in date_cols:
        match_count = (df[base_col] == df[col]).sum()
        comparisons[col] = match_count
        print(f"Matches {base_col} == {col}: {match_count} out of {len(df)}")

    # ---------------------- 2️⃣ AMOUNT MATCHING ----------------------
    print("\n✅ Check 2: Amount matching (SUMPAY vs others)")

    # Identify rows with invalid '#' entries in SUMPAY and SUMPAYFRM
    mask_invalid_sumpay = df['SUMPAY'].astype(str).str.contains('#', na=False)
    mask_invalid_sumpayfrm = df['SUMPAYFRM'].astype(str).str.contains('#', na=False)

    # Replace these with EXPENSESUM_TOTAL_PARSED
    df.loc[mask_invalid_sumpay, 'SUMPAY'] = df.loc[mask_invalid_sumpay, 'EXPENSESUM_TOTAL_PARSED']
    df.loc[mask_invalid_sumpayfrm, 'SUMPAYFRM'] = df.loc[mask_invalid_sumpayfrm, 'EXPENSESUM_TOTAL_PARSED']

    # Ensure numeric types for safe comparison
    df['SUMPAY'] = pd.to_numeric(df['SUMPAY'], errors='coerce')
    df['SUMPAYFRM'] = pd.to_numeric(df['SUMPAYFRM'], errors='coerce')

    # Compare and print results
    comp_sumpay_sumpayfrm = (df['SUMPAY'] == df['SUMPAYFRM']).sum()
    comp_sumpay_expensesum = (df['SUMPAY'] == df['EXPENSESUM_TOTAL_PARSED']).sum()
    print(f"Matches SUMPAY == SUMPAYFRM: {comp_sumpay_sumpayfrm} out of {len(df)}")
    print(f"Matches SUMPAY == EXPENSESUM_TOTAL_PARSED: {comp_sumpay_expensesum} out of {len(df)}")

    # ---------------------- 3️⃣ FILIAL vs FILINPUT ----------------------
    print("\n✅ Check 3: FILIAL vs FILINPUT (only where FILIAL != 0)")

    mask_filial = df['FILIAL'] != 0
    comp_filial = (df.loc[mask_filial, 'FILIAL'] == df.loc[mask_filial, 'FILINPUT']).sum()
    print(f"Matches FILIAL == FILINPUT: {comp_filial} out of {mask_filial.sum()}")

    # ---------------------- Display mismatches ----------------------
    print("\n🔎 3 examples of mismatches OPERDAY vs DOCDATE:")
    print(df.loc[df['OPERDAY'] != df['DOCDATE'], ['ID', 'OPERDAY', 'DOCDATE']].head(3).to_string(index=False))

    print("\n🔎 3 examples of mismatches SUMPAY vs EXPENSESUM_TOTAL_PARSED:")
    print(df.loc[df['SUMPAY'] != df['EXPENSESUM_TOTAL_PARSED'],
                 ['ID', 'SUMPAY', 'EXPENSESUM_TOTAL_PARSED']].head(3).to_string(index=False))

    print("\n🔎 3 examples of mismatches FILIAL vs FILINPUT (FILIAL != 0):")
    mask = df['FILIAL'] != 0
    print(df.loc[mask & (df['FILIAL'] != df['FILINPUT']), ['ID', 'FILIAL', 'FILINPUT']].head(3).to_string(index=False))

    print("\n🎯 Column hypothesis validation completed.")

    return df

# ==========================================================
# 📊 INTERMEDIATE INSIGHTS AFTER INITIAL DATA ANALYSIS
# ==========================================================

# 1️⃣ Date Columns Analysis:
# - The vast majority of values across the 4 date columns match:
#   * OPERDAY vs DOCDATE: 93% match
#   * OPERDAY vs DATEENTER, DATEEXEC: 97% match
# - The typical difference is 1 day, likely due to technical processing delays
#   in the system causing transactions to shift to the next day.
# - Conclusion:
#   ✅ Retain OPERDAY as the primary transaction date.
#   ✅ Remove DOCDATE, DATEENTER, DATEEXEC.
#   ✅ Create a new feature DIFF_OPERDAY_DOCDATE (in days) to later
#      detect unusual or potentially fraudulent transactions.
#   This hypothesis will be further validated using descriptive statistics and outlier analysis.

# 2️⃣ Transaction Amount Analysis:
# - The parameter SUMPAYFRM matches SUMPAY 100% after cleaning,
#   confirming it as a duplicate.
# - The derived feature EXPENSESUM_TOTAL_PARSED matches SUMPAY in 99.88% of cases,
#   confirming the correctness of our parsing and extraction logic from EXPENSESUM.
# - However, anomalies were identified:
#   * Manual checks revealed duplicate transaction IDs with different EXPENSESUM and PURPOSE details.
#   * For scoring and clustering, duplicates by ID must be removed.
# - Conclusion:
#   ✅ Retain SUMPAY as the primary transaction amount variable.
#   ✅ Remove SUMPAYFRM and EXPENSESUM_TOTAL_PARSED after generating an additional feature:
#       - Feature "HAS_DUPLICATE_ID" (boolean), indicating if an ID has duplicates with different details.
#   ✅ When filtering duplicates, retain rows with filled PURPOSE_ID_25DIGIT and PURPOSE_CODE_5DIGIT,
#      as they are prioritized for payment chain analysis and transaction linking.

# 3️⃣ FILIAL vs FILINPUT Analysis:
# - 88% of FILIAL values match FILINPUT.
# - '000' in FILIAL is likely technical noise from database errors or extraction issues.
# - FILINPUT is more consistently and systematically filled.
# - Conclusion:
#   ✅ Retain FILINPUT as the primary branch feature.
#   ✅ Remove FILIAL.

# 4️⃣ Removal of unused helper columns:
# - Remove EXPENSESUM and PURPOSE as they are no longer needed
#   for the main model and analysis after feature generation.

# ----------------------------------------
# 📌 Function 6: Intermediate Cleanup Checks
# ----------------------------------------

def intermediate_cleanup_checks(df):
    """
    📌 Intermediate consistency checks and dataset preparation before final cleanup:

    1️⃣ Check unique values in SECONDS_DIFF_DATEEXEC_DATEENTER to understand time difference distribution.
    2️⃣ Remove duplicate rows based on ID, prioritizing rows where PURPOSE_CODE_5DIGIT != 'No'.
    3️⃣ Check the number of unique values in PURPOSE_CODE_5DIGIT and PURPOSE_ID_25DIGIT for data diversity.
    4️⃣ Drop the column PURPOSE_ID_25DIGIT after feature generation.
    5️⃣ Convert PURPOSE_CODE_5DIGIT to numeric for model compatibility.
    """

    print("\n✅ Performing intermediate cleanup and consistency checks...")

    # 1️⃣ Check unique values in SECONDS_DIFF_DATEEXEC_DATEENTER
    print("\n🔹 Unique values in SECONDS_DIFF_DATEEXEC_DATEENTER for review:")
    print(df['SECONDS_DIFF_DATEEXEC_DATEENTER'].unique())

    # Create HAS_DUPLICATE_ID: 1 if the ID is duplicated elsewhere, else 0
    df['HAS_DUPLICATE_ID'] = df.duplicated(subset=['ID'], keep=False).astype(int)
    print(f"✅ Column HAS_DUPLICATE_ID created. Total duplicates flagged: {df['HAS_DUPLICATE_ID'].sum()}")

    # 2️⃣ Remove duplicates by ID, prioritizing rows with PURPOSE_CODE_5DIGIT filled
    dup_counts = df['ID'].duplicated(keep=False).sum()
    print(f"\n🔹 Total duplicate entries based on ID: {dup_counts}")

    # Add technical column to ensure stable duplicate removal
    df['_dup_rank'] = df.groupby('ID').cumcount()
    to_drop = []

    for idx, group in df[df.duplicated('ID', keep=False)].groupby('ID'):
        has_numeric = group['PURPOSE_CODE_5DIGIT'].apply(lambda x: x != 'No').any()
        if has_numeric:
            # If any row has a valid 5-digit code, drop rows without it
            drop_idxs = group.loc[group['PURPOSE_CODE_5DIGIT'] == 'No'].index
        else:
            # If no row has a valid 5-digit code, keep the first row and drop the rest
            drop_idxs = group.iloc[1:].index
        to_drop.extend(drop_idxs)

    df.drop(index=to_drop, inplace=True)
    df.drop(columns=['_dup_rank'], inplace=True)
    print(f"✅ Removed {len(to_drop)} duplicate rows. New dataframe shape: {df.shape}")

    # 3️⃣ Check uniqueness of PURPOSE_CODE_5DIGIT and PURPOSE_ID_25DIGIT
    print(f"\n🔹 Unique PURPOSE_CODE_5DIGIT values: {df['PURPOSE_CODE_5DIGIT'].nunique()}")
    print(f"🔹 Unique PURPOSE_ID_25DIGIT values: {df['PURPOSE_ID_25DIGIT'].nunique()}")

    # 4️⃣ Drop PURPOSE_ID_25DIGIT after extracting relevant features
    df.drop(columns=['PURPOSE_ID_25DIGIT'], inplace=True)
    print("✅ Dropped column PURPOSE_ID_25DIGIT after feature extraction.")

    # 5️⃣ Convert PURPOSE_CODE_5DIGIT to numeric type, replacing 'No' with 0
    df['PURPOSE_CODE_5DIGIT'] = pd.to_numeric(
        df['PURPOSE_CODE_5DIGIT'].replace('No', '0'),
        errors='coerce'
    )
    print("✅ Converted PURPOSE_CODE_5DIGIT to numeric for modeling.")

    return df

# ----------------------------------------
# 📌 Function 7: Prefinal Cleanup and Feature Generation
# ----------------------------------------

def prefinal_cleanup_and_features(df):
    """
    📌 Prefinal data cleanup and feature engineering:

    - Removes redundant columns no longer needed for modeling.
    - Generates DIFF_OPERDAY_DOCDATE (difference in days between OPERDAY and DOCDATE).
    - Generates OPERDAY_MONTH and OPERDAY_WEEKDAY for seasonal and weekday pattern analysis.
    - Displays control outputs (top 20 rows, dtypes, descriptive statistics) to verify correctness.
    """

    print("\n✅ Starting prefinal cleanup and new feature generation...")

    # ➡️ Generate DIFF_OPERDAY_DOCDATE: difference in days, useful for anomaly detection
    df['DIFF_OPERDAY_DOCDATE'] = (df['OPERDAY'] - df['DOCDATE']).dt.days

    # ➡️ Generate OPERDAY_MONTH and OPERDAY_WEEKDAY for seasonality analysis and weekday patterns
    df['OPERDAY_MONTH'] = df['OPERDAY'].dt.month
    df['OPERDAY_WEEKDAY'] = df['OPERDAY'].dt.weekday + 1  # 1 = Monday, 7 = Sunday

    # ➡️ Remove redundant columns that are no longer needed post feature generation
    cols_to_drop = [
        'DOCDATE', 'DATEENTER', 'DATEEXEC', 'FILIAL',
        'SUMPAYFRM', 'EXPENSESUM_TOTAL_PARSED', 'EXPENSESUM', 'PURPOSE',
        'SECONDS_DIFF_DATEEXEC_DATEENTER', 'OPERDAY'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"✅ Dropped redundant columns: {cols_to_drop}")

    # 📊 Post-cleanup control outputs to verify data integrity and readiness for modeling
    print("\n🔹 Sample of 20 random rows after prefinal cleanup:")
    print(df.sample(20, random_state=42))

    print("\n🔹 Data types after cleanup:")
    print(df.dtypes)

    print(f"\n🔹 Dataframe shape after cleanup: {df.shape}")

    print("\n🔹 Descriptive statistics for numeric and categorical features:")
    pd.set_option('display.float_format', '{:,.2f}'.format)
    print(df.describe(include='all').transpose())

    return df

# ----------------------------------------
# 📌 Execution Block
# ----------------------------------------
if __name__ == '__main__':
    # Path to the initial raw dataset
    input_path = 'Raw_data/transactions_dataset.csv.gz'

    # 1️⃣ Load and perform initial dataset checks
    df = load_and_check(input_path)

    # 2️⃣ Drop non-informative columns identified during the EDA phase
    columns_to_drop = ['ACC', 'DOCNUMB', 'DOCID', 'EMPNAME', 'REASONIDNAME', 'ORGNAME']
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"\n✅ Dropped non-informative columns: {columns_to_drop}")

    # 3️⃣ Convert date columns and generate initial time difference features
    df = convert_dates(df)

    # 4️⃣ Convert numeric columns, ensuring correct data types for monetary columns
    df = convert_numbers(df)

    # 5️⃣ Post-conversion type check for validation
    print("\n🔹 Data types after date and numeric conversion:")
    print(df.dtypes)
    print("\n🎯 Date and numeric conversions completed successfully.\n")

    # Preview the top 10 rows for structural verification
    print(df.head(10))

    # 6️⃣ Check for non-numeric strings within columns that should be numeric
    columns_to_check = ['SUMPAY', 'SUMPAYFRM']
    check_non_numeric_objects(df, columns_to_check)

    # 7️⃣ Extract structured information from EXPENSESUM and PURPOSE columns for feature engineering
    df = extract_expensesum_and_purpose(df)
    print(df.head(50))

    # 8️⃣ Perform column hypothesis checks to validate duplication and relationships
    df = verify_column_hypotheses(df)

    # 9️⃣ Perform intermediate cleanup, including removal of duplicates while retaining priority rows
    df = intermediate_cleanup_checks(df)

    # 🔟 Perform prefinal cleanup and generate additional engineered features
    df = prefinal_cleanup_and_features(df)

    # ✅ Save the pre-cleaned dataset for validation before final feature selection and modeling
    output_path = 'check_files/transactions_dataset_pre_cleaned.csv.gz'
    df.to_csv(output_path, index=False, encoding='utf-8-sig', compression='gzip')
    print(f"\n✅ Pre-cleaned dataset saved to {output_path} (gzip).")

    # 📌 Observation 1:
    # We detected extreme outliers in SUMPAY (up to ~9.8 quadrillion) while the 75th percentile is only 11.7 million.
    # This indicates potential system errors or fraudulent transactions worth further inspection.

    # 📌 Observation 2:
    # The DIFF_OPERDAY_DOCDATE feature has a maximum of 623 days while the median is 0.
    # Large gaps may indicate delays, system issues, or suspicious transaction patterns requiring attention.

"""
🔹 Intermediate reasoning:
This module is critical in preparing raw treasury transaction data for scalable modeling pipelines.

Key points:
- Standardizes dates and amounts, ensuring consistency for ML/EDA.
- Extracts structured features from semi-structured columns (EXPENSESUM, PURPOSE).
- Removes redundant or leakage-prone columns to reduce noise.
- Adds engineered features (e.g., DIFF_OPERDAY_DOCDATE, HAS_DUPLICATE_ID) to support anomaly detection.
- Flags and prepares outliers for further inspection without losing traceability.

Recommended to keep this module clean and reusable for ETL pipelines across projects.
"""
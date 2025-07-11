"""
📦 Module: Module0_converting_excel.py
Author: Ivan Tatarchuk

Description:
- Converts an Excel file into a compressed CSV (.csv.gz) to accelerate downstream pipelines.
- Loads a specified sheet, removes non-informative columns, and saves a .csv.gz for further efficient processing.

💡 Why:
Converting heavy Excel files to compressed CSVs speeds up I/O operations, saves disk space, and standardizes the data
pipeline format for ingestion into scoring and anomaly detection models.
"""

import pandas as pd
import os

def convert_excel_to_csv_gz(input_excel_path: str,
                            output_csv_gz_path: str,
                            sheet_name: str = "Select accounts",
                            usecols: list = None):
    """
    📌 Converts Excel → .csv.gz

    Parameters:
    - input_excel_path: str — path to the source Excel file
    - output_csv_gz_path: str — path to save the resulting .csv.gz
    - sheet_name: str — sheet name to load (default: 'Select accounts')
    - usecols: list — list of columns to load (default: None, loads all columns)

    ⚡ Optimization:
    Removes non-informative columns that contain a single unique value across 738K+ rows:
    ['FINYEAR', 'STATE', 'STATENAME', 'REASONMSG', 'DOCTYPE', 'SYS', 'ACTION', 'SYSNAME', 'DOCTYPENAME']
    to reduce memory usage and loading time.

    Additionally removes the first column if it is an auto-generated index without analytical value.
    """

    print("✅ Starting Excel → CSV.GZ conversion...")

    # Load Excel file
    print(f"Loading Excel from: {input_excel_path}")
    df = pd.read_excel(input_excel_path,
                       sheet_name=sheet_name,
                       usecols=usecols)

    print(f"Excel loaded! Rows: {len(df)}, Columns: {len(df.columns)}")

    # Drop non-informative columns if they exist
    drop_columns = ['FINYEAR', 'STATE', 'STATENAME', 'REASONMSG',
                    'DOCTYPE', 'SYS', 'ACTION', 'SYSNAME', 'DOCTYPENAME']

    existing_drop_columns = [col for col in drop_columns if col in df.columns]
    if existing_drop_columns:
        df = df.drop(columns=existing_drop_columns)
        print(f"Removed non-informative columns: {existing_drop_columns}")

    # Remove first column if it is a sequential auto-generated index
    first_col_name = df.columns[0]
    from pandas.api.types import is_integer_dtype
    if df[first_col_name].is_monotonic_increasing and is_integer_dtype(df[first_col_name]):
        df = df.drop(columns=[first_col_name])
        print(f"Removed auto-generated index column: {first_col_name}")

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_csv_gz_path), exist_ok=True)

    # Save as .csv.gz
    print(f"Saving to: {output_csv_gz_path}")
    df.to_csv(output_csv_gz_path, index=False, compression='gzip', encoding='utf-8-sig')

    print("🎉 Conversion completed successfully!")

# ----------------------------------------
# 📌 Run block for standalone testing
# ----------------------------------------
if __name__ == '__main__':
    # Input Excel file path
    input_excel = 'Raw_data/Abror aka effectivenes.xlsx'

    # Output compressed CSV path
    output_csv_gz = 'Raw_data/transactions_dataset.csv.gz'

    # Load all columns
    usecols = None

    convert_excel_to_csv_gz(input_excel, output_csv_gz, usecols=usecols)

"""
🔹 Intermediate reasoning:
This step is critical for large-scale pipelines where Excel files can slow down processing.
By standardizing the format into .csv.gz:
- You reduce file size for storage and backups.
- You drastically speed up pandas read/load times in downstream modules.
- You simplify data ingestion for modeling, scoring, and EDA workflows.

Recommended to keep this module in the repo for reusable ETL pipelines.
"""
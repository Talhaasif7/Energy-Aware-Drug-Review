import os
import pandas as pd

def refine_webmd(webmd_dir):
    input_path = os.path.join(webmd_dir, 'webmd.csv')
    
    print("=" * 60)
    print("REFINING WEBMD DATASET")
    print("=" * 60)
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
        
    initial_size = os.path.getsize(input_path) / (1024 * 1024)
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    
    print(f"Current Rows: {initial_rows:,}")
    print(f"Current Size: {initial_size:.2f} MB")
    
    # Strip whitespace from string columns
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
        
    # Remove exact duplicates
    dupes_count = df.duplicated().sum()
    if dupes_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Exact Duplicate Rows Removed: {dupes_count:,}")
        
    # Filter out missing or empty Reviews
    df['Reviews'] = df['Reviews'].replace(['nan', 'NaN', 'None', ''], pd.NA)
    null_reviews = df['Reviews'].isna().sum()
    if null_reviews > 0:
        df = df.dropna(subset=['Reviews']).reset_index(drop=True)
        print(f"Empty/Null Review Rows Removed: {null_reviews:,}")
        
    final_rows = len(df)
    print(f"Final Refined Rows: {final_rows:,}")
    
    df.to_csv(input_path, index=False, encoding='utf-8')
    final_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Refined File Size: {final_size:.2f} MB\n")

def refine_druglib(druglib_dir):
    print("=" * 60)
    print("REFINING DRUGLIB DATASETS (UCI ML DRUG REVIEW)")
    print("=" * 60)
    
    files = [
        ('drugLibTrain_cleaned.csv', 'drugLibTrain_cleaned.tsv'),
        ('drugLibTest_cleaned.csv', 'drugLibTest_cleaned.tsv')
    ]
    
    for csv_file, tsv_file in files:
        csv_path = os.path.join(druglib_dir, csv_file)
        tsv_path = os.path.join(druglib_dir, tsv_file)
        
        target_path = csv_path if os.path.exists(csv_path) else tsv_path
        if not os.path.exists(target_path):
            print(f"File not found: {target_path}")
            continue
            
        df = pd.read_csv(target_path, sep=',' if target_path.endswith('.csv') else '\t')
        print(f"\nProcessing {os.path.basename(target_path)}:")
        print(f"  Rows: {len(df):,}")
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            print("  Removed redundant 'Unnamed: 0' index column.")
            
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'NaN', 'None'], '')
            
        dupes_count = df.duplicated().sum()
        if dupes_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"  Removed {dupes_count:,} duplicate rows.")
            
        print(f"  Final Rows: {len(df):,}")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        df.to_csv(tsv_path, sep='\t', index=False, encoding='utf-8')
        print(f"  Saved cleaned CSV and TSV in {druglib_dir}")

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    webmd_directory = os.path.join(project_root, 'data', '02_secondary_sentiment_scaling', 'external_val_webmd')
    druglib_directory = os.path.join(project_root, 'data', '02_secondary_sentiment_scaling', 'dev_uci_drug_review')
    
    refine_webmd(webmd_directory)
    refine_druglib(druglib_directory)

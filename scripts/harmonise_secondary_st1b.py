"""
ST1b — Secondary Task Label Harmonisation (UCI DrugLib + WebMD)

Fixes applied per mentor review:
  - Define and lock a common label mapping BEFORE seeing results
  - UCI DrugLib 'effectiveness' (5 ordinal levels) → 3-class target
  - WebMD 'Effectiveness' (1–5 integer scale) → same 3-class target
  - Document the mapping rules explicitly
  - Report 10 example units per corpus

Dataset identity note:
  The files in dev_uci_drug_review/ are the UCI Drug Library (DrugLib)
  dataset (~4,108 reviews), NOT the ~215k drugsCom Drug Review dataset.
  These are different UCI ML Repository datasets.
"""
import os
import sys
import pandas as pd


def reconfigure_stdout():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


# ---------------------------------------------------------------
# Harmonisation Mapping (LOCKED before results)
# ---------------------------------------------------------------
# Common target: 3-class ordinal effectiveness
#   0 = Negative (Low effectiveness)
#   1 = Neutral  (Moderate effectiveness)
#   2 = Positive (High effectiveness)
#
# UCI DrugLib 'effectiveness' column (string ordinal):
#   'Ineffective'             → 0 (Negative)
#   'Marginally Effective'    → 0 (Negative)
#   'Moderately Effective'    → 1 (Neutral)
#   'Considerably Effective'  → 2 (Positive)
#   'Highly Effective'        → 2 (Positive)
#
# WebMD 'Effectiveness' column (integer 1–5):
#   1 → 0 (Negative)
#   2 → 0 (Negative)
#   3 → 1 (Neutral)
#   4 → 2 (Positive)
#   5 → 2 (Positive)
# ---------------------------------------------------------------

UCI_EFFECTIVENESS_MAP = {
    'Ineffective': 0,
    'Marginally Effective': 0,
    'Moderately Effective': 1,
    'Considerably Effective': 2,
    'Highly Effective': 2,
}

WEBMD_EFFECTIVENESS_MAP = {
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 2,
}

LABEL_NAMES = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


def harmonise_uci_druglib(train_csv, test_csv, output_csv):
    """Load UCI DrugLib train+test, map effectiveness to 3-class target."""
    print(f"\n--- Processing UCI DrugLib Dataset ---")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"  Raw: {len(df_train)} train + {len(df_test)} test = {len(df)} total")
    print(f"  Columns: {df.columns.tolist()}")

    # Combine review text columns
    text_cols = ['benefitsReview', 'sideEffectsReview', 'commentsReview']
    df['text'] = df[text_cols].fillna('').agg(' '.join, axis=1).str.strip()

    # Map effectiveness
    df['label'] = df['effectiveness'].map(UCI_EFFECTIVENESS_MAP)
    df = df.dropna(subset=['label', 'text'])
    df = df[df['text'].str.len() > 0]
    df['label'] = df['label'].astype(int)

    # Keep metadata columns for subgroup analysis
    out_df = df[['text', 'label', 'urlDrugName', 'condition',
                 'rating', 'sideEffects']].copy()
    out_df.columns = ['text', 'label', 'drug_name', 'condition',
                      'rating_original', 'side_effects']

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"  Saved harmonised UCI DrugLib ({len(out_df)} rows) to: {output_csv}")
    return out_df


def harmonise_webmd(webmd_csv, output_csv):
    """Load WebMD dataset, map Effectiveness to 3-class target."""
    print(f"\n--- Processing WebMD Dataset ---")

    df = pd.read_csv(webmd_csv)
    print(f"  Raw: {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")

    # Use Reviews as text
    df = df.dropna(subset=['Reviews', 'Effectiveness'])
    df['text'] = df['Reviews'].astype(str).str.strip()
    df = df[df['text'].str.len() > 0]

    # Map effectiveness
    df['Effectiveness'] = df['Effectiveness'].astype(int)
    df['label'] = df['Effectiveness'].map(WEBMD_EFFECTIVENESS_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    out_df = df[['text', 'label', 'Drug', 'Condition',
                 'Satisfaction', 'EaseofUse']].copy()
    out_df.columns = ['text', 'label', 'drug_name', 'condition',
                      'satisfaction', 'ease_of_use']

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"  Saved harmonised WebMD ({len(out_df)} rows) to: {output_csv}")
    return out_df


def print_report(df_uci, df_webmd):
    print("\n" + "=" * 80)
    print("      ST1b — SECONDARY TASK LABEL HARMONISATION REPORT")
    print("=" * 80)

    print("\n--- 1. MAPPING RULES (LOCKED BEFORE RESULTS) ---")
    print("  Target: 3-class ordinal effectiveness")
    print("    0 = Negative (Low)  |  1 = Neutral (Moderate)  |  2 = Positive (High)")
    print("\n  UCI DrugLib Mapping:")
    for k, v in UCI_EFFECTIVENESS_MAP.items():
        print(f"    '{k}' → {v} ({LABEL_NAMES[v]})")
    print("\n  WebMD Mapping:")
    for k, v in WEBMD_EFFECTIVENESS_MAP.items():
        print(f"    Effectiveness={k} → {v} ({LABEL_NAMES[v]})")

    print("\n--- 2. COUNT TABLE ---")
    def stats(df, name):
        total = len(df)
        counts = df['label'].value_counts().sort_index()
        return {
            'Dataset': name, 'Total': total,
            'Negative (0)': counts.get(0, 0),
            'Neutral (1)': counts.get(1, 0),
            'Positive (2)': counts.get(2, 0),
            'Positive %': f"{counts.get(2, 0)/total*100:.1f}%"
        }

    print(pd.DataFrame([
        stats(df_uci, 'UCI DrugLib'),
        stats(df_webmd, 'WebMD')
    ]).to_string(index=False))

    print("\n--- 3. UCI DRUGLIB — 10 EXAMPLE UNITS ---")
    for i, row in df_uci.sample(10, random_state=42).reset_index(drop=True).iterrows():
        print(f"  [{i+1:02d}] Label={row['label']} | "
              f"\"{row['text'][:100]}{'...' if len(row['text'])>100 else ''}\"")

    print("\n--- 4. WebMD — 10 EXAMPLE UNITS ---")
    for i, row in df_webmd.sample(10, random_state=42).reset_index(drop=True).iterrows():
        print(f"  [{i+1:02d}] Label={row['label']} | "
              f"\"{row['text'][:100]}{'...' if len(row['text'])>100 else ''}\"")

    print("\n--- 5. DATASET IDENTITY NOTE ---")
    print("  The files drugLibTrain_cleaned.csv / drugLibTest_cleaned.csv")
    print(f"  contain {len(df_uci)} reviews from the UCI Drug Library (DrugLib) dataset.")
    print("  This is NOT the ~215k drugsCom Drug Review dataset.")
    print("  Both are distinct UCI ML Repository datasets.")

    print("\n--- 6. HARMONISATION LOCK CONFIRMATION ---")
    print("  [OK] Label mapping LOCKED before any model results were examined.")
    print("  [OK] Effectiveness chosen as alignment dimension (present in both corpora).")
    print("  [OK] 3-class ordinal target preserves ordering information.")
    print("=" * 80 + "\n")


def main():
    reconfigure_stdout()
    base = r"e:\AI Green\data\02_secondary_sentiment_scaling"
    uci_train = os.path.join(base, "dev_uci_drug_review", "drugLibTrain_cleaned.csv")
    uci_test = os.path.join(base, "dev_uci_drug_review", "drugLibTest_cleaned.csv")
    uci_out = os.path.join(base, "dev_uci_drug_review", "uci_druglib_harmonised.csv")
    webmd_csv = os.path.join(base, "external_val_webmd", "webmd.csv")
    webmd_out = os.path.join(base, "external_val_webmd", "webmd_harmonised.csv")

    df_uci = harmonise_uci_druglib(uci_train, uci_test, uci_out)
    df_webmd = harmonise_webmd(webmd_csv, webmd_out)
    print_report(df_uci, df_webmd)


if __name__ == "__main__":
    main()

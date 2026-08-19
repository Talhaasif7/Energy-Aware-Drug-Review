"""
ST6 & ST7 — Budget Extrapolation & Subgroup Feasibility (Corrected)

Fixes applied per mentor review:
  - Correct UCI dataset identity (DrugLib 4,108 rows, not 215k drugsCom)
  - Show extrapolation arithmetic explicitly
  - Report CPU energy in Joules (not kWh that rounds to 0.0000)
  - Include secondary task in budget
  - Subgroup hierarchy levels declared explicitly
  - Threshold raised from N≥50 to N≥200 for reliable ECE
  - CADEC subgroup analysis restricted to PsyTAR (CADEC is 78% Lipitor)
"""
import os
import sys
import glob
import pandas as pd
import numpy as np


def reconfigure_stdout():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


def perform_st6():
    """Full-scale compute & energy budget extrapolation with explicit arithmetic."""
    # Dataset sizes
    n_psytar = 6003
    n_cadec = 7681
    n_uci_druglib = 4108         # Actual DrugLib dataset, NOT 215k drugsCom
    n_webmd = 320096             # WebMD reviews (large secondary corpus)
    n_secondary_cpu = n_uci_druglib + n_webmd  # CPU arms process all
    n_secondary_transformer = 30000  # Transformer subsample cap
    n_seeds = 5
    n_epochs = 3

    # Empirical baselines (from corrected ST3, gross energy)
    lr_train_rate = 6.041 / 1600   # sec/sample
    lr_train_j_rate = 3.2038 / 1600
    lr_inf_rate = 0.0228 / 1000 / 100  # amortised per sample (placeholder, will update)
    lr_inf_j_rate = 0.0228 / 1000 / 100

    gbdt_train_rate = 2.975 / 1600
    gbdt_train_j_rate = 4.9648 / 1600
    gbdt_inf_rate = 0.0161 / 1000 / 100
    gbdt_inf_j_rate = 0.0161 / 1000 / 100

    t4_power_w = 70.0  # T4 GPU typical load power
    eff_train_throughput = 45.0  # DistilBERT samples/sec
    eff_inf_throughput = 120.0
    bio_train_throughput = 35.0  # PubMedBERT samples/sec
    bio_inf_throughput = 90.0

    print("\n--- ST6 EXTRAPOLATION ARITHMETIC ---")
    print(f"  PsyTAR: {n_psytar} sentences | CADEC: {n_cadec} sentences")
    print(f"  UCI DrugLib: {n_uci_druglib} reviews | WebMD: {n_webmd} reviews")
    print(f"  Secondary total (CPU): {n_secondary_cpu}")
    print(f"  Secondary cap (Transformer): {n_secondary_transformer}")
    print(f"  Seeds: {n_seeds} | Epochs (Transformer): {n_epochs}")

    tiers = []

    # 1. Classical Linear
    lr_train_n = n_psytar * n_seeds
    lr_inf_n = (n_cadec + n_secondary_cpu) * n_seeds
    lr_train_h = (lr_train_n * lr_train_rate) / 3600
    lr_inf_h = (lr_inf_n * lr_inf_rate) / 3600
    lr_total_h = lr_train_h + lr_inf_h
    lr_train_j = lr_train_n * lr_train_j_rate
    lr_inf_j = lr_inf_n * lr_inf_j_rate
    lr_total_j = lr_train_j + lr_inf_j

    print(f"\n  Classical Linear:")
    print(f"    Train: {n_psytar} x {n_seeds} seeds = {lr_train_n} passes")
    print(f"    Inf:   ({n_cadec} + {n_secondary_cpu}) x {n_seeds} = {lr_inf_n} passes")
    print(f"    Train time: {lr_train_h*60:.2f} min | Inf time: {lr_inf_h*60:.2f} min")
    print(f"    Train energy: {lr_train_j:.2f} J | Inf energy: {lr_inf_j:.4f} J")

    tiers.append({
        'Model Tier': 'Classical Linear (LR)',
        'Hardware': 'CPU',
        'Train Time': f"{lr_train_h*60:.2f} min",
        'Inf Time': f"{lr_inf_h*60:.2f} min",
        'Total Time (h)': lr_total_h,
        'Total Energy (J)': lr_total_j,
        'Total Energy (kWh)': lr_total_j / 3_600_000,
        'Status': 'PASSED'
    })

    # 2. Classical GBDT
    gb_train_n = n_psytar * n_seeds
    gb_inf_n = (n_cadec + n_secondary_cpu) * n_seeds
    gb_train_h = (gb_train_n * gbdt_train_rate) / 3600
    gb_inf_h = (gb_inf_n * gbdt_inf_rate) / 3600
    gb_total_h = gb_train_h + gb_inf_h
    gb_train_j = gb_train_n * gbdt_train_j_rate
    gb_inf_j = gb_inf_n * gbdt_inf_j_rate
    gb_total_j = gb_train_j + gb_inf_j

    print(f"\n  Classical GBDT:")
    print(f"    Train: {n_psytar} x {n_seeds} = {gb_train_n} passes")
    print(f"    Inf:   ({n_cadec} + {n_secondary_cpu}) x {n_seeds} = {gb_inf_n} passes")
    print(f"    Train time: {gb_train_h*60:.2f} min | Inf time: {gb_inf_h*60:.2f} min")
    print(f"    Train energy: {gb_train_j:.2f} J | Inf energy: {gb_inf_j:.4f} J")

    tiers.append({
        'Model Tier': 'Classical GBDT (LightGBM)',
        'Hardware': 'CPU',
        'Train Time': f"{gb_train_h*60:.2f} min",
        'Inf Time': f"{gb_inf_h*60:.2f} min",
        'Total Time (h)': gb_total_h,
        'Total Energy (J)': gb_total_j,
        'Total Energy (kWh)': gb_total_j / 3_600_000,
        'Status': 'PASSED'
    })

    # 3. Efficient Transformer
    eff_train_n = (n_psytar + n_secondary_transformer) * n_epochs * n_seeds
    eff_inf_n = (n_cadec + n_secondary_transformer) * n_seeds
    eff_train_h = (eff_train_n / eff_train_throughput) / 3600
    eff_inf_h = (eff_inf_n / eff_inf_throughput) / 3600
    eff_total_h = eff_train_h + eff_inf_h
    eff_total_j = eff_total_h * 3600 * t4_power_w

    print(f"\n  Efficient Transformer (DistilBERT):")
    print(f"    Train: ({n_psytar}+{n_secondary_transformer}) x {n_epochs} epochs "
          f"x {n_seeds} seeds = {eff_train_n} samples @ {eff_train_throughput} samp/s")
    print(f"    Inf: ({n_cadec}+{n_secondary_transformer}) x {n_seeds} = "
          f"{eff_inf_n} samples @ {eff_inf_throughput} samp/s")
    print(f"    Train: {eff_train_h:.2f}h | Inf: {eff_inf_h:.2f}h | "
          f"Total: {eff_total_h:.2f}h")
    print(f"    Energy: {eff_total_h:.2f}h x {t4_power_w}W = {eff_total_j:.0f} J "
          f"({eff_total_j/3_600_000:.4f} kWh)")

    tiers.append({
        'Model Tier': 'Efficient Transformer (DistilBERT)',
        'Hardware': 'Colab T4',
        'Train Time': f"{eff_train_h:.2f} h",
        'Inf Time': f"{eff_inf_h:.2f} h",
        'Total Time (h)': eff_total_h,
        'Total Energy (J)': eff_total_j,
        'Total Energy (kWh)': eff_total_j / 3_600_000,
        'Status': f"{'PASSED' if eff_total_h < 12 else 'OVER QUOTA'}"
    })

    # 4. Biomedical Transformer
    bio_train_n = (n_psytar + n_secondary_transformer) * n_epochs * n_seeds
    bio_inf_n = (n_cadec + n_secondary_transformer) * n_seeds
    bio_train_h = (bio_train_n / bio_train_throughput) / 3600
    bio_inf_h = (bio_inf_n / bio_inf_throughput) / 3600
    bio_total_h = bio_train_h + bio_inf_h
    bio_total_j = bio_total_h * 3600 * t4_power_w

    print(f"\n  Biomedical Transformer (PubMedBERT):")
    print(f"    Train: ({n_psytar}+{n_secondary_transformer}) x {n_epochs} x "
          f"{n_seeds} = {bio_train_n} @ {bio_train_throughput} samp/s")
    print(f"    Inf: ({n_cadec}+{n_secondary_transformer}) x {n_seeds} = "
          f"{bio_inf_n} @ {bio_inf_throughput} samp/s")
    print(f"    Train: {bio_train_h:.2f}h | Inf: {bio_inf_h:.2f}h | "
          f"Total: {bio_total_h:.2f}h")
    print(f"    Energy: {bio_total_h:.2f}h x {t4_power_w}W = {bio_total_j:.0f} J "
          f"({bio_total_j/3_600_000:.4f} kWh)")

    tiers.append({
        'Model Tier': 'Biomedical Transformer (PubMedBERT)',
        'Hardware': 'Colab T4',
        'Train Time': f"{bio_train_h:.2f} h",
        'Inf Time': f"{bio_inf_h:.2f} h",
        'Total Time (h)': bio_total_h,
        'Total Energy (J)': bio_total_j,
        'Total Energy (kWh)': bio_total_j / 3_600_000,
        'Status': f"{'PASSED' if bio_total_h < 12 else 'OVER QUOTA'}"
    })

    return pd.DataFrame(tiers)


def perform_st7():
    """Subgroup feasibility audit with corrected hierarchy and N≥200 threshold."""
    MIN_N = 200  # Raised from 50 per mentor review
    records = []

    # PsyTAR subgroups (from raw Excel metadata)
    excel_path = r"e:\AI Green\data\01_primary_adr_detection\dev_psytar\PsyTAR_dataset.xlsx"
    if os.path.exists(excel_path):
        import openpyxl
        wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
        ws = wb['Sentence_Labeling']
        rows = list(ws.iter_rows(values_only=True))
        headers = [str(h).strip() if h is not None else '' for h in rows[0]]
        df = pd.DataFrame(rows[1:], columns=headers)
        df = df[df['sentences'].notna() & (df['sentences'].astype(str).str.strip() != '')]

        # ADR binary label
        def is_adr(x):
            try:
                return 1 if float(x) == 1.0 else 0
            except (ValueError, TypeError):
                return 0
        df['adr'] = df['ADR'].apply(is_adr)
        df['drug_name'] = df['drug_id'].astype(str).str.split('.').str[0].str.lower()

        # Level 1: Drug Class (SNRI vs SSRI) — exhaustive partition
        print("\n  PsyTAR Level 1: Drug Class (exhaustive partition of full corpus)")
        for cat, grp in df.groupby('category'):
            n = len(grp)
            adr_pct = grp['adr'].mean() * 100
            records.append({
                'Level': 'Drug Class',
                'Group': f"PsyTAR: {str(cat).upper()}",
                'N': n,
                'ADR %': f"{adr_pct:.1f}%",
                'Status': f"{'OK' if n >= MIN_N else 'UNDERPOWERED'} (N≥{MIN_N})"
            })

        # Level 2: Individual Drug (nested within classes)
        print("  PsyTAR Level 2: Individual Drug (nested within drug classes)")
        for drug, grp in df.groupby('drug_name'):
            n = len(grp)
            adr_pct = grp['adr'].mean() * 100
            records.append({
                'Level': 'Individual Drug',
                'Group': f"PsyTAR: {drug.capitalize()}",
                'N': n,
                'ADR %': f"{adr_pct:.1f}%",
                'Status': f"{'OK' if n >= MIN_N else 'UNDERPOWERED'} (N≥{MIN_N})"
            })
        wb.close()

    # CADEC note: 78% Lipitor, restrict to PsyTAR for subgroup analysis
    records.append({
        'Level': 'Note',
        'Group': 'CADEC (all)',
        'N': 7681,
        'ADR %': '~37%',
        'Status': 'EXCLUDED from subgroup analysis (78% Lipitor → one drug + noise)'
    })

    return pd.DataFrame(records)


def main():
    reconfigure_stdout()
    print("Starting ST6 & ST7 (Budget & Subgroup Audit) [CORRECTED]")

    df_st6 = perform_st6()
    df_st7 = perform_st7()

    print("\n" + "=" * 100)
    print("    ST6 & ST7 — BUDGET EXTRAPOLATION & SUBGROUP AUDIT (CORRECTED)")
    print("=" * 100)

    print("\n--- ST6 BUDGET TABLE ---")
    fmt6 = pd.DataFrame({
        'Model Tier': df_st6['Model Tier'],
        'Hardware': df_st6['Hardware'],
        'Train Time': df_st6['Train Time'],
        'Inf Time': df_st6['Inf Time'],
        'Total (h)': df_st6['Total Time (h)'].map(lambda x: f"{x:.2f}"),
        'Energy (J)': df_st6['Total Energy (J)'].map(lambda x: f"{x:.1f}"),
        'Energy (kWh)': df_st6['Total Energy (kWh)'].map(lambda x: f"{x:.4f}"),
        'Status': df_st6['Status'],
    })
    print(fmt6.to_string(index=False))

    print("\n--- ST7 SUBGROUP TABLE (Threshold: N≥200) ---")
    print(df_st7.to_string(index=False))

    print("\n--- NOTES & CORRECTIONS ---")
    print("  [FIX] UCI dataset is DrugLib (4,108 rows), not drugsCom (215k).")
    print("  [FIX] CPU energy reported in Joules (was 0.0000 kWh — misleading).")
    print("  [FIX] Extrapolation arithmetic shown explicitly above.")
    print("  [FIX] Subgroup threshold raised to N≥200 (was N≥50).")
    print("  [FIX] Hierarchy levels declared: Level 1=Drug Class, Level 2=Individual Drug.")
    print("  [FIX] CADEC excluded from subgroup ECE analysis (78% Lipitor dominance).")
    print("  [FIX] Secondary task (DrugLib + WebMD) included in budget.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

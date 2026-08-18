import os
import glob
import re
import pandas as pd
import openpyxl
import nltk
from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize

def setup_nltk():
    """Ensure NLTK sentence tokenization resources are downloaded safely."""
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"[NLTK Setup Note] punkt download attempt: {e}")
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"[NLTK Setup Note] punkt_tab download attempt: {e}")

def harmonise_psytar(psytar_excel_path, output_csv_path):
    """
    Load PsyTAR Excel file dynamically using fast read-only mode, map sentences to binary target,
    and save harmonised CSV with columns ['text', 'label'].
    """
    print(f"\n--- Processing PsyTAR Dataset ({psytar_excel_path}) ---")
    wb = openpyxl.load_workbook(psytar_excel_path, read_only=True, data_only=True)
    sheet_names = wb.sheetnames
    print(f"Detected Excel Sheet Names: {sheet_names}")

    target_sheet = 'Sentence_Labeling' if 'Sentence_Labeling' in sheet_names else sheet_names[0]
    print(f"Reading sheet: '{target_sheet}'")

    ws = wb[target_sheet]
    row_generator = ws.iter_rows(values_only=True)
    header = next(row_generator)
    header_list = [str(h).strip() if h is not None else '' for h in header]

    print(f"Headers in '{target_sheet}': {header_list}")

    sent_idx = header_list.index('sentences') if 'sentences' in header_list else -1
    adr_idx = header_list.index('ADR') if 'ADR' in header_list else -1
    id_idx = header_list.index('id') if 'id' in header_list else -1

    if sent_idx == -1:
        raise ValueError("Could not find 'sentences' column in PsyTAR Excel file.")

    sentences = []
    labels = []

    for row in row_generator:
        if not row:
            continue
        
        # Skip summary/totals rows where id or sentence is None
        row_id = row[id_idx] if id_idx != -1 and id_idx < len(row) else None
        sent_val = row[sent_idx] if sent_idx < len(row) else None

        if sent_val is None or str(sent_val).strip() == '':
            continue
        if id_idx != -1 and row_id is None:
            continue

        sent_text = str(sent_val).strip()
        adr_val = row[adr_idx] if adr_idx != -1 and adr_idx < len(row) else None

        # Binary label mapping: 1 if ADR == 1, 0 otherwise
        is_adr = 0
        if adr_val is not None:
            try:
                if float(adr_val) == 1.0:
                    is_adr = 1
            except (ValueError, TypeError):
                pass

        sentences.append(sent_text)
        labels.append(is_adr)

    wb.close()

    df_harmonised = pd.DataFrame({
        'text': sentences,
        'label': labels
    })

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_harmonised.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Saved PsyTAR harmonised dataset ({len(df_harmonised)} rows) to: {output_csv_path}")
    return df_harmonised

def find_cadec_ann_path(txt_path):
    """
    Locate the corresponding .ann Brat annotation file for a CADEC .txt post.
    """
    # Option 1: Same directory
    ann_candidate1 = os.path.splitext(txt_path)[0] + '.ann'
    if os.path.exists(ann_candidate1):
        return ann_candidate1

    # Option 2: Replace /text/ or \text\ with /original/ or \original\
    ann_candidate2 = txt_path.replace('\\text\\', '\\original\\').replace('/text/', '/original/')
    ann_candidate2 = os.path.splitext(ann_candidate2)[0] + '.ann'
    if os.path.exists(ann_candidate2):
        return ann_candidate2

    # Option 3: Replace with /v2/annotations/ or /annotations/
    ann_candidate3 = txt_path.replace('\\text\\', '\\v2\\annotations\\').replace('/text/', '/v2/annotations/')
    ann_candidate3 = os.path.splitext(ann_candidate3)[0] + '.ann'
    if os.path.exists(ann_candidate3):
        return ann_candidate3

    # Option 4: Direct lookup in same folder structure replacing text with original
    base_dir = os.path.dirname(os.path.dirname(txt_path))
    filename = os.path.splitext(os.path.basename(txt_path))[0] + '.ann'
    ann_candidate4 = os.path.join(base_dir, 'original', filename)
    if os.path.exists(ann_candidate4):
        return ann_candidate4

    return None

def parse_brat_adr_spans(ann_path):
    """
    Extract all ADR entity spans (start_char, end_char) from Brat .ann file.
    Matches lines starting with 'T' and entity type 'ADR' or 'Adverse_Drug_Reaction'.
    Handles discontinuous spans separated by semicolons.
    """
    adr_spans = []
    if not ann_path or not os.path.exists(ann_path):
        return adr_spans

    with open(ann_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_str = line.strip()
            if not line_str.startswith('T'):
                continue

            parts = line_str.split('\t')
            if len(parts) < 2:
                continue

            entity_info = parts[1].strip()
            tokens = entity_info.split()
            if not tokens:
                continue

            entity_type = tokens[0]
            if entity_type.lower() in ['adr', 'adverse_drug_reaction']:
                # The rest of tokens contain offsets, e.g. "9 19" or "9 15;20 25"
                span_str = " ".join(tokens[1:])
                numbers = [int(n) for n in re.findall(r'\d+', span_str)]
                for i in range(0, len(numbers) - 1, 2):
                    start_off, end_off = numbers[i], numbers[i+1]
                    if start_off < end_off:
                        adr_spans.append((start_off, end_off))

    return adr_spans

def harmonise_cadec(cadec_folder_path, output_csv_path):
    """
    Locate text posts and Brat annotations, extract ADR spans, split into sentences,
    map binary sentence labels, and save harmonised CSV with columns ['text', 'label'].
    """
    print(f"\n--- Processing CADEC Dataset ({cadec_folder_path}) ---")

    txt_files = glob.glob(os.path.join(cadec_folder_path, '**', '*.txt'), recursive=True)
    txt_files.sort()
    print(f"Found {len(txt_files)} text posts in CADEC directory.")

    sentences_records = []
    boundary_cross_count = 0
    missing_ann_count = 0
    total_adr_spans_found = 0

    tokenizer = PunktSentenceTokenizer()

    for txt_path in txt_files:
        ann_path = find_cadec_ann_path(txt_path)
        if not ann_path:
            missing_ann_count += 1
            adr_spans = []
        else:
            adr_spans = parse_brat_adr_spans(ann_path)
            total_adr_spans_found += len(adr_spans)

        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            post_text = f.read()

        if not post_text.strip():
            continue

        # Get sentence spans (start_char, end_char)
        sent_spans = list(tokenizer.span_tokenize(post_text))

        for s_start, s_end in sent_spans:
            sentence_text = post_text[s_start:s_end].strip()
            if not sentence_text:
                continue

            # Check overlap with any ADR span
            has_adr = 0
            for a_start, a_end in adr_spans:
                if max(s_start, a_start) < min(s_end, a_end):
                    has_adr = 1
                    if a_start < s_start or a_end > s_end:
                        boundary_cross_count += 1
                    break

            sentences_records.append({
                'text': sentence_text,
                'label': has_adr
            })

    df_harmonised = pd.DataFrame(sentences_records)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_harmonised.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Saved CADEC harmonised dataset ({len(df_harmonised)} rows) to: {output_csv_path}")

    alignment_notes = []
    if missing_ann_count > 0:
        alignment_notes.append(f"Missing annotation files for {missing_ann_count} posts.")
    if boundary_cross_count > 0:
        alignment_notes.append(
            f"Detected {boundary_cross_count} instances where annotated ADR entity spans crossed sentence boundaries."
        )
    alignment_notes.append(f"Total ADR character spans parsed from Brat annotations: {total_adr_spans_found}.")

    return df_harmonised, alignment_notes

def print_st1_report(df_psytar, df_cadec, cadec_alignment_notes):
    """
    Print ST1 Report:
    1. Count Table (Total units, positive cases, class balance %)
    2. 10 Example Units for PsyTAR
    3. 10 Example Units for CADEC
    4. Sentence boundary alignment & tokenization notes
    """
    print("\n" + "="*80)
    print("                      ST1 HARMONISATION REPORT METRICS")
    print("="*80)

    def compute_metrics(df, name):
        total = len(df)
        pos = int(df['label'].sum())
        neg = total - pos
        pct = (pos / total * 100) if total > 0 else 0.0
        return {
            'Dataset': name,
            'Total Sentences': total,
            'ADR-Present (1)': pos,
            'ADR-Absent (0)': neg,
            'ADR Percentage (%)': f"{pct:.2f}%"
        }

    m_psytar = compute_metrics(df_psytar, 'PsyTAR (Dev)')
    m_cadec = compute_metrics(df_cadec, 'CADEC (External Val)')
    df_metrics = pd.DataFrame([m_psytar, m_cadec])

    print("\n--- 1. COUNT TABLE (CLASS BALANCE & UNITS) ---")
    print(df_metrics.to_string(index=False))

    def print_examples(df, name, n=10):
        print(f"\n--- 2. {name.upper()} 10 EXAMPLE UNITS ---")
        pos_df = df[df['label'] == 1].head(5)
        neg_df = df[df['label'] == 0].head(5)

        sample_df = pd.concat([pos_df, neg_df])
        if len(sample_df) < n:
            sample_df = df.head(n)
        sample_df = sample_df.head(n).reset_index(drop=True)

        for idx, row in sample_df.iterrows():
            print(f"[{idx+1:02d}] Label: {row['label']} | Text: \"{row['text']}\"")

    print_examples(df_psytar, "PsyTAR (Development Corpus)")
    print_examples(df_cadec, "CADEC (External Validation Corpus)")

    print("\n--- 3. SENTENCE BOUNDARY ALIGNMENT & TOKENIZATION AUDIT NOTES ---")
    print("- PsyTAR: Sentences were pre-segmented in Excel sheet 'Sentence_Labeling'. Total valid sentences extracted: ", len(df_psytar))
    print("- CADEC Alignment Notes:")
    for note in cadec_alignment_notes:
        print(f"  * {note}")
    print("- Tokenization Note: NLTK PunktSentenceTokenizer span_tokenize was used to extract precise character offsets for CADEC sentence mapping.")
    print("="*80 + "\n")

def main():
    setup_nltk()

    base_dir = r"e:\AI Green"
    psytar_excel = os.path.join(base_dir, r"data\01_primary_adr_detection\dev_psytar\PsyTAR_dataset.xlsx")
    psytar_out = os.path.join(base_dir, r"data\01_primary_adr_detection\dev_psytar\psytar_harmonised.csv")

    cadec_folder = os.path.join(base_dir, r"data\01_primary_adr_detection\external_val_cadec\cadec")
    cadec_out = os.path.join(base_dir, r"data\01_primary_adr_detection\external_val_cadec\cadec_harmonised.csv")

    # 1. Harmonise PsyTAR
    df_psytar = harmonise_psytar(psytar_excel, psytar_out)

    # 2. Harmonise CADEC
    df_cadec, cadec_alignment_notes = harmonise_cadec(cadec_folder, cadec_out)

    # 3. Print ST1 Report
    print_st1_report(df_psytar, df_cadec, cadec_alignment_notes)

if __name__ == "__main__":
    main()

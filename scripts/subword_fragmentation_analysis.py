"""
Subword Fragmentation Rate Analysis for Clinical ADR Terms.

Empirically tests Insight 1 by quantifying how different tokenizers
(Word-level TF-IDF, DistilBERT, PubMedBERT) tokenize domain-specific medical ADR terms.
"""
import os
import sys
import types

os.environ["DISABLE_TRANSFORMERS_VERSION_CHECK"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Bypass transformers version check bug
dummy_dep = types.ModuleType('transformers.dependency_versions_check')
dummy_dep.dep_version_check = lambda *args, **kwargs: None
dummy_dep.PKGS_TO_CHECK = []
sys.modules['transformers.dependency_versions_check'] = dummy_dep

import pandas as pd
import numpy as np

def reconfigure_stdout():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

def main():
    reconfigure_stdout()
    print("==================================================================================")
    print("      SUBWORD FRAGMENTATION RATE ANALYSIS FOR CLINICAL ADR TERMS")
    print("==================================================================================\n")

    adr_terms = [
        "nausea", "headache", "weight gain", "akathisia", "tachycardia",
        "gastrointestinal", "extrapyramidal", "drowsiness", "insomnia", "fatigue",
        "dizziness", "agitation", "tremor", "diarrhea", "constipation",
        "xerostomia", "restlessness", "myalgia", "paresthesia", "hypertension",
        "arrhythmia", "thrombocytopenia", "hepatotoxicity", "hypotension",
        "somnolence", "hyperhidrosis", "dyspepsia", "anhedonia", "bruxism",
        "galactorrhea", "rhabdomyolysis", "leukopenia", "neutropenia"
    ]

    print(f"Testing {len(adr_terms)} representative clinical ADR terms across tokenizers...\n")

    results = []

    # 1. Word-level / TF-IDF baseline (whitespace/punctuation tokenizer)
    word_tokens_count = []
    for term in adr_terms:
        words = term.split()
        word_tokens_count.append((term, len(words), words))
    
    avg_word_frag = np.mean([cnt / len(words) for term, cnt, words in word_tokens_count])

    results.append({
        'Tokenizer': 'Word-Level (TF-IDF Baseline)',
        'Domain Scope': 'General Vocabulary (Count/TF-IDF)',
        'Total Subwords': sum(cnt for _, cnt, _ in word_tokens_count),
        'Total Words': sum(len(words) for _, _, words in word_tokens_count),
        'Mean Frag Rate (tokens/word)': f"{avg_word_frag:.2f}",
        'Intact Terms (%)': "100.0%"
    })

    # 2. DistilBERT and PubMedBERT tokenizers
    try:
        import transformers.utils.versions
        transformers.utils.versions.require_version = lambda *args, **kwargs: None
        transformers.utils.versions.require_version_core = lambda *args, **kwargs: None

        from transformers import AutoTokenizer
        tokenizers = {
            'DistilBERT (distilbert-base-uncased)': ('General Domain', 'distilbert-base-uncased'),
            'PubMedBERT (BiomedNLP-PubMedBERT)': ('Biomedical Domain', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        }

        term_breakdowns = {}

        for name, (domain, hf_path) in tokenizers.items():
            tok = AutoTokenizer.from_pretrained(hf_path)
            total_subwords = 0
            total_words = 0
            intact_count = 0
            breakdown = []

            for term in adr_terms:
                words = term.split()
                subwords = tok.tokenize(term)
                num_sub = len(subwords)
                num_w = len(words)
                
                total_subwords += num_sub
                total_words += num_w
                if num_sub == num_w:
                    intact_count += 1
                
                breakdown.append({'term': term, 'words': num_w, 'subwords': num_sub, 'tokens': subwords})

            frag_rate = total_subwords / total_words
            intact_pct = (intact_count / len(adr_terms)) * 100.0

            results.append({
                'Tokenizer': name,
                'Domain Scope': domain,
                'Total Subwords': total_subwords,
                'Total Words': total_words,
                'Mean Frag Rate (tokens/word)': f"{frag_rate:.2f}",
                'Intact Terms (%)': f"{intact_pct:.1f}%"
            })
            term_breakdowns[name] = breakdown

    except Exception as e:
        import traceback
        print(f"[NOTE] HuggingFace Transformers tokenizer loading note: {e}")
        traceback.print_exc()

    df_results = pd.DataFrame(results)
    print("--- SUBWORD FRAGMENTATION SUMMARY TABLE ---")
    print(df_results.to_string(index=False))

    if 'term_breakdowns' in locals():
        print("\n--- SAMPLE TOKENS COMPARISON FOR COMPLEX MEDICAL TERMS ---")
        sample_terms = ["extrapyramidal", "gastrointestinal", "rhabdomyolysis", "thrombocytopenia", "galactorrhea"]
        for st in sample_terms:
            print(f"\n  Term: '{st}'")
            for name in term_breakdowns:
                b_item = next(item for item in term_breakdowns[name] if item['term'] == st)
                print(f"    - {name:40s} -> Tokens: {b_item['tokens']} (count={b_item['subwords']})")

    print("\n==================================================================================")
    print("  Subword fragmentation analysis complete.")
    print("==================================================================================\n")

if __name__ == "__main__":
    main()

"""Validate the two returned Linux archives and retain all repeatability results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from revision_analysis import ROOT, RESULTS, REPORTS, PSYTAR, grouped_split, write_json


def verify_case(case):
    assert case['status']=='measured'
    repeats=case['repeats'];assert len(repeats)==7
    wraps=0
    for r in repeats:
        n,t,j=r['sentences'],r['elapsed_s'],r['energy_j']
        assert n>0 and t>=20 and np.isfinite(j) and j>0
        assert np.isclose(r['throughput_sps'],n/t)
        assert np.isclose(r['gross_j_1000'],j*1000/n)
        assert np.isclose(r['average_power_w'],j/t)
        before=np.array(r['counter_before_uj']);after=np.array(r['counter_after_uj'])
        assert len(before)==len(after)==1
        delta=float((after-before).sum())/1e6
        if delta>=0:assert np.isclose(j,delta,atol=1e-8)
        else:
            # The archive omits max_energy_range_uj. Retain, but do not claim
            # independent verification of, the reader's wrap correction.
            assert j>delta;wraps+=1
        assert r['board_trace'] is None
    energies=np.array([r['gross_j_1000'] for r in repeats])
    cv=float(energies.std(ddof=1)/energies.mean()*100)
    assert np.isclose(case['cv_pct'],cv)
    assert bool(case['repeatability_pass'])==bool(cv<=10)
    assert np.isclose(case['gross_j_1000'],np.median(energies))
    assert any(all(np.isclose(case[k],r[rk]) for k,rk in [('gross_j_1000','gross_j_1000'),('throughput_sps','throughput_sps'),('load_w','average_power_w')]) for r in repeats)
    assert case['complete_system_energy_j'] is None and not case['wall_meter_available']
    if 'net_j_1000' in case:
        idle=(case['idle_w_pre']+case['idle_w_post'])/2
        assert np.isclose(case['net_j_1000'],max(0,case['load_w']-idle)/case['throughput_sps']*1000)
    return wraps


def main():
    ps=pd.read_csv(PSYTAR);ca=pd.read_csv(ROOT/'data/01_primary_adr_detection/external_val_cadec/cadec_harmonised.csv')
    split=grouped_split(ps,42);records=[];tokenizers=[];archives=[];wraps=0
    for archive_name,folder in [('revision_linux_results.zip',RESULTS),('revision_linux_pubmed_results.zip',ROOT/'results/revision_pubmed_cpu')]:
        archive=ROOT/archive_name
        with zipfile.ZipFile(archive) as z:assert z.testzip() is None
        archives.append({'path':archive_name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'crc_valid':True})
        manifest_path=folder/'linux_energy/run_manifest.json';manifest=json.loads(manifest_path.read_text())
        host=manifest['environment']['host']
        assert host['rapl_available'] and host['rapl_domain_names']==['package-0']
        assert manifest['environment']['benchmark_device']=='cpu' and not manifest['environment']['GPU_available']
        for item in manifest['completed']:
            stem=item['model'];path=folder/'cpu_encoder_predictions'/f'{stem}_seed42_predictions.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['prediction_sha256']
            with np.load(path,allow_pickle=False) as saved:
                for name,df in zip(('train','calib','test'),split):
                    assert np.array_equal(saved[f'{name}_texts'],df.text.astype(str).to_numpy())
                    assert np.array_equal(saved[f'y_{name}'],df.label.to_numpy())
                    assert np.array_equal(saved[f'{name}_groups'].astype(str),df.review_id.astype(str).to_numpy())
                assert np.array_equal(saved['cadec_texts'],ca.text.astype(str).to_numpy())
                assert np.array_equal(saved['y_cadec'],ca.label.to_numpy())
                for key in saved.files:
                    assert saved[key].dtype.kind!='O'
                    if key.startswith('probs_'):
                        p=saved[key];assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                        assert np.allclose(p.sum(axis=1),1,atol=1e-6)
                gpu_stem='efficient_transformer' if stem.startswith('distilbert') else 'biomedical_transformer' if stem.startswith('pubmedbert') else 'compact_transformer'
                gpu_path=RESULTS/'gpu_predictions'/f'{gpu_stem}_seed42_predictions.npz'
                with np.load(gpu_path,allow_pickle=False) as gpu:
                    maximum=max(float(np.max(np.abs(saved[f'probs_{domain}_uncal'][:,1]-gpu[f'probs_{domain}_uncal'][:,1]))) for domain in ('calib','test','cadec'))
                if stem.endswith('fp32'):assert maximum<.01,(stem,maximum)
                metrics={domain+'_auroc':roc_auc_score(saved['y_'+domain],saved[f'probs_{domain}_uncal'][:,1]) for domain in ('test','cadec')}
            energy_path=folder/'linux_energy'/f'{stem}.json';energy=json.loads(energy_path.read_text());assert len(energy['cells'])==1
            case=energy['cells'][0];wraps+=verify_case(case)
            records.append({'model':stem,'prediction_path':path.relative_to(ROOT).as_posix(),'prediction_sha256':item['prediction_sha256'],'energy_path':energy_path.relative_to(ROOT).as_posix(),**metrics,'maximum_uncalibrated_probability_difference_from_gpu_fp16':maximum,**{k:case[k] for k in ('gross_j_1000','net_j_1000','throughput_sps','cv_pct','repeatability_pass')}})
        tok_path=folder/'linux_energy/tokenization.json'
        if 'revision_pubmed_cpu' in str(folder):
            published=RESULTS/'linux_energy/tokenization_pubmed_followup.json'
            published.write_bytes(tok_path.read_bytes());tok_path=published
        for cell in json.loads(tok_path.read_text())['cells']:
            wraps+=verify_case(cell)
            tokenizers.append({'model':cell['model'],'run':'follow-up' if 'revision_pubmed_cpu' in str(folder) else 'original','source':tok_path.relative_to(ROOT).as_posix(),**{k:cell[k] for k in ('gross_j_1000','throughput_sps','cv_pct','repeatability_pass')}})
    assert len(records)==6 and all(r['repeatability_pass'] for r in records)
    excluded=[{'model':r['model'],'prediction_sha256':r['prediction_sha256']} for r in records if r['model'].startswith('pubmedbert_cpu_')]
    records=[r for r in records if not r['model'].startswith('pubmedbert_cpu_')]
    result={'validation_passed':True,'archives':archives,'cpu_predictions':records,'excluded_extra_cpu_predictions':excluded,'exclusion_reason':'Author scope: original PubMedBERT GPU baseline retained; extra PubMedBERT CPU experiment excluded from the paper and model-selection analysis','tokenizer_measurements':tokenizers,'primary_seed':42,'models':4,'received_models':6,'pipeline_repeats':28,'tokenizer_repeats':28,'raw_repeats_validated':70,'counter_wraps_not_independently_verified':wraps,'measurement_scope':'RAPL package-0; no RAM or whole-system measurement','weight_identity':'Checkpoint files not supplied locally; included FP32 probabilities agree with saved GPU FP16 outputs within .01, which is compatibility evidence, not a cryptographic weight-identity proof'}
    write_json(RESULTS/'linux_artifact_audit.json',result)
    pd.DataFrame(records).to_csv(RESULTS/'cpu_model_summary.csv',index=False)
    pd.DataFrame(tokenizers).to_csv(RESULTS/'cpu_tokenizer_summary.csv',index=False)
    lines=['# Returned Linux results: audit','', 'Both archives pass CRC and all six received prediction hashes match the supplied manifests. Text, label and group arrays match the source partitions and external corpus exactly. All arrays load without pickle. Every pipeline measurement retains seven repeats and passes repeatability; raw joule/count/time identities and coupled-median summaries have been checked. The final paper and selection catalogue include only the four planned DistilBERT/BERT-tiny CPU configurations. The extra PubMedBERT CPU experiment is excluded at the author’s instruction; its raw transfer files are preserved locally. Original PubMedBERT GPU configurations remain included.','', '| Included CPU configuration | Source AUROC | External AUROC | Package J/1,000 | Sentences/s | CV (%) |','|---|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['model']} | {r['test_auroc']:.4f} | {r['cadec_auroc']:.4f} | {r['gross_j_1000']:.3f} | {r['throughput_sps']:.2f} | {r['cv_pct']:.2f} |")
    lines.extend(['','The initial PubMedBERT tokenizer measurement has CV 10.81%, exceeding the configured 10% limit. It is retained and flagged, alongside the separately returned follow-up measurement (CV 0.38%). Tokenizer-only measurement for all three GPU encoder families was already part of the planned protocol. The passing follow-up is used descriptively; the two runs are not silently averaged or substituted in the raw artifacts.','',f'Counter wraps whose max-range value cannot be independently reconstructed from the archive: {wraps}.','',result['weight_identity']+'.','', 'CPU-package and GPU-board measurements are different scopes on separate hosts. They are not summed or described as complete-system energy.'])
    (REPORTS/'LINUX_RESULTS_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Validated six received CPU artifacts and 70 repeats; four planned CPU configurations retained. Extra PubMedBERT CPU results excluded; original GPU model retained.')
    for r in records:print(r['model'],round(r['test_auroc'],6),round(r['cadec_auroc'],6),round(r['gross_j_1000'],3))


if __name__=='__main__':main()

"""Linux CPU-only revision: real RAPL energy for saved FP32/INT8 encoders.

Prerequisites: extract revision_checkpoints.zip at repository root, install
requirements-revision.txt. Run: python scripts/revision_linux.py
No GPU or wall meter is assumed. Output is CPU-package energy, not wall energy.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from rapl_utils import RAPLReader,probe_environment
from revision_hardware import infer_all,measure_case,runtime_metadata,calibrated_arrays,canonical_text_sha256
from revision_colab import parts

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision'


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def idle_power(rapl,seconds):
    before=rapl.read();t0=time.perf_counter();time.sleep(seconds);elapsed=time.perf_counter()-t0
    energy=rapl.delta_j(before,rapl.read())
    if energy is None:raise RuntimeError('RAPL counter became unavailable during idle measurement')
    return energy/elapsed


def save_predictions(path,model,tok,split,cadec):
    cal,test=split[1:];pc=infer_all(model,tok,cal.text.tolist(),'cpu');pt=infer_all(model,tok,test.text.tolist(),'cpu');po=infer_all(model,tok,cadec.text.tolist(),'cpu')
    methods=calibrated_arrays(cal.label.to_numpy(),pc,pt,po);odds=np.log(np.clip(pc,1e-7,1-1e-7)/np.clip(1-pc,1e-7,1))
    payload={'logits_calib':np.column_stack([np.zeros(len(pc)),odds]),'seed':42,'cadec_texts':np.asarray(cadec.text.astype(str).tolist(),dtype=str),'y_cadec':cadec.label.to_numpy()}
    for name,df in zip(['train','calib','test'],split):
        payload[f'{name}_texts']=np.asarray(df.text.astype(str).tolist(),dtype=str);payload[f'y_{name}']=df.label.to_numpy();payload[f'{name}_groups']=np.asarray(df.review_id.astype(str).tolist(),dtype=str)
    for method,arrays in methods.items():
        for domain,p in zip(['calib','test','cadec'],arrays):payload[f'probs_{domain}_{method}']=np.column_stack([1-p,p])
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**payload)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints',type=Path,default=OUT/'checkpoints');ap.add_argument('--threads',type=int,default=None)
    ap.add_argument('--models',nargs='+',choices=['efficient_transformer','biomedical_transformer','compact_transformer'],default=['efficient_transformer','biomedical_transformer','compact_transformer'])
    ap.add_argument('--batch-sizes',nargs='+',type=int,default=[64]);ap.add_argument('--lengths',nargs='+',type=int,default=[128])
    ap.add_argument('--measure-s',type=float,default=20);ap.add_argument('--warmup-s',type=float,default=60);ap.add_argument('--repeats',type=int,default=7)
    ap.add_argument('--idle-s',type=float,default=30);ap.add_argument('--cooldown-s',type=float,default=15);ap.add_argument('--preflight',action='store_true')
    args=ap.parse_args();rapl=RAPLReader()
    info={'host':probe_environment(),'checkpoint_directory':str(args.checkpoints),'wall_meter_available':False}
    print(json.dumps(info,indent=2),flush=True)
    if args.preflight:
        if not rapl.ok:raise SystemExit('Preflight FAILED: '+rapl.reason)
        return
    if not rapl.ok:raise SystemExit('RAPL unavailable: '+rapl.reason+'. No energy estimates will be emitted.')
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    info['GPU_available']=torch.cuda.is_available();info['benchmark_device']='cpu'
    args.threads=args.threads or info['host'].get('physical_cores') or os.cpu_count() or 1
    if min(args.measure_s,args.warmup_s,args.idle_s,args.repeats,args.threads)<=0:raise ValueError('Measurement arguments must be positive')
    torch.set_num_threads(args.threads)
    info['runtime']=runtime_metadata()
    cfg=json.loads((ROOT/'configs/revision_config.json').read_text());cfg.update(energy_measure_s=args.measure_s,energy_warmup_s=args.warmup_s,energy_repeats=args.repeats,energy_cooldown_s=args.cooldown_s)
    ps=pd.read_csv(ROOT/'data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv');ca=pd.read_csv(ROOT/'data/01_primary_adr_detection/external_val_cadec/cadec_harmonised.csv');split=parts(ps,42)
    saved=OUT/'cpu_encoder_predictions';rows=[];tokenizers=[]
    for stem in args.models:
        checkpoint=args.checkpoints/f'{stem}_seed42'
        if not checkpoint.exists():raise FileNotFoundError(f'Checkpoint missing: {checkpoint}. Extract Colab revision_checkpoints.zip at repo root.')
        tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        print(f'[tokenizer] {stem}: measured RAPL package only',flush=True)
        tokenizers.append({'model':stem,**measure_case(None,tok,split[2].text.tolist(),'cpu',64,128,cfg,rapl,operation='tokenization')})
        dump(OUT/'linux_energy/tokenization.json',{'environment':info,'cells':tokenizers,'scope':'CPU package tokenization; separate Linux host; not composed into a measured Colab system'})
        if stem=='biomedical_transformer':continue
        base=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True,attn_implementation='eager').eval()
        provenance=json.loads((checkpoint/'revision_provenance.json').read_text())
        if set(provenance['train_review_ids'])!=set(split[0].review_id):raise ValueError('Checkpoint training reviews differ from evaluation protocol')
        source_bytes=(ROOT/'data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv').read_bytes()
        if 'psytar_canonical_sha256' in provenance:
            if provenance['psytar_canonical_sha256']!=canonical_text_sha256(source_bytes):raise ValueError('Checkpoint source corpus content differs')
        elif provenance['psytar_sha256']!=hashlib.sha256(source_bytes).hexdigest():
            raise ValueError('Older checkpoint has a different byte fingerprint. Use the updated Colab bundle with cross-platform fingerprints.')
        for precision in ['fp32','int8']:
            if precision=='fp32':model=base
            else:
                engines=torch.backends.quantized.supported_engines
                engine='fbgemm' if 'fbgemm' in engines else 'qnnpack' if 'qnnpack' in engines else None
                if engine is None:raise RuntimeError('No supported CPU dynamic INT8 quantization backend')
                torch.backends.quantized.engine=engine
                model=torch.ao.quantization.quantize_dynamic(copy.deepcopy(base),{torch.nn.Linear},dtype=torch.qint8,inplace=False).eval()
            prefix={'efficient_transformer':'distilbert_cpu_','compact_transformer':'bert_tiny_cpu_'}[stem]
            name=prefix+precision
            print(f'[accuracy] {name}: grouped seed 42, no retraining',flush=True)
            save_predictions(saved/f'{name}_seed42_predictions.npz',model,tok,split,ca)
            before=idle_power(rapl,args.idle_s)
            cases=[]
            for batch in args.batch_sizes:
                for length in args.lengths:
                    print(f'[RAPL] {name}: batch={batch}, length={length}, repeats={args.repeats}',flush=True)
                    case=measure_case(model,tok,split[2].text.tolist(),'cpu',batch,length,cfg,rapl)
                    cases.append(case);dump(OUT/f'linux_energy/{name}.json',{'model':name,'precision':precision,'cells':cases,'environment':info,'complete_system_energy_measured':False})
            after=idle_power(rapl,args.idle_s);idle=(before+after)/2
            for case in cases:
                if case['status']=='measured':case['idle_w_pre']=before;case['idle_w_post']=after;case['net_j_1000']=max(0,case['load_w']-idle)/case['throughput_sps']*1000
            dump(OUT/f'linux_energy/{name}.json',{'model':name,'precision':precision,'cells':cases,'environment':info,'complete_system_energy_measured':False,'scope':'CPU package only; includes tokenization and model probability output'})
            rows.append({'model':name,'prediction_sha256':hashlib.sha256((saved/f'{name}_seed42_predictions.npz').read_bytes()).hexdigest()})
            if precision=='int8':del model;gc.collect()
        del base,tok;gc.collect()
    dump(OUT/'linux_energy/run_manifest.json',{'environment':info,'completed':rows,'arguments':vars(args)|{'checkpoints':str(args.checkpoints)},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'hardware_helper_sha256':hashlib.sha256((ROOT/'scripts/revision_hardware.py').read_bytes()).hexdigest(),'complete_system_measurement':'unavailable: no wall meter; CPU-only host'})
    archive=ROOT/'revision_linux_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in [saved,OUT/'linux_energy']:
            for p in folder.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(ROOT))
    print(f'Send back {archive}. No model weights are included.',flush=True)


if __name__=='__main__':main()

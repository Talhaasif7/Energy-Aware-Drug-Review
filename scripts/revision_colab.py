"""Colab T4 revision: grouped splits, compact encoder, checkpointing, energy grid.

Upload revision_colab_bundle.zip, extract under /content/aih_revision, then:
 python scripts/revision_colab.py
 python scripts/revision_colab.py --seeds 1001 1002 --skip-sweep  # resume subset
No repository clone/login is required. Runtime duration is measured, not promised.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from revision_hardware import infer_all,measure_case,runtime_metadata,calibrated_arrays,canonical_text_sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision'
MODELS={'efficient_transformer':'distilbert-base-uncased',
        'biomedical_transformer':'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'compact_transformer':'prajjwal1/bert-tiny'}


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def parts(df,seed):
    tr,rest=next(GroupShuffleSplit(train_size=.6,random_state=seed).split(df,groups=df.review_id))
    ca,te=next(GroupShuffleSplit(train_size=.5,random_state=seed).split(df.iloc[rest],groups=df.iloc[rest].review_id))
    result=[df.iloc[ix] for ix in [tr,rest[ca],rest[te]]]
    group_sets=[set(d.review_id) for d in result]
    assert all(not group_sets[a]&group_sets[b] for a,b in [(0,1),(0,2),(1,2)])
    return result


def train(model,tokenizer,df,device,seed):
    # Same documented schedule: AdamW, 3 epochs, lr 2e-5, batch 64, length 128.
    data=tokenizer(df.text.astype(str).tolist(),padding='max_length',truncation=True,max_length=128,return_tensors='pt')
    labels=torch.tensor(df.label.to_numpy(),dtype=torch.long)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    scaler=torch.amp.GradScaler('cuda')
    generator=torch.Generator().manual_seed(seed)
    model.train()
    for epoch in range(3):
        order=torch.randperm(len(df),generator=generator)
        for start in range(0,len(df),64):
            ix=order[start:start+64];batch={k:v[ix].to(device) for k,v in data.items()};batch['labels']=labels[ix].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16):loss=model(**batch).loss
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        print(f'  epoch {epoch+1}/3 completed',flush=True)
    model.eval()


def save_predictions(path,model,tokenizer,split,cadec,device,seed,stem):
    train_df,cal_df,test_df=split
    # Store calibration probabilities as logits with exactly equivalent odds.
    pc=infer_all(model,tokenizer,cal_df.text.tolist(),device,use_fp16=True)
    pt=infer_all(model,tokenizer,test_df.text.tolist(),device,use_fp16=True)
    po=infer_all(model,tokenizer,cadec.text.tolist(),device,use_fp16=True)
    methods=calibrated_arrays(cal_df.label.to_numpy(),pc,pt,po)
    clipped=np.clip(pc,1e-7,1-1e-7);odds=np.log(clipped/(1-clipped))
    payload={'logits_calib':np.column_stack([np.zeros(len(pc)),odds]),'seed':seed,
        'model_name':stem,'calib_probs_uncal':pc}
    for name,df in zip(['train','calib','test'],split):
        payload[f'{name}_texts']=np.asarray(df.text.astype(str).tolist(),dtype=str);payload[f'y_{name}']=df.label.to_numpy();payload[f'{name}_groups']=np.asarray(df.review_id.astype(str).tolist(),dtype=str)
    payload['cadec_texts']=np.asarray(cadec.text.astype(str).tolist(),dtype=str);payload['y_cadec']=cadec.label.to_numpy()
    for method,arrs in methods.items():
        for domain,p in zip(['calib','test','cadec'],arrs):payload[f'probs_{domain}_{method}']=np.column_stack([1-p,p])
    np.savez_compressed(path,**payload)


def make_archive(path,folders):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in folders:
            if folder.is_file():
                z.write(folder,folder.relative_to(ROOT));continue
            for p in sorted(folder.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(ROOT))


def main():
    global torch
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seeds',nargs='+',type=int);ap.add_argument('--models',nargs='+',choices=list(MODELS));ap.add_argument('--skip-sweep',action='store_true');ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    if not torch.cuda.is_available():raise SystemExit('A CUDA GPU runtime is required; choose Colab T4.')
    cfg=json.loads((ROOT/'configs/revision_config.json').read_text());seeds=args.seeds or cfg['existing_split_seeds']+cfg['additional_split_seeds']
    ps=pd.read_csv(ROOT/'data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv')
    ca=pd.read_csv(ROOT/'data/01_primary_adr_detection/external_val_cadec/cadec_harmonised.csv')
    pred=OUT/'gpu_predictions';checks=OUT/'checkpoints';energy=OUT/'gpu_energy';pred.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0');manifest={'environment':runtime_metadata(),'seeds_requested':seeds,'models_requested':args.models or list(MODELS),'scope':'GPU board only; host CPU and wall energy unavailable','completed':[]}
    for stem in args.models or MODELS:
        for seed in seeds:
            path=pred/f'{stem}_seed{seed}_predictions.npz';checkpoint=checks/f'{stem}_seed42'
            sweep_path=energy/f'{stem}_surface.json'
            if path.exists() and not args.overwrite and (seed!=42 or (checkpoint.exists() and (args.skip_sweep or sweep_path.exists()))):
                print(f'[resume] {path.name}',flush=True);continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            print(f'[train] {stem}, grouped seed {seed}',flush=True);t0=time.perf_counter();split=parts(ps,seed)
            tok=AutoTokenizer.from_pretrained(MODELS[stem]);model=AutoModelForSequenceClassification.from_pretrained(MODELS[stem],num_labels=2,attn_implementation='eager').to(device)
            train(model,tok,split[0],device,seed);save_predictions(path,model,tok,split,ca,device,seed,stem)
            if seed==42:
                checkpoint.mkdir(parents=True,exist_ok=True);model.save_pretrained(checkpoint,safe_serialization=True);tok.save_pretrained(checkpoint)
                dump(checkpoint/'revision_provenance.json',{'seed':42,'model':MODELS[stem],'hf_commit':getattr(model.config,'_commit_hash',None),'max_length_training':128,'batch_training':64,'epochs':3,'lr':2e-5,'psytar_sha256':hashlib.sha256((ROOT/'data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv').read_bytes()).hexdigest(),'psytar_canonical_sha256':canonical_text_sha256((ROOT/'data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv').read_bytes()),'train_review_ids':sorted(split[0].review_id.unique().tolist()),'environment':runtime_metadata()})
                if not args.skip_sweep:
                    rows=[]
                    for batch in cfg['batch_sizes']:
                        for length in cfg['sequence_lengths']:
                            print(f'[board energy] {stem}: batch={batch}, length={length}',flush=True)
                            rows.append(measure_case(model,tok,split[2].text.tolist(),device,batch,length,cfg))
                            dump(sweep_path,{'model':stem,'cells':rows,'planned_cells':len(cfg['batch_sizes'])*len(cfg['sequence_lengths']),'environment':runtime_metadata(),
                                'host_energy_measured':False,'complete_system_energy_measured':False,'lengths_are_inference_sensitivity':'training max length remains 128; longer inputs may change predictions'})
            manifest['completed'].append({'model':stem,'seed':seed,'seconds':time.perf_counter()-t0,'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest()});dump(OUT/'colab_run_manifest.json',manifest)
            del model,tok;gc.collect();torch.cuda.empty_cache()
    make_archive(ROOT/'revision_predictions.zip',[pred,energy,OUT/'colab_run_manifest.json'])
    make_archive(ROOT/'revision_checkpoints.zip',[checks])
    # Lightweight analysis output plus large trained checkpoints are separate.
    print('Download revision_predictions.zip, revision_checkpoints.zip and results/revision/colab_run_manifest.json.',flush=True)
    try:
        from google.colab import files
        files.download(str(ROOT/'revision_predictions.zip'));files.download(str(ROOT/'revision_checkpoints.zip'))
    except ImportError:pass


if __name__=='__main__':main()

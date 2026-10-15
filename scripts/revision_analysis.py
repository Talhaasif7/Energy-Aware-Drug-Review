"""Round-1 analysis from saved predictions; never overwrites submission artifacts.

Run: python scripts/revision_analysis.py
New GPU artifacts may be supplied with --predictions results/revision/gpu_predictions.
Primary selection is source-only. CADEC-assisted gating is sensitivity analysis.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb

from metrics_utils import compute_ece_adaptive, compute_clinical_utility_metrics, compute_decision_curve_analysis, TemperatureScaler
from harmonise_st1 import harmonise_cadec
from revision_hardware import canonical_text_sha256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'revision'
REPORTS = ROOT / 'reports' / 'revision'
PSYTAR = ROOT / 'data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv'
CADEC = ROOT / 'data/01_primary_adr_detection/external_val_cadec/cadec_harmonised.csv'
METHODS = [('Uncalibrated', 'uncal'), ('TempScale', 'temp'), ('Isotonic', 'iso')]
MODEL_FILES = {'DistilBERT': 'efficient_transformer', 'PubMedBERT': 'biomedical_transformer', 'BERT-tiny': 'compact_transformer'}
MODEL_FILES.update({'DistilBERT CPU FP32':'distilbert_cpu_fp32','DistilBERT CPU INT8':'distilbert_cpu_int8',
                    'BERT-tiny CPU FP32':'bert_tiny_cpu_fp32','BERT-tiny CPU INT8':'bert_tiny_cpu_int8'})


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped_split(df, seed):
    tr, rest = next(GroupShuffleSplit(train_size=.6, random_state=seed).split(df, groups=df.review_id))
    ca, te = next(GroupShuffleSplit(train_size=.5, random_state=seed).split(df.iloc[rest], groups=df.iloc[rest].review_id))
    out = [df.iloc[i].copy() for i in [tr, rest[ca], rest[te]]]
    sets = [set(d.review_id) for d in out]
    assert all(not sets[a] & sets[b] for a,b in itertools.combinations(range(3),2))
    return out


def cadec_metadata():
    """Regenerate exact submitted sentence order, with original post identities."""
    path = RESULTS / 'cadec_with_groups.csv'
    with contextlib.redirect_stdout(io.StringIO()):
        df, _ = harmonise_cadec(str(CADEC.parent / 'cadec'), str(path))
    submitted = pd.read_csv(CADEC)
    if not df[['text','label']].equals(submitted[['text','label']]):
        raise ValueError('CADEC reconstruction differs from submitted texts/labels; stop before pairing')
    if len(df) != 7823 or df.label.sum() != 2865 or df.post_id.nunique() != 1248:
        raise ValueError('Unexpected submitted CADEC cardinalities')
    return df


def classical_predictions(parts, cadec):
    train, cal, test = parts
    cfg = json.loads((ROOT/'configs/default_config.json').read_text())['primary_adr_detection']['classical_models']['tfidf']
    vec = TfidfVectorizer(ngram_range=tuple(cfg['ngram_range']), max_features=cfg['max_features'])
    X = [vec.fit_transform(train.text), vec.transform(cal.text), vec.transform(test.text), vec.transform(cadec.text)]
    models = {'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
              'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=.05, num_leaves=31, random_state=42, n_jobs=4, verbose=-1)}
    arms = {}
    for name, clf in models.items():
        clf.fit(X[0], train.label)
        pc, pt, po = [clf.predict_proba(x)[:,1] for x in X[1:]]
        ts = TemperatureScaler().fit(cal.label.to_numpy(), pc)
        iso = IsotonicRegression(out_of_bounds='clip').fit(pc, cal.label)
        for method, transform in [('Uncalibrated', lambda p:p), ('TempScale', ts.transform), ('Isotonic', iso.predict)]:
            arms[f'{name} + {method}'] = {'model':name, 'calib':transform(pc), 'test':transform(pt), 'cadec':transform(po)}
    return arms


def transformer_predictions(path, model, parts, cadec):
    with np.load(path, allow_pickle=False) as z:
        for split, df in zip(['train','calib','test'], parts):
            if not np.array_equal(z[f'{split}_texts'].astype(str), df.text.astype(str).to_numpy()) or not np.array_equal(z[f'y_{split}'], df.label.to_numpy()):
                raise ValueError(f'{path}: {split} text/label alignment failed')
        if not np.array_equal(z['cadec_texts'].astype(str), cadec.text.astype(str).to_numpy()) or not np.array_equal(z['y_cadec'], cadec.label.to_numpy()):
            raise ValueError(f'{path}: CADEC alignment failed')
        if all(f'probs_calib_{suffix}' in z for _,suffix in METHODS):
            return {f'{model} + {method}':{'model':model,'calib':z[f'probs_calib_{suffix}'][:,1].copy(),
                    'test':z[f'probs_test_{suffix}'][:,1].copy(),'cadec':z[f'probs_cadec_{suffix}'][:,1].copy()} for method,suffix in METHODS}
        logits = z['logits_calib']; odds = logits[:,1] - logits[:,0]
        pc = softmax(logits, axis=1)[:,1]
        T = minimize_scalar(lambda t:log_loss(parts[1].label, expit(odds/t), labels=[0,1]), bounds=(.01,10), method='bounded').x
        iso = IsotonicRegression(out_of_bounds='clip').fit(pc, parts[1].label)
        cal_probs = {'uncal':pc, 'temp':expit(odds/T), 'iso':iso.predict(pc)}
        return {f'{model} + {method}': {'model':model, 'calib':cal_probs[suffix], 'test':z[f'probs_test_{suffix}'][:,1].copy(), 'cadec':z[f'probs_cadec_{suffix}'][:,1].copy()} for method,suffix in METHODS}


def auc_sorted(y, p):
    """Mann-Whitney AUROC with exact handling of repeated/tied scores."""
    positives = y.sum(); negatives = len(y)-positives
    if not positives or not negatives: return np.nan
    ends = np.r_[np.flatnonzero(np.diff(p))+1, len(p)]
    starts = np.r_[0, ends[:-1]]
    pos = np.add.reduceat(y, starts)
    neg = ends-starts-pos
    return float(np.sum(pos * (np.cumsum(neg)-.5*neg))/(positives*negatives))


def ece_sorted(y, p, bins):
    size = max(1,len(y)//bins)
    starts = np.arange(0, min(bins*size,len(y)), size)
    return float(np.abs(np.add.reduceat(y-p, starts)).sum()/len(y))


def joint_bootstrap(y, probabilities, groups, bins, repeats, seed, clustered=True):
    """Shared cluster draws across every arm; rank bins match submission estimator.

    Percentile intervals are used for both row and cluster comparisons. They
    need not have ordered widths. Same-score rows retain corpus order, including
    all repeats, so bootstrap draws use a deterministic rank-bin convention.
    """
    y = np.asarray(y, dtype=float); matrix = np.asarray(probabilities, dtype=float)
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    members = [np.flatnonzero(inverse==g) for g in range(inverse.max()+1)]
    if clustered and len(members)<2: raise ValueError('At least two clusters required')
    rng = np.random.RandomState(seed)
    n_arms = matrix.shape[1]
    auc = np.full((repeats,n_arms), np.nan)
    ece = np.empty((repeats,n_arms,len(bins)))
    orders = [np.argsort(matrix[:,a], kind='stable') for a in range(n_arms)]
    for b in range(repeats):
        if clustered:
            drawn = rng.randint(0,len(members),size=len(members))
            idx = np.concatenate([members[g] for g in drawn])
        else: idx = rng.randint(0,len(y),size=len(y))
        weights = np.bincount(idx,minlength=len(y))
        for a,order in enumerate(orders):
            ranked = np.repeat(order,weights[order]); yy=y[ranked]; pp=matrix[ranked,a]
            auc[b,a] = auc_sorted(yy,pp)
            for j,k in enumerate(bins):ece[b,a,j] = ece_sorted(yy,pp,k)
    return {'auc':auc, 'ece':ece}


def interval(samples, level=.95):
    values=np.asarray(samples);values=values[np.isfinite(values)]
    if not len(values):raise ValueError('No valid bootstrap samples')
    alpha=(1-level)/2
    return np.quantile(values,[alpha,1-alpha]).tolist()


def paired_table(y, matrix, names, draws):
    point=[roc_auc_score(y,matrix[:,i]) for i in range(len(names))]
    out=[]
    for a,b in itertools.combinations(range(len(names)),2):
        lo,hi=interval(draws[:,a]-draws[:,b])
        out.append({'arm_a':names[a],'arm_b':names[b],'delta':point[a]-point[b],'ci_lo':lo,'ci_hi':hi,'different_at_95pct':bool(lo>0 or hi<0)})
    return out


def pair_lookup(rows):
    out={}
    for r in rows:
        a,b=r['arm_a'],r['arm_b'];out[a,b]=(r['ci_lo'],r['ci_hi']);out[b,a]=(-r['ci_hi'],-r['ci_lo'])
    return out


def select(catalogue,tau,energy,delta,source_pairs,external_pairs=None):
    missing=[a['name'] for a in catalogue if a['energy_j_1000'] is None]
    status={'energy_status':'incomplete_missing_energy' if missing else 'complete_measured_catalogue','missing_energy_arms':missing}
    feasible=[a for a in catalogue if a['ece_ci_hi']<=tau+1e-12 and a['energy_j_1000'] is not None and a['energy_j_1000']<=energy+1e-12]
    if not feasible:return {'feasible':0,'selected':None,'leader':None,'candidates':[],**status}
    leader=max(feasible,key=lambda a:(a['auroc'],-a['energy_j_1000'],a['name']))
    def passes(arm,lookup):
        if arm['name']==leader['name']:return True
        lo,hi=lookup[leader['name'],arm['name']]
        return lo>=-delta and hi<=delta
    candidates=[a for a in feasible if passes(a,source_pairs) and (external_pairs is None or passes(a,external_pairs))]
    chosen=min(candidates,key=lambda a:(a['energy_j_1000'],-a['auroc'],a['ece_ci_hi'],a['name']))
    return {'feasible':len(feasible),'selected':chosen['name'],'leader':leader['name'],'candidates':[a['name'] for a in candidates],**status}


def grid(catalogue,config,source_pairs,external_pairs):
    rows=[]
    for tau,E in itertools.product(config['tau_grid'],config['energy_grid_j_per_1000']):
        result=select(catalogue,tau,E,config['illustrative_delta'],source_pairs)
        assisted=select(catalogue,tau,E,config['illustrative_delta'],source_pairs,external_pairs)
        rows.append({'tau':tau,'energy_budget':E,**result,'external_assisted_selected':assisted['selected']})
    assert_monotonic(rows)
    return rows


def assert_monotonic(rows):
    for key,other in [('tau','energy_budget'),('energy_budget','tau')]:
        for v in sorted({r[key] for r in rows}):
            counts=[r['feasible'] for r in sorted([r for r in rows if r[key]==v],key=lambda r:r[other])]
            if any(a>b for a,b in zip(counts,counts[1:])):raise AssertionError(f'Feasibility is not monotone: {key}={v}')


def sensitivity_threshold(y,p,target):
    positives=np.sort(np.asarray(p)[np.asarray(y)==1])[::-1]
    if not len(positives):raise ValueError('No positive calibration examples')
    return float(positives[int(np.ceil(target*len(positives)))-1])


def operating(y,p,groups,threshold):
    flag=np.asarray(p)>=threshold;y=np.asarray(y)
    gp=pd.DataFrame({'group':groups,'flag':flag,'y':y}).groupby('group').max()
    return {'threshold':threshold,'sensitivity':float(flag[y==1].mean()),'specificity':float((~flag[y==0]).mean()),
            'flagged_sentences':int(flag.sum()),'sentence_fraction':float(flag.mean()),
            'flagged_reviews':int(gp.flag.sum()),'review_count':len(gp),'review_fraction':float(gp.flag.mean()),
            'review_sensitivity':float(gp.loc[gp.y==1,'flag'].mean())}


def workload(arms,parts,cadec,seed):
    rows=[]
    for name,a in arms.items():
        for target in [.85,.90,.95]:
            threshold=sensitivity_threshold(parts[1].label,a['calib'],target)
            cal_sens=float((a['calib'][parts[1].label.to_numpy()==1]>=threshold).mean())
            for domain,df,g in [('test',parts[2],'review_id'),('cadec',cadec,'post_id')]:
                for policy in ['calibration_frozen','retrospective_matched_sensitivity']:
                    t=threshold if policy=='calibration_frozen' else sensitivity_threshold(df.label,a[domain],target)
                    rows.append({'seed':seed,'arm':name,'domain':domain,'threshold_policy':policy,'target_calibration_sensitivity':target,'calibration_sensitivity':cal_sens,
                        'retrospective_only':policy!='calibration_frozen',**operating(df.label,a[domain],df[g],t)})
    return rows


def load_energy():
    old=json.loads((ROOT/'results/frozen_split_reconciled.json').read_text())
    values={name:entry.get('inf_j_gross') for name,entry in old['energy_by_model'].items()}
    for name,stem in MODEL_FILES.items():
        folder='linux_energy' if ' CPU ' in name else 'gpu_energy'
        path=RESULTS/folder/(f'{stem}.json' if folder=='linux_energy' else f'{stem}_surface.json')
        if path.exists():
            data=json.loads(path.read_text());cells=data.get('cells',[])
            canonical=next((c for c in cells if c.get('batch_size')==64 and c.get('max_length')==128 and c.get('status')=='measured'),None)
            values[name]=canonical['gross_j_1000'] if canonical else None
    return values


def run_seed(seed,config,psytar,cadec,prediction_dirs):
    parts=grouped_split(psytar,seed)
    arms=classical_predictions(parts,cadec);inputs=[]
    for model,stem in MODEL_FILES.items():
        matches=[folder/f'{stem}_seed{seed}_predictions.npz' for folder in prediction_dirs if (folder/f'{stem}_seed{seed}_predictions.npz').exists()]
        if not matches:
            if model not in ['DistilBERT','PubMedBERT']:continue
            raise FileNotFoundError(f'Missing {stem}_seed{seed}_predictions.npz')
        # Explicit --predictions directories take precedence over submitted
        # artifacts; chosen paths and hashes are recorded in every output.
        path=matches[-1];arms.update(transformer_predictions(path,model,parts,cadec));inputs.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
    names=list(arms);energy=load_energy();domain_outputs={};draw_arrays={};ci_comparison=[];bin_rows=[]
    for domain,df,g in [('test',parts[2],'review_id'),('cadec',cadec,'post_id')]:
        print(f'[seed {seed}] {domain}: {len(df)} rows, {df[g].nunique()} clusters, {config["bootstrap_repeats"]} resamples',flush=True)
        y=df.label.to_numpy();matrix=np.column_stack([arms[n][domain] for n in names]);bins=config['ece_bin_sensitivity']
        primary=joint_bootstrap(y,matrix,df[g],bins,config['bootstrap_repeats'],config['bootstrap_seed'])
        iid=joint_bootstrap(y,matrix,df[g],[config['ece_bins']],config['bootstrap_repeats'],config['bootstrap_seed'],clustered=False)
        pairs=paired_table(y,matrix,names,primary['auc']);metrics={}
        for i,name in enumerate(names):
            p=matrix[:,i];j=bins.index(config['ece_bins']);ci=interval(primary['ece'][:,i,j]);iidci=interval(iid['ece'][:,i,0])
            metrics[name]={'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'ece':compute_ece_adaptive(y,p,config['ece_bins']),'ece_ci':ci,
                'brier':brier_score_loss(y,p),'nll':log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),'f1_at_05':f1_score(y,p>=.5),
                'clinical_utility':compute_clinical_utility_metrics(y,p),'decision_curve':compute_decision_curve_analysis(y,p)}
            ci_comparison.append({'seed':seed,'domain':domain,'arm':name,'iid_lower':iidci[0],'iid_upper':iidci[1],'cluster_lower':ci[0],'cluster_upper':ci[1],
                'iid_width':iidci[1]-iidci[0],'cluster_width':ci[1]-ci[0]})
            for k,bincount in enumerate(bins):
                lo,hi=interval(primary['ece'][:,i,k]);bin_rows.append({'seed':seed,'domain':domain,'arm':name,'bins':bincount,'ece':compute_ece_adaptive(y,p,bincount),'ci_lo':lo,'ci_hi':hi})
        domain_outputs[domain]={'N':len(df),'clusters':int(df[g].nunique()),'metrics':metrics,'paired_auroc':pairs}
        draw_arrays[f'{domain}_auroc']=primary['auc'];draw_arrays[f'{domain}_ece']=primary['ece']
    catalogue=[{'name':n,'model':arms[n]['model'],'auroc':domain_outputs['test']['metrics'][n]['auroc'],
                'ece_ci_hi':domain_outputs['test']['metrics'][n]['ece_ci'][1], 'ece':domain_outputs['test']['metrics'][n]['ece'],
                'energy_j_1000':energy.get(arms[n]['model'])} for n in names]
    src=pair_lookup(domain_outputs['test']['paired_auroc']);ext=pair_lookup(domain_outputs['cadec']['paired_auroc'])
    delta_rows=[]
    for delta in config['delta_grid']:
        for tau,E in [(config['tau_grid'][6],120),(.03,120),(.05,120),(.10,200)]:
            result=select(catalogue,tau,E,delta,src);assisted=select(catalogue,tau,E,delta,src,ext)
            pair=('PubMedBERT + Uncalibrated','DistilBERT + Uncalibrated');lo,hi=src[pair];elo,ehi=ext[pair]
            delta_rows.append({'seed':seed,'tau':tau,'energy_budget':E,'delta':delta,**result,'external_assisted_selected':assisted['selected'],
                'base_transformer_source_equivalent':lo>=-delta and hi<=delta,'base_transformer_external_equivalent':elo>=-delta and ehi<=delta,
                'minimum_delta_for_base_source_ci':max(abs(lo),abs(hi)),'minimum_delta_for_base_external_ci':max(abs(elo),abs(ehi))})
    wrows=workload(arms,parts,cadec,seed)
    wpairs=[]
    for domain in ['test','cadec']:
        for target in [.85,.90,.95]:
            for policy in ['calibration_frozen','retrospective_matched_sensitivity']:
                indexed={r['arm']:r for r in wrows if r['domain']==domain and r['target_calibration_sensitivity']==target and r['threshold_policy']==policy}
                for a,b in itertools.combinations(names,2):
                    wa,wb=indexed[a]['flagged_reviews'],indexed[b]['flagged_reviews']
                    wpairs.append({'seed':seed,'domain':domain,'threshold_policy':policy,'target_sensitivity':target,'arm_a':a,'arm_b':b,'delta_auroc_b_minus_a':domain_outputs[domain]['metrics'][b]['auroc']-domain_outputs[domain]['metrics'][a]['auroc'],
                        'relative_flagged_review_change_pct':100*(wb-wa)/max(wa,1),'review_fraction_difference_pp':100*(indexed[b]['review_fraction']-indexed[a]['review_fraction']),
                        'achieved_sensitivity_a':indexed[a]['sensitivity'],'achieved_sensitivity_b':indexed[b]['sensitivity']})
    np.savez_compressed(RESULTS/f'seed{seed}_bootstrap_draws.npz',arm_names=np.array(names),bins=np.array(config['ece_bin_sensitivity']),**draw_arrays)
    out={'seed':seed,'input_predictions':inputs,'split_sizes':[len(p) for p in parts],'split_review_counts':[int(p.review_id.nunique()) for p in parts],
         'domains':domain_outputs,'catalogue':catalogue,'grid':grid(catalogue,config,src,ext),'delta_sensitivity':delta_rows,'workload':wrows,'workload_pairs':wpairs,'ci_comparison':ci_comparison,'bin_sensitivity':bin_rows}
    write_json(RESULTS/f'seed{seed}_analysis.json',out)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seeds',nargs='+',type=int,default=None);ap.add_argument('--bootstrap',type=int);ap.add_argument('--predictions',nargs='*',default=[])
    args=ap.parse_args();config=json.loads((ROOT/'configs/revision_config.json').read_text())
    if args.bootstrap is not None:config['bootstrap_repeats']=args.bootstrap
    if config['bootstrap_repeats']<100:raise ValueError('Use at least 100 resamples; publication default is 2000')
    RESULTS.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    psytar=pd.read_csv(PSYTAR);cadec=cadec_metadata();seeds=args.seeds or config['existing_split_seeds']
    dirs=[ROOT/'results']+[Path(p).resolve() for p in args.predictions]
    runs=[run_seed(s,config,psytar,cadec,dirs) for s in seeds]
    for key in ['grid','delta_sensitivity','workload','workload_pairs','ci_comparison','bin_sensitivity']:
        rows=[{'seed':r['seed'],**row} for r in runs for row in r[key]];pd.DataFrame(rows).to_csv(RESULTS/f'{key}.csv',index=False)
    frequencies=[]
    # Never pool different candidate catalogues (CPU encoder artifacts may be
    # available only for primary seed 42); frequency denominators stay explicit.
    cohorts={tuple(a['name'] for a in run['catalogue']) for run in runs}
    for cohort in cohorts:
        selected_runs=[r for r in runs if tuple(a['name'] for a in r['catalogue'])==cohort]
        for tau,E in itertools.product(config['tau_grid'],config['energy_grid_j_per_1000']):
            picks=[row['selected'] or 'Infeasible' for run in selected_runs for row in run['grid'] if row['tau']==tau and row['energy_budget']==E]
            for arm in sorted(set(picks)):frequencies.append({'tau':tau,'energy_budget':E,'arm':arm,'count':picks.count(arm),'splits':len(picks),'frequency':picks.count(arm)/len(picks),'candidate_count':len(cohort),'cohort_seeds':[r['seed'] for r in selected_runs]})
    pd.DataFrame(frequencies).to_csv(RESULTS/'selection_frequencies.csv',index=False)
    energy_paths=[ROOT/'results/frozen_split_reconciled.json']
    energy_paths.extend(sorted((RESULTS/'gpu_energy').glob('*_surface.json')))
    energy_paths.extend(sorted((RESULTS/'linux_energy').glob('*.json')))
    summary={'provenance':{'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'script':'revision_analysis.py','config':config,'seeds':seeds,
            'ece_interval':'95% percentile, 2000 whole-cluster draws by default','resampling_unit_source':'review_id','resampling_unit_external':'post_id',
            'primary_selection':'source audit split only; external corpus nonbinding diagnostic','selection_performance_is_apparent':'selection and source discrimination use the same development audit split, not a new independent test',
            'energy_scope':'CPU package including TF-IDF; revision GPU board during raw-text pipeline timing when supplied, otherwise submitted forward-only benchmark; host energy and whole-system energy are not measured',
            'energy_inputs':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'canonical_text_sha256':canonical_text_sha256(p.read_bytes())} for p in energy_paths],
            'energy_operating_point':{'batch_size':64,'max_length':128,'revision_energy_checkpoint_seed':42,'shared_across_grouped_splits':True},
            'psytar_sha256':sha(PSYTAR),'cadec_sha256':sha(CADEC),'cadec_group_metadata_sha256':sha(RESULTS/'cadec_with_groups.csv'),
            'analysis_code_sha256':sha(Path(__file__)),'metrics_code_sha256':sha(ROOT/'scripts/metrics_utils.py'),
            'classical_config_sha256':sha(ROOT/'configs/default_config.json'),
            'library_versions':{'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':lgb.__version__},
            'ci_widths_need_not_be_ordered':True,'hardware_experiments_pending':True},'runs':runs,'selection_frequencies':frequencies}
    write_json(RESULTS/'revision_analysis.json',summary)
    print(f'Completed {len(runs)} grouped splits. Output: {RESULTS}',flush=True)


if __name__=='__main__':main()

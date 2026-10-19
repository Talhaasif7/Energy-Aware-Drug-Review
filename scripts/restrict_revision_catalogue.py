"""Remove execution variants from a saved analysis without changing retained draws.

Used to restore the planned revision catalogue after an extra, unplanned
PubMedBERT CPU run was returned. Shared bootstrap draws are column-independent;
retained metrics/intervals are unchanged and every selection is recomputed.
Raw input artifacts are preserved.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_analysis import ROOT,RESULTS,MODEL_FILES,grid,pair_lookup,select,sha,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exclude-models',nargs='+',required=True)
    args=parser.parse_args();excluded=set(args.exclude_models)
    path=RESULTS/'revision_analysis.json';original_sha=sha(path);summary=json.loads(path.read_text());cfg=summary['provenance']['config']
    removed_stems={MODEL_FILES[m] for m in excluded}
    for run in summary['runs']:
        old_names=[a['name'] for a in run['catalogue']]
        kept={a['name'] for a in run['catalogue'] if a['model'] not in excluded}
        if len(kept)==len(old_names):continue
        indices=[i for i,n in enumerate(old_names) if n in kept]
        draw_path=RESULTS/f"seed{run['seed']}_bootstrap_draws.npz"
        with np.load(draw_path,allow_pickle=False) as saved:
            arrays={key:(saved[key][indices] if key=='arm_names' else saved[key][:,indices,...] if key.endswith(('_auroc','_ece')) else saved[key]).copy() for key in saved.files}
        assert arrays['arm_names'].tolist()==[n for n in old_names if n in kept]
        np.savez_compressed(draw_path,**arrays)
        run['catalogue']=[a for a in run['catalogue'] if a['name'] in kept]
        run['input_predictions']=[r for r in run['input_predictions'] if not any(Path(r['path']).name.startswith(stem+'_seed') for stem in removed_stems)]
        for domain in ('test','cadec'):
            data=run['domains'][domain]
            data['metrics']={n:m for n,m in data['metrics'].items() if n in kept}
            data['paired_auroc']=[r for r in data['paired_auroc'] if r['arm_a'] in kept and r['arm_b'] in kept]
        source=pair_lookup(run['domains']['test']['paired_auroc']);external=pair_lookup(run['domains']['cadec']['paired_auroc'])
        run['grid']=grid(run['catalogue'],cfg,source,external)
        for row in run['delta_sensitivity']:
            row.update(select(run['catalogue'],row['tau'],row['energy_budget'],row['delta'],source))
            row['external_assisted_selected']=select(run['catalogue'],row['tau'],row['energy_budget'],row['delta'],source,external)['selected']
        for key in ('workload','ci_comparison','bin_sensitivity'):run[key]=[r for r in run[key] if r['arm'] in kept]
        run['workload_pairs']=[r for r in run['workload_pairs'] if r['arm_a'] in kept and r['arm_b'] in kept]
        write_json(RESULTS/f"seed{run['seed']}_analysis.json",run)
    for key in ('grid','delta_sensitivity','workload','workload_pairs','ci_comparison','bin_sensitivity'):
        pd.DataFrame([{'seed':run['seed'],**row} for run in summary['runs'] for row in run[key]]).to_csv(RESULTS/f'{key}.csv',index=False)
    frequencies=[]
    cohorts={tuple(a['name'] for a in run['catalogue']) for run in summary['runs']}
    for cohort in cohorts:
        runs=[r for r in summary['runs'] if tuple(a['name'] for a in r['catalogue'])==cohort]
        for tau,budget in itertools.product(cfg['tau_grid'],cfg['energy_grid_j_per_1000']):
            picks=[row['selected'] or 'Infeasible' for run in runs for row in run['grid'] if row['tau']==tau and row['energy_budget']==budget]
            for arm in sorted(set(picks)):
                frequencies.append({'tau':tau,'energy_budget':budget,'arm':arm,'count':picks.count(arm),'splits':len(picks),'frequency':picks.count(arm)/len(picks),'candidate_count':len(cohort),'cohort_seeds':[r['seed'] for r in runs]})
    summary['selection_frequencies']=frequencies;pd.DataFrame(frequencies).to_csv(RESULTS/'selection_frequencies.csv',index=False)
    primary=next(r for r in summary['runs'] if r['seed']==42)
    cpu=[a for a in primary['catalogue'] if ' CPU ' in a['model'] or a['model'] in ('Logistic Regression','LightGBM')]
    cpu_cfg=dict(cfg,energy_grid_j_per_1000=sorted(set(cfg['energy_grid_j_per_1000'])|{500,1000,2000,3000,4000}))
    summary['cpu_primary_grid']=grid(cpu,cpu_cfg,pair_lookup(primary['domains']['test']['paired_auroc']),pair_lookup(primary['domains']['cadec']['paired_auroc']))
    pd.DataFrame(summary['cpu_primary_grid']).to_csv(RESULTS/'cpu_primary_grid.csv',index=False)
    summary['provenance']['energy_inputs']=[r for r in summary['provenance']['energy_inputs'] if not r['path'].startswith('results/revision_pubmed_cpu/')]
    summary['provenance']['catalogue_restriction']={'excluded_models':sorted(excluded),'reason':'Author instruction: retain original PubMedBERT GPU baseline; exclude the extra CPU run outside the planned revision catalogue','retained_draws':'Identical shared cluster-bootstrap columns; all selection outcomes recomputed','source_summary_sha256':original_sha,'postprocessing_script_sha256':sha(Path(__file__))}
    summary['provenance']['include_extra_pubmed_cpu']=False
    write_json(path,summary)
    print(f"Primary catalogue: {len(primary['catalogue'])} arms; common catalogue unchanged at {len(summary['common_catalogue']['arm_names'])}; raw artifacts preserved.")


if __name__=='__main__':main()

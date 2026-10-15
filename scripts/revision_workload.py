"""Refresh both frozen-threshold and retrospective matched-sensitivity workload.

Uses existing cluster analysis; does not recompute confidence intervals. The
matched-sensitivity view is a retrospective ROC diagnostic, not a deployment rule.
"""
import itertools
import json
import numpy as np
import pandas as pd
from revision_analysis import ROOT,RESULTS,PSYTAR,CADEC,sha,grouped_split,classical_predictions,transformer_predictions,workload,write_json


def main():
    summary=json.loads((RESULTS/'revision_analysis.json').read_text())
    if sha(PSYTAR)!=summary['provenance']['psytar_sha256'] or sha(CADEC)!=summary['provenance']['cadec_sha256']:
        raise ValueError('Corpus changed; run complete revision_analysis.py')
    ps=pd.read_csv(PSYTAR);ca=pd.read_csv(RESULTS/'cadec_with_groups.csv')
    for run in summary['runs']:
        seed=run['seed'];parts=grouped_split(ps,seed);arms=classical_predictions(parts,ca)
        for item in run['input_predictions']:
            path=ROOT/item['path']
            if sha(path)!=item['sha256']:raise ValueError('Predictions changed; recompute cluster analysis')
            stem=path.name.split('_seed')[0]
            from revision_analysis import MODEL_FILES
            model=next(n for n,s in MODEL_FILES.items() if s==stem)
            arms.update(transformer_predictions(path,model,parts,ca))
        for name,a in arms.items():
            from sklearn.metrics import roc_auc_score
            if not np.isclose(roc_auc_score(parts[2].label,a['test']),run['domains']['test']['metrics'][name]['auroc'],atol=1e-10,rtol=0):
                raise ValueError('Classical model changed; recompute complete analysis')
        rows=workload(arms,parts,ca,seed);pairs=[]
        for domain,target,policy in itertools.product(['test','cadec'],[.85,.90,.95],['calibration_frozen','retrospective_matched_sensitivity']):
            index={r['arm']:r for r in rows if r['domain']==domain and r['target_calibration_sensitivity']==target and r['threshold_policy']==policy}
            for a,b in itertools.combinations(arms,2):
                wa,wb=index[a]['flagged_reviews'],index[b]['flagged_reviews']
                pairs.append({'seed':seed,'domain':domain,'threshold_policy':policy,'target_sensitivity':target,'arm_a':a,'arm_b':b,
                    'delta_auroc_b_minus_a':run['domains'][domain]['metrics'][b]['auroc']-run['domains'][domain]['metrics'][a]['auroc'],
                    'relative_flagged_review_change_pct':100*(wb-wa)/max(wa,1),'review_fraction_difference_pp':100*(index[b]['review_fraction']-index[a]['review_fraction']),
                    'achieved_sensitivity_a':index[a]['sensitivity'],'achieved_sensitivity_b':index[b]['sensitivity']})
        run['workload']=rows;run['workload_pairs']=pairs;write_json(RESULTS/f'seed{seed}_analysis.json',run)
        print(f'Seed {seed}: refreshed {len(rows)} operating points, {len(pairs)} workload pairs.',flush=True)
    for key in ['workload','workload_pairs']:
        pd.DataFrame([row for run in summary['runs'] for row in run[key]]).to_csv(RESULTS/f'{key}.csv',index=False)
    summary['provenance']['matched_sensitivity_workload']='Retrospective thresholds chosen on evaluated labels; not a deployment policy or clinical margin validation'
    write_json(RESULTS/'revision_analysis.json',summary)


if __name__=='__main__':main()

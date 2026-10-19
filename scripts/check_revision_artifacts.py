"""Check exported numerical tables, uncertainty arrays and DOCX figure placement."""
import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as E
from zipfile import ZipFile
import numpy as np
from revision_analysis import ROOT,RESULTS,REPORTS,assert_monotonic,sha
from revision_hardware import canonical_text_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-only',action='store_true',help='Verify numerical artifacts without claiming the manuscript/figures have been regenerated')
    args=parser.parse_args()
    summary=json.loads((RESULTS/'revision_analysis.json').read_text());cfg=summary['provenance']['config']
    assert cfg['bootstrap_repeats']==2000,'Publication artifacts require 2000 draws'
    for run in summary['runs']:
        assert len(run['grid'])==len(cfg['tau_grid'])*len(cfg['energy_grid_j_per_1000'])
        assert_monotonic(run['grid'])
        names=[a['name'] for a in run['catalogue']]
        with np.load(RESULTS/f'seed{run["seed"]}_bootstrap_draws.npz',allow_pickle=False) as draws:
            assert draws['arm_names'].tolist()==names
            for domain in ['test','cadec']:
                assert draws[f'{domain}_auroc'].shape==(2000,len(names))
                assert draws[f'{domain}_ece'].shape==(2000,len(names),len(cfg['ece_bin_sensitivity']))
                j=cfg['ece_bin_sensitivity'].index(cfg['ece_bins'])
                for i,name in enumerate(names):
                    expected=np.nanquantile(draws[f'{domain}_ece'][:,i,j],[.025,.975])
                    assert np.allclose(expected,run['domains'][domain]['metrics'][name]['ece_ci'],atol=1e-12)
                index={name:i for i,name in enumerate(names)}
                for pair in run['domains'][domain]['paired_auroc']:
                    delta=draws[f'{domain}_auroc'][:,index[pair['arm_a']]]-draws[f'{domain}_auroc'][:,index[pair['arm_b']]]
                    assert np.allclose(np.nanquantile(delta,[.025,.975]),[pair['ci_lo'],pair['ci_hi']],atol=1e-12)
        for row in run['workload']:
            if row['threshold_policy']=='retrospective_matched_sensitivity':
                assert row['sensitivity']+1e-12>=row['target_calibration_sensitivity']
            assert 0<=row['flagged_reviews']<=row['review_count']
        for item in run['input_predictions']:assert sha(ROOT/item['path'])==item['sha256']
        print(f'Seed {run["seed"]}: grid, clusters, draws, workload and artifact hashes verified.')
    for item in summary['provenance'].get('energy_inputs',[]):
        path=ROOT/item['path']
        same_bytes=sha(path)==item['sha256']
        same_text='canonical_text_sha256' in item and canonical_text_sha256(path.read_bytes())==item['canonical_text_sha256']
        assert same_bytes or same_text,f'Energy artifact changed: {item["path"]}'
    common=summary.get('common_catalogue')
    if common:
        intersection=set.intersection(*(set(a['name'] for a in run['catalogue']) for run in summary['runs']))
        assert set(common['arm_names'])==intersection
        for run in summary['runs']:
            rows=[r for r in common['grid'] if r['seed']==run['seed']]
            assert len(rows)==len(cfg['tau_grid'])*len(cfg['energy_grid_j_per_1000'])
            assert_monotonic(rows)
        for tau in cfg['tau_grid']:
            for budget in cfg['energy_grid_j_per_1000']:
                rows=[r for r in common['selection_frequencies'] if r['tau']==tau and r['energy_budget']==budget]
                assert sum(r['count'] for r in rows)==len(summary['runs'])
                assert all(r['splits']==len(summary['runs']) and r['candidate_count']==len(intersection) for r in rows)
        print('Common-catalogue frequencies include every grouped split with one consistent denominator.')
    if summary.get('cpu_primary_grid'):assert_monotonic(summary['cpu_primary_grid'])
    if args.analysis_only:
        print('Numerical artifact checks passed. Manuscript and figure checks were not requested.')
        return
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    with ZipFile(ROOT/'AIH Paper - Revised.docx') as z:
        body=E.fromstring(z.read('word/document.xml')).find('w:body',ns)
        tables=body.findall('w:tbl',ns);primary=next(r for r in summary['runs'] if r['seed']==42)
        assert len(tables[2].findall('w:tr',ns))==len(primary['catalogue'])+1,'Primary manuscript table has stale candidate rows'
        if common:assert len(tables[3].findall('w:tr',ns))==len(common['arm_names'])+1,'Multi-split table has mixed candidate catalogues'
        prose=' '.join(t.text or '' for t in body.findall('.//w:t',ns))
        for stale in ['Hardware extensions are pending','hardware runs remain pending','Ten-split encoder scripts are prepared, but additional split results are pending']:
            assert stale not in prose,stale
        rel={n.get('Id'):n.get('Target') for n in E.fromstring(z.read('word/_rels/document.xml.rels'))}
        last=None;verified=set()
        for block in body:
            refs=block.findall('.//a:blip',ns)
            if refs:last=rel[refs[-1].get('{'+ns['r']+'}embed')]
            text=''.join(t.text or '' for t in block.findall('.//w:t',ns))
            for number,name in [(7,'figure7_decision_curves.png'),(8,'figure8_regime.png')]:
                if text.startswith(f'Figure {number}.'):
                    assert z.read('word/'+last)==(REPORTS/'figures'/name).read_bytes(),f'Figure {number} does not match caption'
                    verified.add(number)
        assert verified=={7,8}
    print('DOCX figures 7/8 match regenerated decision-curve/regime files exactly.')
    for path in REPORTS.glob('*.md'):
        assert not any(ord(c)<32 and c not in '\n\r\t' for c in path.read_text(encoding='utf-8')),f'Bad escape in {path}'
    print('Revision artifact checks passed. Complete-system energy remains outside the available measurements.')


if __name__=='__main__':main()

"""Check exported numerical tables, uncertainty arrays and DOCX figure placement."""
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as E
from zipfile import ZipFile
import numpy as np
from revision_analysis import ROOT,RESULTS,REPORTS,assert_monotonic,sha


def main():
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
        for row in run['workload']:
            if row['threshold_policy']=='retrospective_matched_sensitivity':
                assert row['sensitivity']+1e-12>=row['target_calibration_sensitivity']
            assert 0<=row['flagged_reviews']<=row['review_count']
        for item in run['input_predictions']:assert sha(ROOT/item['path'])==item['sha256']
        print(f'Seed {run["seed"]}: grid, clusters, draws, workload and artifact hashes verified.')
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    with ZipFile(ROOT/'AIH Paper - Revised.docx') as z:
        body=E.fromstring(z.read('word/document.xml')).find('w:body',ns)
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
    print('Revision artifact checks passed. Hardware experiments are still pending.')


if __name__=='__main__':main()

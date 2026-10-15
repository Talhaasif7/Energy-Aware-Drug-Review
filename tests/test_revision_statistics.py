import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from metrics_utils import bootstrap_indices, bootstrap_ece, compute_ece_adaptive
from eccms_selection import _fast_auroc, paired_delta_auroc, get_energy
from revision_analysis import joint_bootstrap, auc_sorted, ece_sorted, assert_monotonic, select, sensitivity_threshold


class RevisionStatisticsTests(unittest.TestCase):
    def test_colab_and_linux_exports_load_without_pickle(self):
        import tempfile
        import pandas as pd
        from unittest.mock import patch
        from revision_colab import save_predictions as gpu_save
        from revision_linux import save_predictions as cpu_save
        df=pd.DataFrame({'review_id':['a','b','c','d','e','f'],'text':['text '+str(i) for i in range(6)],'label':[0,0,1,0,1,1]})
        def fake_infer(model,tokenizer,texts,*args,**kwargs):return np.linspace(.1,.9,len(texts))
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='revision-export-test-',dir=root/'tests') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(root))
            with patch('revision_colab.infer_all',side_effect=fake_infer):gpu_save(Path(folder)/'gpu.npz',None,None,[df,df,df],df,'cpu',42,'efficient_transformer')
            with patch('revision_linux.infer_all',side_effect=fake_infer):cpu_save(Path(folder)/'cpu.npz',None,None,[df,df,df],df)
            for file in Path(folder).glob('*.npz'):
                with np.load(file,allow_pickle=False) as saved:
                    for key in saved.files:self.assertNotEqual(saved[key].dtype.kind,'O',key)
                    np.testing.assert_array_equal(saved['test_texts'],df.text.to_numpy())
                    self.assertEqual(saved['probs_cadec_iso'].shape,(6,2))

    def test_checkpoint_hash_survives_git_line_ending_conversion(self):
        from revision_hardware import canonical_text_sha256
        self.assertEqual(canonical_text_sha256(b'header\r\nreview\r\n'),canonical_text_sha256(b'header\nreview\n'))
        self.assertNotEqual(canonical_text_sha256(b'review A\n'),canonical_text_sha256(b'review B\n'))

    def test_unknown_cpu_has_no_invented_tdp(self):
        from rapl_utils import _spec_tdp_watts
        self.assertIsNone(_spec_tdp_watts('unspecified processor'))
        self.assertEqual(_spec_tdp_watts('Intel Core i5-8500'),65)

    def test_cluster_draws_keep_every_member_and_multiplicity(self):
        groups=np.array(['a','a','b','c','c','c'])
        for seed in range(20):
            idx=bootstrap_indices(6,np.random.RandomState(seed),groups)
            counts=np.bincount(idx,minlength=6)
            for g in set(groups):self.assertEqual(len(set(counts[groups==g])),1)
        with self.assertRaises(ValueError):bootstrap_indices(6,np.random.RandomState(0),['a']*6)

    def test_ties_match_sklearn(self):
        rng=np.random.RandomState(1)
        for _ in range(20):
            y=rng.randint(0,2,80);p=rng.randint(0,4,80)/3
            self.assertAlmostEqual(_fast_auroc(y,p),roc_auc_score(y,p))
            order=np.argsort(p,kind='stable')
            self.assertAlmostEqual(auc_sorted(y[order],p[order]),roc_auc_score(y,p))
        self.assertEqual(_fast_auroc(np.array([0,1,0,1]),[.5]*4),.5)

    def test_joint_bootstrap_preserves_perfect_pairing(self):
        y=np.array([0,1,1,0,0,1,0,1]);p=np.array([.1,.9,.7,.3,.2,.8,.4,.6])
        out=joint_bootstrap(y,np.column_stack([p,p]),np.repeat(np.arange(4),2),[5,10],100,7)
        np.testing.assert_allclose(out['auc'][:,0],out['auc'][:,1],equal_nan=True)
        np.testing.assert_allclose(out['ece'][:,0],out['ece'][:,1])

    def test_sorted_ece_matches_primary(self):
        y=np.array([0,1,0,1,1,0,1,0,1,0,1]);p=np.array([.2,.2,.4,.5,.6,.6,.7,.8,.9,.9,.9])
        order=np.argsort(p,kind='stable')
        for bins in [5,10,15,20]:self.assertAlmostEqual(ece_sorted(y[order],p[order],bins),compute_ece_adaptive(y,p,bins))

    def test_cache_includes_labels(self):
        p=np.array([.1,.2,.3,.4]);q=np.array([.4,.3,.2,.1])
        a=paired_delta_auroc([0,0,1,1],p,q,n_bootstrap=100)
        b=paired_delta_auroc([1,1,0,0],p,q,n_bootstrap=100)
        self.assertEqual(a[0],1);self.assertEqual(b[0],-1)

    def test_single_class_variable_length_cluster_draws_are_skipped(self):
        y=np.array([1,1,1,0,0]);groups=np.array(['a','a','a','b','b'])
        result=paired_delta_auroc(y,y.astype(float),1-y.astype(float),n_bootstrap=100,groups=groups)
        self.assertEqual(result,(1,1,1))

    def test_cluster_intervals_use_percentile_not_row_jackknife(self):
        args=(np.array([0,0,1,1,0,1]),np.array([.1,.2,.8,.9,.3,.7]))
        a=bootstrap_ece(*args,n_bootstrap=100,groups=['a','a','b','b','c','c'],use_bca=True)
        b=bootstrap_ece(*args,n_bootstrap=100,groups=['a','a','b','b','c','c'],use_bca=False)
        self.assertEqual(a,b)

    def test_monotonicity_detects_reviewers_failure(self):
        assert_monotonic([{'tau':.07,'energy_budget':120,'feasible':10},{'tau':.07,'energy_budget':200,'feasible':11}])
        with self.assertRaises(AssertionError):assert_monotonic([{'tau':.07,'energy_budget':120,'feasible':10},{'tau':.07,'energy_budget':200,'feasible':0}])

    def test_equivalence_does_not_confuse_nonsignificance_with_a_tie(self):
        cat=[{'name':'expensive','auroc':.9,'ece_ci_hi':.04,'energy_j_1000':100},{'name':'cheap','auroc':.89,'ece_ci_hi':.04,'energy_j_1000':1}]
        wide={('expensive','cheap'):(-.02,.04)}
        self.assertEqual(select(cat,.05,120,.015,wide)['selected'],'expensive')
        narrow={('expensive','cheap'):(.003,.014)}
        self.assertEqual(select(cat,.05,120,.015,narrow)['selected'],'cheap')
        self.assertEqual(select(cat,.03,120,.015,narrow)['feasible'],0)

    def test_sensitivity_threshold_and_unknown_energy(self):
        y=np.array([0]+[1]*10);p=np.linspace(0,1,11)
        t=sensitivity_threshold(y,p,.9)
        self.assertGreaterEqual(np.mean(p[y==1]>=t),.9)
        self.assertEqual(get_energy({'inf_j_gross':None},True),float('inf'))

    def test_board_trace_uses_timestamps_and_discloses_missing_samples(self):
        from revision_hardware import BoardTracer
        from types import SimpleNamespace
        tracer=BoardTracer();tracer.thread=SimpleNamespace(join=lambda **kwargs:None)
        tracer.readings=[(.1,10),(1,20),(1.9,10)]
        result=tracer.stop(0,2)
        self.assertAlmostEqual(result['energy_j'],29)
        self.assertAlmostEqual(result['boundary_extension_s'],.2)
        tracer.readings=[(1,10)]
        self.assertIsNone(tracer.stop(0,2)['energy_j'])

    def test_cpu_probability_and_dynamic_int8_path(self):
        import torch
        from types import SimpleNamespace
        from revision_hardware import probabilities
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(4,2)
            def forward(self,input_ids,attention_mask):
                return SimpleNamespace(logits=self.linear(input_ids.float()*attention_mask))
        def tokenizer(texts,**kwargs):
            return {'input_ids':torch.ones(len(texts),4,dtype=torch.long),'attention_mask':torch.ones(len(texts),4,dtype=torch.long)}
        model=Model().eval();quantized=torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8)
        for m in [model,quantized]:
            p=probabilities(m,tokenizer,['a','b'],'cpu',4)
            self.assertEqual(p.shape,(2,));self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>=0)&(p<=1)).all())


if __name__=='__main__':unittest.main()

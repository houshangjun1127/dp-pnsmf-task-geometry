"""Public synthetic rank32/oracle checks; no benchmark data or training."""
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pytest
from src.training import private_als as core
from scripts.run_private_als_comparison import simulate_pair,make_config
from scripts.recompute_r2_tables import loss

spec=importlib.util.spec_from_file_location('r2_math_oracle',Path(__file__).with_name('r2_dense_oracle.py'))
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('profile_id',['L1','L5'])
def test_rank32_two_arms_dense_reference(profile_id):
    p=next(p for p in json.loads((ROOT/'configs/private_als_r2_portable_v1.json').read_text())['profiles'] if p['id']==profile_id)
    rs=list(__import__('csv').DictReader((ROOT/'configs/private_als_r2_run_manifest_v1.csv').open(encoding='utf-8-sig')))
    rows=[x for x in rs if x['dataset']=='ml1m' and x['copy']=='1' and x['label']=='20260821' and x['profile']==profile_id]
    cfg=make_config(dict(N=64,M=640),p)
    rng=np.random.default_rng(20261012)
    histories=tuple(np.sort(rng.choice(cfg.M,80+(uid%9),replace=False)) if uid else np.array([],dtype=np.int64) for uid in range(cfg.N))
    pre,outputs=simulate_pair(histories,p,dict(N=64,M=640),rows)
    assert max(map(len,pre.capped_histories))<=60
    assert pre.present_mask.all() and np.array_equal(outputs[0]['initial_items'],outputs[1]['initial_items'])
    for out in outputs:
        private=out['row']['arm']=='private';v=out['initial_items'].copy()
        noise_rng=np.random.default_rng(np.random.SeedSequence(json.loads(out['row']['training_noise_entropy'])))
        for _ in range(p['T']):
            users=oracle.users(v,pre.local_histories,cfg.ridge,cfg.gravity,cfg.radius)
            draws=core.gaussian_draws(noise_rng,len(v),cfg.rank) if private else {k:None for k in ('gram','rhs','shared')}
            v,_=oracle.items(users,pre.capped_histories,len(v),cfg.ridge,cfg.gravity,cfg.radius,p['training_sigmas'],draws,private)
        u=oracle.users(v,pre.local_histories,cfg.ridge,cfg.gravity)
        np.testing.assert_allclose(out['items'][pre.frequent_ids],v,atol=1e-10,rtol=1e-8)
        np.testing.assert_allclose(out['users'],u,atol=1e-10,rtol=1e-8)
        rare=np.setdiff1d(np.arange(cfg.M),pre.frequent_ids)
        assert not np.any(out['items'][rare])
        assert len(out['events'])==2+3*p['T']
        if private:assert abs(out['accounting']['epsilon_library']-p['epsilon_library'])<=1e-10
        else:
            assert out['accounting']['private'] is False and out['accounting']['epsilon_library'] is None
            assert all(x['noise_sd']==0 for x in out['events'][2:])

def test_full_iid_sym_svec_covariance():
    z=np.random.default_rng(20261012).normal(size=(40000,4,4));sym=(z+z.swapaxes(1,2))/2
    assert abs(np.var(sym[:,0,0])-1)<.05
    assert abs(np.var(sym[:,0,1])-.5)<.025
    assert abs(np.var(math.sqrt(2)*sym[:,0,1])-1)<.05

def test_als_empty_present_and_null_labels_are_distinct():
    cfg=core.Config(4,8,2,2,beta=.5)
    h=[[],[1,2],[2,3],[4,5]];all_present=core.preprocess(h,cfg,2.,123)
    mask=np.array([False,True,True,True]);null=core.preprocess(h,cfg,2.,123,present_mask=mask)
    assert all_present.present_mask[0] and not null.present_mask[0]
    np.testing.assert_array_equal(all_present.first_counts,null.first_counts)
    np.testing.assert_array_equal(all_present.second_counts,null.second_counts)
    # ALS's empty-history solution can be zero while presence labels still differ.
    v=np.ones((4,2));u,_=core.solve_users(v,null,cfg)
    assert not np.any(u[0]) and len(u)==4

def test_percentages_preserve_negative_and_undefined():
    assert loss(2,1)==-100
    assert loss(1,0)==''

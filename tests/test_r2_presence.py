"""Fixed-slot direct mechanism invariants, not a proof of production privacy."""
import numpy as np
import pytest
import torch
from dataclasses import fields,is_dataclass
from scipy.special import gammaln,logsumexp
from src.training.pnsmf_presence import run_presence_direct_round
from src.training.pnsmf_dp import run_uniform_dp_pnsmf_bgd_round

def same_diagnostics(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if is_dataclass(a):return all(same_diagnostics(getattr(a,f.name),getattr(b,f.name)) for f in fields(a))
    return a==b

def setup():
    g=torch.Generator().manual_seed(100)
    u=(torch.rand((32,4),generator=g,dtype=torch.float64)-.5)*.2
    v=(torch.rand((48,4),generator=g,dtype=torch.float64)-.5)*.2
    users=[];items=[]
    for uid in range(1,32):
        for j in range(1+uid%5):users.append(uid);items.append((7*uid+11*j)%48)
    kwargs=dict(learning_rate=.2,item_learning_rate=.03,sampling_probability=.3,clip_norm=.15,
        noise_multiplier=3.,omega=4.,regularization=.01,contribution_space='direct_item_gradient',contribution_rule='clip')
    return u,v,torch.tensor(users),torch.tensor(items),kwargs

def test_all_present_ten_rounds_identical():
    u,v,tu,ti,kwargs=setup();a=u.clone();b=u.clone();va=v.clone();vb=v.clone()
    rng=np.random.default_rng(20261009);mask=torch.ones(32,dtype=torch.bool)
    for t in range(10):
        selected=torch.tensor(np.flatnonzero(rng.random(32)<.3))
        a,va,da=run_uniform_dp_pnsmf_bgd_round(a,va,tu,ti,selected,generator=torch.Generator().manual_seed(200+t),**kwargs)
        b,vb,db=run_presence_direct_round(b,vb,tu,ti,selected,mask,generator=torch.Generator().manual_seed(200+t),**kwargs)
        assert torch.equal(a,b) and torch.equal(va,vb) and same_diagnostics(da,db)

def test_empty_present_is_not_null_and_qN_stays_public():
    u,v,tu,ti,kwargs=setup();selected=torch.tensor([0]);present=torch.ones(32,dtype=torch.bool);null=present.clone();null[0]=False
    up,_,dp=run_presence_direct_round(u,v,tu,ti,selected,present,generator=torch.Generator().manual_seed(303),**kwargs)
    un,_,dn=run_presence_direct_round(u,v,tu,ti,selected,null,generator=torch.Generator().manual_seed(303),**kwargs)
    assert not torch.equal(up[0],u[0]) and dp.aggregate_signal_norm>0
    assert torch.equal(un,u) and dn.aggregate_signal_norm==0
    assert dp.expected_clients==dn.expected_clients==.3*32
    with pytest.raises(ValueError):run_presence_direct_round(u,v,tu,ti,selected,present[:-1],**kwargs)

def test_main_poisson_moment_conversion_independent():
    orders=np.asarray(list(range(2,65))+[128,256],dtype=float);curve=[]
    for a in orders.astype(int):
        k=np.arange(a+1,dtype=float)
        terms=gammaln(a+1)-gammaln(k+1)-gammaln(a-k+1)+(a-k)*np.log(.9)+k*np.log(.1)+k*(k-1)/18
        curve.append(float(logsumexp(terms)/(a-1)))
    total=200*np.asarray(curve)
    eps=float(np.min(total+np.log1p(-1/orders)-np.log(1e-5*orders)/(orders-1)))
    assert abs(eps-2.193078238018666)<1e-10

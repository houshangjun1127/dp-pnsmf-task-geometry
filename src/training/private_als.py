"""T06 scale adaptation: same T05B equations; batched solves and item grouped Gram."""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Config:
    N: int; M: int; rank: int; k: int
    beta: float = .5
    radius: float = 1.
    ridge: float = .5
    gravity: float = .4
    def validate(self):
        assert 0 < self.N <= 9862 and 0 < self.M <= 11298
        assert self.rank in (2,4,32) and 0 < self.k <= self.M
        assert 0 < self.beta <= 1 and int(self.beta*self.M) >= self.rank
        assert self.radius > 0 and self.ridge > 0 and self.gravity >= 0

@dataclass
class Preprocessed:
    frequent_ids: np.ndarray
    first_counts: np.ndarray
    second_counts: np.ndarray
    local_histories: tuple
    capped_histories: tuple
    events: list
    present_mask: np.ndarray

def count_event(name, k, standard_deviation, space):
    return dict(name=name,query='bounded binary item count vector',
                sensitivity=math.sqrt(k),noise_sd=float(standard_deviation),
                gaussian_multiplier=float(standard_deviation/math.sqrt(k)),
                repetitions=1,noise_space=space,protected=True)

def preprocess(histories, cfg, count_sigma, seed, count_draws=None, present_mask=None):
    cfg.validate()
    assert len(histories)==cfg.N
    mask=np.ones(cfg.N,dtype=bool) if present_mask is None else np.asarray(present_mask,dtype=bool)
    assert mask.shape==(cfg.N,)
    h=tuple(np.asarray(x,dtype=np.int64) if mask[uid] else np.array([],dtype=np.int64)
            for uid,x in enumerate(histories))
    for x in h:
        assert len(np.unique(x))==len(x) and np.all((x>=0)&(x<cfg.M))
    # Per-slot random tape prevents another slot's deletion from shifting RNG use.
    sampled=[]
    for uid,x in enumerate(h):
        rng=np.random.default_rng(np.random.SeedSequence([seed,uid]))
        sampled.append(np.sort(rng.choice(x,min(cfg.k,len(x)),replace=False)))
    exact=np.zeros(cfg.M)
    for x in sampled: np.add.at(exact,x,1.)
    rng=np.random.default_rng(np.random.SeedSequence([seed,999999]))
    z1=rng.normal(size=cfg.M) if count_draws is None else count_draws[0]
    first=np.maximum(1.,exact+count_sigma*math.sqrt(cfg.k)*z1)
    # Stable public-original-ID ordering, never data-dependent exact counts.
    order=np.lexsort((np.arange(cfg.M),-first))
    frequent=order[:int(cfg.beta*cfg.M)]
    inv=np.full(cfg.M,-1,dtype=int); inv[frequent]=np.arange(len(frequent))
    local=[]; capped=[]
    for x in h:
        valid=x[inv[x]>=0]
        local.append(inv[valid])
        tail=valid[np.lexsort((valid,first[valid]))][:cfg.k]
        capped.append(inv[tail])
    count2=np.zeros(len(frequent))
    for x in capped: np.add.at(count2,x,1.)
    z2=rng.normal(size=len(frequent)) if count_draws is None else count_draws[1]
    second=np.maximum(1.,count2+count_sigma*math.sqrt(cfg.k)*z2)
    return Preprocessed(frequent,first,second,tuple(local),tuple(capped),
        [count_event('private_count_for_filter_tail',cfg.k,count_sigma*math.sqrt(cfg.k),'R^M'),
         count_event('private_count_after_tail',cfg.k,count_sigma*math.sqrt(cfg.k),'R^F conditional on first DP count')],mask.copy())

def project_psd(raw):
    symmetric=(raw+raw.T)/2
    values,vectors=np.linalg.eigh(symmetric)
    return (vectors*np.maximum(values,0))@vectors.T

def solve_users(items, pre, cfg, clip=True):
    systems=np.empty((cfg.N,cfg.rank,cfg.rank)); rhs=np.empty((cfg.N,cfg.rank))
    shared=cfg.gravity*(items.T@items)+cfg.ridge*np.eye(cfg.rank)
    for uid,h in enumerate(pre.local_histories):
        v=items[h]; systems[uid]=shared+v.T@v; rhs[uid]=v.sum(axis=0)
    raw=np.linalg.solve(systems,rhs[...,None])[...,0]
    residual=float(np.max(np.linalg.norm((systems@raw[...,None])[...,0]-rhs,axis=1)/(1+np.linalg.norm(rhs,axis=1))))
    result=raw*np.minimum(1.,cfg.radius/np.maximum(np.linalg.norm(raw,axis=1),1e-300))[:,None] if clip else raw
    return result,residual

def sufficient_statistics(users,pre,cfg):
    nf=len(pre.frequent_ids)
    grams=np.zeros((nf,cfg.rank,cfg.rank)); rhs=np.zeros((nf,cfg.rank))
    if not hasattr(pre,'item_user_groups'):
        groups=[[] for _ in range(nf)]
        for uid,h in enumerate(pre.capped_histories):
            for j in h: groups[j].append(uid)
        pre.item_user_groups=tuple(np.asarray(x,dtype=np.int64) for x in groups)
    for j,indices in enumerate(pre.item_user_groups):
        u=users[indices]; grams[j]=u.T@u; rhs[j]=u.sum(axis=0)
    return grams,rhs,users.T@users

def solve_items(users,pre,cfg,sigmas,draws,private=True):
    assert np.max(np.linalg.norm(users,axis=1)) <= cfg.radius*(1+1e-12)
    grams,rhs,shared=sufficient_statistics(users,pre,cfg)
    sG,sR,sK=sigmas
    if private:
        grams=grams+sG*cfg.radius**2*draws['gram']
        rhs=rhs+sR*cfg.radius*draws['rhs']
        shared=shared+sK*cfg.radius**2*draws['shared']
    # This order is the author practice branch; ridge is NOT PSD-clipped away.
    global_psd=project_psd(shared)
    symmetric=(grams+grams.swapaxes(1,2))/2
    values,vectors=np.linalg.eigh(symmetric)
    projected=(vectors*np.maximum(values,0)[:,None,:])@vectors.swapaxes(1,2)
    systems=projected+cfg.gravity*global_psd[None]+cfg.ridge*np.eye(cfg.rank)[None]
    items=np.linalg.solve(systems,rhs[...,None])[...,0]
    events=[]
    for name,sens,sd,space in [
        ('local_gram',math.sqrt(cfg.k)*cfg.radius**2,sG*cfg.radius**2,'full iid R^(F*r*r), then sym'),
        ('local_rhs',math.sqrt(cfg.k)*cfg.radius,sR*cfg.radius,'R^(F*r)'),
        ('shared_gram',cfg.radius**2,sK*cfg.radius**2,'full iid R^(r*r), then sym')]:
        events.append(dict(name=name,sensitivity=sens,noise_sd=sd if private else 0.,
            nominal_private_noise_sd=sd,
            gaussian_multiplier=sd/sens if private else 0.,repetitions=1,
            noise_space=space,protected=private))
    residual=float(np.max(np.linalg.norm((systems@items[...,None])[...,0]-rhs,axis=1)/(1+np.linalg.norm(rhs,axis=1))))
    return items,events,dict(grams=grams,rhs=rhs,shared=shared,systems=systems,item_residual=residual)

def gaussian_draws(rng,nf,rank):
    return dict(gram=rng.normal(size=(nf,rank,rank)),rhs=rng.normal(size=(nf,rank)),
                shared=rng.normal(size=(rank,rank)))

def restore_catalog(frequent_items,pre,cfg):
    all_items=np.zeros((cfg.M,cfg.rank)); all_items[pre.frequent_ids]=frequent_items
    return all_items

def run_pair(histories,cfg,sweeps,count_sigma,sigmas,seed_pre,seed_init,seed_noise):
    assert 1<=sweeps<=5
    pre=preprocess(histories,cfg,count_sigma,seed_pre)
    initial=np.random.default_rng(seed_init).normal(scale=.1,size=(len(pre.frequent_ids),cfg.rank))
    output={}; rng=np.random.default_rng(seed_noise)
    # Shared actual private preprocessing and equal initialization; two arms only.
    for arm in ['private','matched']:
        items=initial.copy(); events=list(pre.events); diagnostics=[]
        for sweep in range(sweeps):
            users,residual=solve_users(items,pre,cfg)
            draws=gaussian_draws(rng,len(pre.frequent_ids),cfg.rank)
            items,e,state=solve_items(users,pre,cfg,sigmas,draws,arm=='private')
            for event in e: event['sweep']=sweep+1
            events.extend(e)
            diagnostics.append(dict(sweep=sweep+1,user_residual=residual,
                clip_max=float(np.max(np.linalg.norm(users,axis=1))),
                condition_max=float(max(np.linalg.cond(x) for x in state['systems']))))
        evaluation_users,_=solve_users(items,pre,cfg,clip=False)
        output[arm]=dict(items=restore_catalog(items,pre,cfg),users=evaluation_users,
                         events=events,diagnostics=diagnostics)
    return output,pre

"""T05C v1: full-catalog user-macro ranking with original-public-ID stable ties.

Read evaluation_spec.md before use. Targets and exclusions contain original IDs,
not factor-row indices; default catalog IDs are one-based. No training/selection.
"""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch

SPEC_ID = 'T05C-full-catalog-original-ID-ties-v1-20261009'

def _ids(values, label):
    raw=np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f'{label}: expected a one-dimensional ID vector')
    if raw.size and not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f'{label}: IDs must be integers, not floats/strings')
    result=raw.astype(np.int64,copy=False)
    if np.unique(result).size != result.size:
        raise ValueError(f'{label}: duplicate IDs forbidden; not silently deduplicated')
    return result

def _prepare(user_factors,item_factors,targets_by_user,train_by_user,item_ids,cutoff,batch_size):
    u=torch.as_tensor(user_factors,device='cpu',dtype=torch.float64)
    v=torch.as_tensor(item_factors,device='cpu',dtype=torch.float64)
    if u.ndim!=2 or v.ndim!=2 or u.shape[1]!=v.shape[1]:
        raise ValueError('factor matrices must be rank two with equal embedding dimensions')
    if not torch.isfinite(u).all() or not torch.isfinite(v).all():
        raise ValueError('NaN/Inf factor values forbidden')
    n,m=u.shape[0],v.shape[0]
    if not isinstance(cutoff,int) or isinstance(cutoff,bool) or cutoff<=0 or cutoff>m:
        raise ValueError('cutoff must be an integer in [1,catalog size]')
    if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<=0:
        raise ValueError('batch_size must be a positive integer')
    if len(targets_by_user)!=n or len(train_by_user)!=n:
        raise ValueError('target/train sequences must cover all user rows')
    ids=_ids(np.arange(1,m+1,dtype=np.int64) if item_ids is None else item_ids,'item_ids')
    if len(ids)!=m:
        raise ValueError('item_ids must cover every declared catalog factor row')
    order=np.argsort(ids,kind='stable'); sorted_ids=ids[order]
    # Reorder only by original public ID; no noisy-count ordering involved.
    v=v.index_select(0,torch.as_tensor(order,dtype=torch.long))
    index={int(item):pos for pos,item in enumerate(sorted_ids)}
    target_ids=[]; excluded=[]
    for user in range(n):
        t=_ids(targets_by_user[user],f'targets user {user}')
        e=_ids(train_by_user[user],f'train user {user}')
        if any(int(x) not in index for x in t) or any(int(x) not in index for x in e):
            raise ValueError(f'user {user}: target/train ID outside declared full catalog')
        if np.intersect1d(t,e).size:
            raise ValueError(f'user {user}: train/test overlap forbidden')
        target_ids.append(t)
        excluded.append(np.asarray([index[int(x)] for x in e],dtype=np.int64))
    return u,v,sorted_ids,target_ids,excluded

def evaluate_factors(
    user_factors,
    item_factors,
    targets_by_user: Sequence[np.ndarray],
    train_by_user: Sequence[np.ndarray],
    *,
    item_ids=None,
    cutoff: int=20,
    batch_size: int=256,
    include_rankings: bool=False,
) -> dict:
    """Evaluate CPU float64 factors without filtering targets or train history.

    For ALS zero-fill rare item rows first (full declared catalog), or pass a
    full row-to-original-ID map. Local eval factors must be solved from train
    only; this evaluator does not solve or modify them. Insufficient candidates
    retain the user and return an explicit diagnostic. Duplicate/overlap errors
    raise before scoring. This function changes neither global thread count nor
    input factors. Callers choose threads and log them before evaluation.
    """
    u,v,ids,targets,excluded=_prepare(user_factors,item_factors,targets_by_user,train_by_user,item_ids,cutoff,batch_size)
    selected=[i for i,t in enumerate(targets) if t.size]
    if not selected:
        raise ValueError('at least one test-nonempty user is required')
    discounts=1/np.log2(np.arange(cutoff,dtype=np.float64)+2)
    ndcg_sum=recall_sum=0.; shortages=[]; internal_ties=boundary_ties=0; rows=[]; max_bytes=0
    with torch.no_grad():
        for start in range(0,len(selected),batch_size):
            users=selected[start:start+batch_size]
            scores=u.index_select(0,torch.as_tensor(users,dtype=torch.long)) @ v.T
            if not torch.isfinite(scores).all():
                raise ValueError('NaN/Inf scores forbidden, including finite-factor overflow')
            max_bytes=max(max_bytes,scores.numel()*scores.element_size())
            for row,user in enumerate(users):
                scores[row,torch.as_tensor(excluded[user],dtype=torch.long)]=-torch.inf
            # Columns already sorted by public ID; stable desc score sort
            # therefore exactly implements the lexicographic ranking rule.
            ranked=torch.argsort(scores,dim=1,descending=True,stable=True)
            for row,user in enumerate(users):
                available=v.shape[0]-len(excluded[user]); take=min(cutoff,available)
                positions=ranked[row,:take].cpu().numpy()
                ranked_ids=ids[positions]
                hits=np.isin(ranked_ids,targets[user])
                recall=float(hits.sum()/len(targets[user]))
                ndcg=float(discounts[:take][hits].sum()/discounts[:min(cutoff,len(targets[user]))].sum())
                recall_sum+=recall;ndcg_sum+=ndcg
                if available<cutoff: shortages.append(user)
                top_scores=scores[row,ranked[row,:take]].cpu().numpy()
                if len(top_scores)>1 and np.any(top_scores[1:]==top_scores[:-1]): internal_ties+=1
                if available>cutoff and scores[row,ranked[row,cutoff-1]]==scores[row,ranked[row,cutoff]]:
                    boundary_ties+=1
                if include_rankings:
                    rows.append({'user_row':user,'original_item_ids':ranked_ids.tolist(),
                        'ndcg':ndcg,'recall':recall,'complete_target_count':len(targets[user]),'available_candidates':available})
    result={'spec_id':SPEC_ID,'users':len(selected),f'ndcg@{cutoff}':ndcg_sum/len(selected),
        f'recall@{cutoff}':recall_sum/len(selected),
        'diagnostics':{'test_empty_users':u.shape[0]-len(selected),'candidate_shortage_users':shortages,
            'candidate_shortage_count':len(shortages),'topK_internal_tie_users':internal_ties,
            'topK_boundary_tie_users':boundary_ties,'max_score_block_bytes':max_bytes,'batch_size':batch_size}}
    if include_rankings: result['rankings']=rows
    return result

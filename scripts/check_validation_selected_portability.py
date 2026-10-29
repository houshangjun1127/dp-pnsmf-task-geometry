"""Data-free arithmetic/negative gates plus public synthetic rank32 smoke."""
import copy,json,pathlib,math
import numpy as np
from scripts.recompute_validation_selected_als import recompute,read,js,decide,expanded,convert,summarize_final
from scripts.run_validation_selected_als import simulate,evaluate
from scripts.run_private_als_comparison import simulate_pair
ROOT=pathlib.Path(__file__).resolve().parents[1]
def rejected(fn):
 try:fn()
 except (ValueError,AssertionError):return True
 raise AssertionError('Invalid input accepted')
def main():
 p=js(ROOT/'configs/validation_selected_als.json');r=ROOT/'results/validation_selected_als';records=read(r/'candidate_results.csv');a=recompute();checks=[]
 checks.append('archived36_candidates_36_final_18pairs_54P_18M')
 for name,fn in [('missing_candidate',lambda:decide(records[:-1],p)),('duplicate_candidate',lambda:decide(records[:-1]+[records[0]],p))]:rejected(fn);checks.append(name)
 for key,value in [('validation_ndcg20','nan'),('validation_ndcg20','1.1'),('label','2026102702'),('arm','matched_no_training_noise'),('score_target','test')]:
  x=copy.deepcopy(records);x[0][key]=value;rejected(lambda:decide(x,p));checks.append('invalid_'+key+'_'+value)
 tied=copy.deepcopy(records)
 for row in tied:row['validation_ndcg20']='.5'
 assert all(v=='C01' for v in decide(tied,p)['selected'].values());checks.append('stable_top_tie_order')
 ev=read(r/'ledger_events.csv');model=next(m for m in read(r/'model_identity.csv') if m['arm']=='private');one=[e for e in ev if e['run_id']==model['run_id']];c=next(c for c in p['candidates'] if c['id']==model['candidate'])
 for key,value in [('noise_sd','999'),('sensitivity','999'),('repetitions','2')]:
  x=copy.deepcopy(one);x[0][key]=value;rejected(lambda:convert(x,c,'private',p));checks.append('wrong_event_'+key)
 mm=read(r/'metrics_by_run.csv');rejected(lambda:summarize_final(mm[:-1]));checks.append('incomplete_final_metrics')
 zero=copy.deepcopy(mm)
 for row in zero:
  if row['arm']=='matched_no_training_noise':row['ndcg20']='0'
 assert all(q['loss_percent'] is None for q in summarize_final(zero)[0]);checks.append('zero_M_retained_undefined')
 rng=np.random.default_rng(9123);N=12;M=640;train=tuple(np.sort(rng.choice(M,40,replace=False)) for _ in range(N));target=tuple(np.setdiff1d(np.arange(M),h)[:25] for h in train);d=dict(N=N,M=M);errors=[]
 for c0 in p['candidates']:
  c=dict(c0,delta=p['accounting']['delta'],epsilon_library=p['accounting']['target_epsilon']);cid=c['id'];n=int(cid[1:]);row=dict(stage='selection',dataset='public_synthetic',candidate=cid,copy='1',label='2026102701',arm='private',pair_id='synthetic_'+cid,run_id='synthetic_'+cid+'_private',preprocess_entropy=json.dumps([20261027,1,1,1,2026102701,n,1]),init_entropy=json.dumps([20261027,1,1,1,2026102701,n,2]),training_noise_entropy=json.dumps([20261027,1,1,1,2026102701,n,3]));mate=dict(row,arm='matched_no_training_noise',run_id='synthetic_'+cid+'_matched')
  refpre,ref=simulate_pair(train,c,d,[row,mate]);shared=None
  for rr,expected in zip([row,mate],ref):
   result,shared=simulate(train,d,c,rr,shared);err=float(np.max(np.abs(result['items']-expected['items'])));ue=float(np.max(np.abs(result['users']-expected['users'])));assert np.allclose(result['items'],expected['items'],atol=1e-10,rtol=1e-8) and np.allclose(result['users'],expected['users'],atol=1e-10,rtol=1e-8);assert np.array_equal(shared[0].first_counts,refpre.first_counts) and np.array_equal(shared[1],expected['initial_items']);m=evaluate(result,train,target);assert m['users']==N;errors.append(dict(candidate=cid,arm=rr['arm'],item_max_error=err,user_max_error=ue,ndcg20=m['ndcg@20'],recall20=m['recall@20'],rank=32,k=60,N=N,M=M))
 checks.append('six_configs_two_arms_rank32_k60_synthetic_against_retained_runner')
 print(json.dumps(dict(passed=True,checks=checks,synthetic_rows=errors,real_training=False,new_real_factor_scoring=False,dense_oracle_rerun=False,scope='archived record arithmetic and public synthetic adapter smoke;not real retraining or an independent dense test'),allow_nan=False))
if __name__=='__main__':main()

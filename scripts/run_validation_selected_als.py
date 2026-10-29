"""Portable staged reproduction of the six-candidate comparison.
Importing this module does not read benchmark data, fit a model, or score factors.
"""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[_k]='4'
import argparse,csv,hashlib,json,pathlib,time
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
from threadpoolctl import threadpool_limits
threadpool_limits(4)
from src.training.private_als import Config,preprocess,solve_users,solve_items,gaussian_draws,restore_catalog
from src.evaluation.full_catalog_stable import evaluate_factors
from scripts.recompute_validation_selected_als import js,read,sha,need,decide,expanded,summarize_final,convert
ROOT=pathlib.Path(__file__).resolve().parents[1]
def save(p,x):
 with pathlib.Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def csvwrite(p,rr):
 with pathlib.Path(p).open('x',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def load_split(root,d,copy,split):
 entry=next(e for e in d['split_files'] if e['copy']==int(copy) and e['split']==split);p=pathlib.Path(root)/entry['path'];need(sha(p)==entry['sha256'],'Data hash mismatch')
 x=np.loadtxt(p,dtype=np.int64,ndmin=2);need(x.shape[1]==2 and len(x)>0 and x.min()>=1 and x[:,0].max()<=d['N'] and x[:,1].max()<=d['M'],'Public IDs');x=x-1;keys=x[:,0]*d['M']+x[:,1];need(len(np.unique(keys))==len(keys),'Duplicate interaction')
 order=np.argsort(x[:,0],kind='stable');x=x[order];ii=np.searchsorted(x[:,0],np.arange(d['N']+1));return tuple(np.sort(x[ii[u]:ii[u+1],1]) for u in range(d['N']))
def target_check(train,target,M):
 need(any(len(v) for v in target),'No target users');need(all(not np.intersect1d(x,y).size for x,y in zip(train,target)),'Train-target overlap');need(all(M-len(x)>=20 for x,y in zip(train,target) if len(y)),'Insufficient candidates')
def config(d,c):return Config(d['N'],d['M'],c['rank'],c['k'],c['beta'],c['Gamma'],c['ridge'],c['gravity'])
def simulate(train,d,c,row,shared=None):
 """Exact preprocessing/noise order of the executed work.train; arrays stay local."""
 cfg=config(d,c);private=row['arm']=='private'
 if shared is None:
  seed=int(np.random.SeedSequence(json.loads(row['preprocess_entropy'])).generate_state(1,dtype=np.uint64)[0]);pre=preprocess(train,cfg,c['normalized_count_sigma'],seed);initial=np.random.default_rng(np.random.SeedSequence(json.loads(row['init_entropy']))).normal(scale=.1,size=(len(pre.frequent_ids),cfg.rank))
 else:pre,initial=shared
 items=initial.copy();rng=np.random.default_rng(np.random.SeedSequence(json.loads(row['training_noise_entropy'])));events=[dict(e) for e in pre.events];diagnostics=[]
 for sweep in range(1,c['T']+1):
  u,ur=solve_users(items,pre,cfg);clip=float(np.linalg.norm(u,axis=1).max());need(ur<=1e-7 and clip<=cfg.radius*(1+1e-10),'User solve/clip')
  draws=gaussian_draws(rng,len(items),cfg.rank) if private else {k:None for k in ['gram','rhs','shared']};items,ev,st=solve_items(u,pre,cfg,c['training_sigmas'],draws,private);need(st['item_residual']<=1e-7 and np.isfinite(items).all(),'Item solve/finite')
  for e in ev:e['sweep']=sweep
  events.extend(ev);diagnostics.append(dict(sweep=sweep,user_residual=ur,item_residual=st['item_residual'],clip_max=clip))
 users,rr=solve_users(items,pre,cfg,clip=False);full=restore_catalog(items,pre,cfg);need(rr<=1e-7 and np.isfinite(users).all() and np.isfinite(full).all(),'Evaluation projection')
 return dict(users=users,items=full,events=events,diagnostics=diagnostics,projection_residual=rr,init_hash=hashlib.sha256(initial.tobytes()).hexdigest()),(pre,initial)
def evaluate(result,train,target):
 target_check(train,target,len(result['items']));m=evaluate_factors(result['users'],result['items'],tuple(x+1 for x in target),tuple(x+1 for x in train),cutoff=20,batch_size=256);need(m['diagnostics']['candidate_shortage_count']==0,'Candidate shortage');need(all(np.isfinite(m[k]) and 0<=m[k]<=1 for k in ['ndcg@20','recall@20']),'Invalid metric');return m
def folder(out,row):return out/'models'/row['run_id']
def inspect_model(out,row,config,p):
 home=folder(out,row);cm=js(home/'COMPLETE.json')
 for k,v in cm['files'].items():need(sha(home/k)==v,'Model artifact changed')
 rec=js(home/'record.json');need(rec['row']==row and rec['candidate']==config,'Model identity/config changed')
 ledger=js(home/'ledger.json');ev=[dict(e,event_index=i) for i,e in enumerate(ledger['events'],1)];convert(ev,config,row['arm'],p)
 return home,rec

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['preflight','selection','lock','final','test','summarize']);ap.add_argument('--data-root');ap.add_argument('--output',required=True);args=ap.parse_args();p=js(ROOT/'configs/validation_selected_als.json');template=read(ROOT/'configs/validation_selected_als_manifest.csv');out=pathlib.Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 ds={d['id']:d for d in p['datasets']};cs={c['id']:c for c in p['candidates']};binding=dict(protocol_sha256=sha(ROOT/'configs/validation_selected_als.json'),manifest_sha256=sha(ROOT/'configs/validation_selected_als_manifest.csv'))
 if args.phase=='preflight':
  need(args.data_root,'--data-root required');data=[]
  for d in ds.values():
   for copy in [1,2,3]:
    hh={k:load_split(args.data_root,d,copy,k) for k in ['train','valid','test']}
    for x,y in [('train','valid'),('train','test'),('valid','test')]:need(all(not np.intersect1d(a,b).size for a,b in zip(hh[x],hh[y])),'Split overlap')
    for k in ['valid','test']:target_check(hh['train'],hh[k],d['M'])
    data.append(dict(dataset=d['id'],copy=copy,counts={k:sum(map(len,v)) for k,v in hh.items()}))
  save(out/'preflight.json',dict(**binding,metadata=data,no_scoring=True));return
 need(js(out/'preflight.json')['protocol_sha256']==binding['protocol_sha256'],'Preflight protocol')
 if (out/'state.json').exists():
  state=js(out/'state.json');need(state['binding']==binding,'Changed run configuration');need(time.time()-state['started_unix']<86400,'24h batch limit')
 else:
  state=dict(binding=binding,started_unix=time.time());save(out/'state.json',state)
 if args.phase in ['selection','final']:
  need(args.data_root,'--data-root required');rr=[r for r in template if r['stage']=='selection'] if args.phase=='selection' else js(out/'selection_lock.json')['final_manifest'];shared=None;pair=None
  if args.phase=='final':
   locked=js(out/'selection_lock.json');need(all(locked[k]==binding[k] for k in binding),'Locked protocol/manifest')
   ss=[]
   for row0 in [v for v in template if v['stage']=='selection']:
    hh,_=inspect_model(out,row0,cs[row0['candidate']],p);need(sha(hh/'COMPLETE.json')==locked['candidates'][row0['run_id']],'Locked candidate changed');mm=js(hh/'validation.json');ss.append(dict(**row0,validation_ndcg20=mm['ndcg@20'],validation_recall20=mm['recall@20']))
   dd=decide(ss,p);need(all(dd[k]==locked[k] for k in ['selected','scores','ties']) and rr==expanded(p,template,dd['selected']),'Locked selection changed')
  for row in rr:
   home=folder(out,row);need(not home.exists(),'Existing model;no overwrite/retry');home.mkdir(parents=True);d=ds[row['dataset']];c=cs[row['candidate']];hist=load_split(args.data_root,d,row['copy'],'train');start=time.monotonic()
   if pair!=row['pair_id']:shared=None
   result,shared=simulate(hist,d,c,row,shared);pair=row['pair_id'];ev=[dict(e,event_index=i) for i,e in enumerate(result['events'],1)];budget=convert(ev,c,row['arm'],p)
   with (home/'factors.npz').open('xb') as f:np.savez_compressed(f,item_factors=result['items'],evaluation_user_factors=result['users'])
   save(home/'ledger.json',dict(events=result['events'],accounting=budget));save(home/'record.json',dict(row=row,candidate=c,diagnostics=result['diagnostics'],projection_residual=result['projection_residual'],init_hash=result['init_hash'],model_seconds=time.monotonic()-start))
   files=['factors.npz','ledger.json','record.json']
   if args.phase=='selection':
    target=load_split(args.data_root,d,row['copy'],'valid');metric=evaluate(result,hist,target);save(home/'validation.json',metric);files.append('validation.json')
   save(home/'COMPLETE.json',dict(files={n:sha(home/n) for n in files}));print(row['run_id'],flush=True)
  save(out/(args.phase+'_complete.json'),dict(**binding,models=[r['run_id'] for r in rr]));return
 if args.phase=='lock':
  need((out/'selection_complete.json').exists(),'All candidates first');scores=[];identities={}
  for row in [r for r in template if r['stage']=='selection']:
   home,rec=inspect_model(out,row,cs[row['candidate']],p);m=js(home/'validation.json');scores.append(dict(**row,validation_ndcg20=m['ndcg@20'],validation_recall20=m['recall@20']));identities[row['run_id']]=sha(home/'COMPLETE.json')
  decision=decide(scores,p);save(out/'selection_lock.json',dict(**binding,**decision,candidates=identities,final_manifest=expanded(p,template,decision['selected'])));return
 lock=js(out/'selection_lock.json');need(all(lock[k]==binding[k] for k in binding),'Locked configuration changed')
 for row in [r for r in template if r['stage']=='selection']:
  home,_=inspect_model(out,row,cs[row['candidate']],p);need(sha(home/'COMPLETE.json')==lock['candidates'][row['run_id']],'Locked candidate changed')
 rr=lock['final_manifest'];need(rr==expanded(p,template,lock['selected']),'Locked final expansion')
 if args.phase=='test':
  need(args.data_root and (out/'final_complete.json').exists(),'All36 final models first');homes={r['run_id']:inspect_model(out,r,cs[r['candidate']],p)[0] for r in rr};metrics=[]
  for row in rr:
   home=homes[row['run_id']];d=ds[row['dataset']];train=load_split(args.data_root,d,row['copy'],'train');target=load_split(args.data_root,d,row['copy'],'test')
   with np.load(home/'factors.npz',allow_pickle=False) as z:metric=evaluate(dict(users=z['evaluation_user_factors'],items=z['item_factors']),train,target)
   metrics.append(dict(**row,ndcg20=metric['ndcg@20'],recall20=metric['recall@20']))
  csvwrite(out/'test_metrics.csv',metrics);save(out/'test_complete.json',dict(selection_lock_sha256=sha(out/'selection_lock.json'),test_metrics_sha256=sha(out/'test_metrics.csv')));return
 if args.phase=='summarize':
  test=js(out/'test_complete.json');need(test['selection_lock_sha256']==sha(out/'selection_lock.json') and test['test_metrics_sha256']==sha(out/'test_metrics.csv'),'Test binding');pairs,s=summarize_final(read(out/'test_metrics.csv'));save(out/'summary.json',dict(selected=lock['selected'],pairs=pairs,summary=s))
if __name__=='__main__':main()

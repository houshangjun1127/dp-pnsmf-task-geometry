"""Recompute the bounded validation comparison from portable archived records.
Standard library only; no factors, interactions, training or scoring required.
"""
import argparse,csv,hashlib,json,math,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):
 with pathlib.Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def js(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def need(x,msg):
 if not x:raise ValueError(msg)
def finite(v):
 v=float(v);need(math.isfinite(v) and 0<=v<=1,'Invalid ranking metric');return v
def boolean(v):
 need(v in [True,False,'True','False','true','false'],'Invalid boolean');return v in [True,'True','true']
def decide(rows,p):
 need(len(rows)==36 and len({r['run_id'] for r in rows})==36,'Need all36 unique candidates')
 selected={};scores={};ties={}
 for d in p['datasets']:
  rr=[r for r in rows if r['dataset']==d['id']]
  need(len(rr)==18 and {(r['candidate'],int(r['copy'])) for r in rr}=={(c['id'],k) for c in p['candidates'] for k in [1,2,3]},'Candidate coverage')
  for r in rr:
   need(r['arm']=='private' and int(r['label'])==p['selection']['seed'] and r['stage']=='selection' and r['score_target']=='validation','Candidate phase/seed/arm/target')
   finite(r['validation_recall20']);finite(r['validation_ndcg20'])
  means={c['id']:sum(float(r['validation_ndcg20']) for r in rr if r['candidate']==c['id'])/3 for c in p['candidates']};best=max(means.values());tie=[c for c in p['selection']['tie_order'] if best-means[c]<=p['selection']['tie_absolute_tolerance']]
  selected[d['id']]=tie[0];scores[d['id']]=means;ties[d['id']]=tie
 return dict(selected=selected,scores=scores,ties=ties)
def expanded(p,template,selected):
 out=[]
 for rr in template:
  if rr['stage']!='final':continue
  r=dict(rr);cid=selected[r['dataset']];r['candidate']=cid
  for k in ['pair_id','run_id']:r[k]=r[k].replace('selected_config',cid)
  for k in ['preprocess_entropy','init_entropy','training_noise_entropy']:
   e=json.loads(r[k]);need(e[5]=='selected_config.id_numeric','Entropy placeholder');e[5]=int(cid[1:]);r[k]=json.dumps(e,separators=(',',':'))
  out.append(r)
 need(len(out)==36 and len({r['run_id'] for r in out})==36,'Final expansion coverage');return out
def convert(events,c,arm,p):
 T=c['T'];need(len(events)==2+3*T,'Actual event count');rho=0.;private=arm=='private'
 for i,e in enumerate(sorted(events,key=lambda x:int(x['event_index']))):
  count=i<2;n=['private_count_for_filter_tail','private_count_after_tail'][i] if count else ['local_gram','local_rhs','shared_gram'][(i-2)%3]
  sens=math.sqrt(c['k']) if count else [math.sqrt(c['k'])*c['Gamma']**2,math.sqrt(c['k'])*c['Gamma'],c['Gamma']**2][(i-2)%3]
  raw=100. if count else c['training_sigmas'][(i-2)%3]*[c['Gamma']**2,c['Gamma'],c['Gamma']**2][(i-2)%3];sd=raw if private or count else 0.
  need(e['name']==n and int(e['repetitions'])==1 and abs(float(e['sensitivity'])-sens)<1e-12 and abs(float(e['noise_sd'])-sd)<1e-12 and boolean(e['protected'])==(private or count),'Event channel/sensitivity/sd/protection')
  need(abs(float(e['gaussian_multiplier'])-(sd/sens if sd else 0))<1e-12,'Multiplier')
  if not count:need(int(e['sweep'])==1+(i-2)//3 and float(e['nominal_private_noise_sd'])==raw,'Sweep/nominal scale')
  if sd:rho+=(sens/sd)**2/2
 delta=p['accounting']['delta'];eps,order=min((max(0,a*rho+math.log1p(-1/a)-math.log(delta*a)/(a-1)),a) for a in p['accounting']['orders'])
 if private:need(abs(eps-p['accounting']['target_epsilon'])<=1e-10,'Actual private budget')
 return dict(rho=rho,epsilon=eps if private else None,optimal_order=order if private else None,delta=delta,counts_rho=.006,complete_M_DP_claim=False)
def summarize_final(metrics):
 need(len(metrics)==36 and len({r['run_id'] for r in metrics})==36,'All36 final metrics required');pairs=[]
 for pair in sorted({r['pair_id'] for r in metrics}):
  rr=[r for r in metrics if r['pair_id']==pair];need(len(rr)==2 and {r['arm'] for r in rr}=={'private','matched_no_training_noise'},'Pair arms')
  pr=next(r for r in rr if r['arm']=='private');ma=next(r for r in rr if r['arm']!='private');M=finite(ma['ndcg20']);P=finite(pr['ndcg20']);mr=finite(ma['recall20']);pv=finite(pr['recall20'])
  need(all(pr[k]==ma[k] for k in ['dataset','candidate','copy','label','preprocess_entropy','init_entropy']),'Paired identity')
  pairs.append(dict(pair_id=pair,dataset=pr['dataset'],candidate=pr['candidate'],copy=pr['copy'],label=pr['label'],M=M,P=P,P_minus_M=P-M,loss_percent=100*(M-P)/M if M else None,M_recall20=mr,P_recall20=pv))
 need(len(pairs)==18,'18 pairs');summaries=[]
 for level in ['copy','task']:
  keys=sorted({(r['dataset'],r['copy'] if level=='copy' else 'all') for r in pairs})
  for d,k in keys:
   pp=[r for r in pairs if r['dataset']==d and (level=='task' or r['copy']==k)];loss=[r['loss_percent'] for r in pp];defined=all(v is not None for v in loss)
   s=dict(level=level,dataset=d,copy=k,n_pairs=len(pp),n_models=2*len(pp),equal_weight_unit='seed within copy;copy within task',mean_paired_loss_percent=sum(loss)/len(loss) if defined else None,defined_percent_pairs=sum(v is not None for v in loss))
   for x in ['M','P','P_minus_M','M_recall20','P_recall20']:s[x]=sum(r[x] for r in pp)/len(pp)
   summaries.append(s)
 return pairs,summaries
def recompute(root=ROOT):
 p=js(root/'configs/validation_selected_als.json');r=root/'results/validation_selected_als';decision=decide(read(r/'candidate_results.csv'),p);lock=js(r/'selection_lock.json');template=read(root/'configs/validation_selected_als_manifest.csv');final=expanded(p,template,decision['selected'])
 need(all(decision[k]==lock[k] for k in ['selected','scores','ties']),'Stored selection differs');need(final==read(r/'final_run_manifest.csv')==lock['final_manifest'],'Final manifest differs')
 for filename,key in [('configs/validation_selected_als.json','protocol_sha256'),('configs/validation_selected_als_manifest.csv','manifest_sha256'),('results/validation_selected_als/candidate_results.csv','candidate_results_sha256'),('results/validation_selected_als/model_identity.csv','model_identity_sha256'),('results/validation_selected_als/ledger_events.csv','events_sha256')]:need(sha(root/filename)==lock[key],'Public lock content binding')
 ids=read(r/'model_identity.csv');events=read(r/'ledger_events.csv');need(len(ids)==72 and len({z['run_id'] for z in ids})==72 and len({e['event_id'] for e in events})==len(events),'Model/event identities')
 configs={c['id']:c for c in p['candidates']};accounts=[]
 for m in ids:
  ev=[e for e in events if e['run_id']==m['run_id']];a=convert(ev,configs[m['candidate']],m['arm'],p);accounts.append(dict(run_id=m['run_id'],arm=m['arm'],**a))
 need(len({e['run_id'] for e in events})==72,'Orphan event');need(sum(a['arm']=='private' for a in accounts)==54,'54P')
 metric=read(r/'metrics_by_run.csv');need({r['run_id'] for r in metric}=={r['run_id'] for r in final},'Final metric model identities');pairs,summaries=summarize_final(metric)
 stored={q['pair_id']:q for q in read(r/'paired_results.csv')}
 for q in pairs:
  for k in ['M','P','P_minus_M','loss_percent']:need(q[k] is None and stored[q['pair_id']][k] in ['','undefined'] or q[k] is not None and math.isclose(q[k],float(stored[q['pair_id']][k]),abs_tol=1e-12,rel_tol=1e-10),'Paired result differs')
 for level,name in [('task','summary_by_task.csv'),('copy','summary_by_copy.csv')]:
  ss=read(r/name)
  for s in [v for v in summaries if v['level']==level]:
   z=next(v for v in ss if v['dataset']==s['dataset'] and (level=='task' or v['copy']==s['copy']))
   for key,storedkey in [('M','M_ndcg20'),('P','P_ndcg20'),('P_minus_M','P_minus_M'),('mean_paired_loss_percent','mean_paired_loss_percent')]:need(math.isclose(s[key],float(z[storedkey]),abs_tol=1e-12,rel_tol=1e-10),'Aggregate differs')
 return dict(selected=decision['selected'],scores=decision['scores'],ties=decision['ties'],models=72,candidates=36,final_models=36,pairs=pairs,summary=summaries,accounting=accounts,training=False,real_factor_scoring=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output');args=ap.parse_args();result=recompute()
 if args.output:
  with pathlib.Path(args.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(dict(selected=result['selected'],models=72,pairs=len(result['pairs']),private_ledgers=54,matched_ledgers=18,training=False,real_factor_scoring=False)))
if __name__=='__main__':main()

"""No-data recomputation of all R2 pairs, summaries, comparisons and budgets."""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path, rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loss(private,matched):return '' if matched==0 else 100*(matched-private)/matched
def pairs_from_metrics(metrics):
    grouped=defaultdict(dict)
    for x in metrics:
        if x['arm'] in grouped[x['pair_id']]:raise ValueError('Duplicate arm')
        grouped[x['pair_id']][x['arm']]=x
    out=[]
    for _,arms in grouped.items():
        if set(arms)!={'private','matched_no_training_noise'}:raise ValueError('Incomplete pair')
        p=arms['private'];m=arms['matched_no_training_noise']
        x={k:p[k] for k in ('pair_id','dataset','copy','label','profile')}
        if any(p[k]!=m[k] for k in ('dataset','copy','label','profile')):raise ValueError('Pair identity mismatch')
        for name in ('ndcg20','recall20'):
            pv=float(p[name]);mv=float(m[name]);x.update({name+'_private':pv,name+'_matched':mv,
                name+'_absolute_difference_M_minus_P':mv-pv,name+'_relative_loss_percent':loss(pv,mv),name+'_loss_defined':mv!=0})
        out.append(x)
    return out

def derived(metrics, pn):
    pairs=pairs_from_metrics(metrics);groups=defaultdict(list)
    for x in metrics:groups[(x['dataset'],x['profile'],x['copy'],x['arm'])].append(x)
    summaries=[]
    for (ds,profile,copy,arm),rs in groups.items():
        if len(rs)!=3:raise ValueError('Expected three labels within each copy')
        x=dict(dataset=ds,profile=profile,copy=copy,arm=arm,scope='three labels conditional on this fixed copy',n_labels=3,no_confidence_interval=True)
        for metric in ('ndcg20','recall20'):
            vals=[float(r[metric]) for r in rs];x[metric+'_mean']=float(np.mean(vals));x[metric+'_sample_sd']=float(np.std(vals,ddof=1))
        summaries.append(x)
    pnmap={(x['dataset'],x['copy'],x['arm']):x for x in pn};comparison=[]
    for r in metrics:
        if int(r['label'])!=20260820+int(r['copy']):continue
        oldarm='P' if r['arm']=='private' else 'M';o=pnmap[(r['dataset'],r['copy'],oldarm)]
        comparison.append(dict(dataset=r['dataset'],copy=r['copy'],label=r['label'],profile=r['profile'],arm=r['arm'],als_run_id=r['run_id'],
            als_ndcg20=float(r['ndcg20']),als_recall20=float(r['recall20']),pnsmf_arm=oldarm,
            pnsmf_ndcg20=float(o['stable_ndcg']),pnsmf_recall20=float(o['stable_recall']),
            pnsmf_factor_path=o['factor_path'],pnsmf_factor_sha256=o['factor_sha256'],pnsmf_batch=o['batch'],
            shared_numeric_label_not_shared_noise=True,comparison_scope='same fixed split/evaluator; literature operating points, not optimality'))
    overview=[]
    for ds in ('amazon_kindle','ml1m'):
        for profile in ('L1','L5'):
            ps=[x for x in pairs if x['dataset']==ds and x['profile']==profile]
            if len(ps)!=9:raise ValueError('Expected nine pairs per dataset/profile')
            x=dict(dataset=ds,profile=profile,n_pairs=9,n_fixed_copies=3,conditional_only=True)
            for metric in ('ndcg20','recall20'):
                for arm in ('private','matched'):x[metric+'_'+arm+'_mean']=float(np.mean([r[metric+'_'+arm] for r in ps]))
                vals=[r[metric+'_relative_loss_percent'] for r in ps if r[metric+'_loss_defined']]
                x[metric+'_mean_paired_relative_loss_percent']=float(np.mean(vals)) if vals else ''
                x[metric+'_undefined_loss_pairs']=len(ps)-len(vals)
            overview.append(x)
    table9=[]
    for ds in ('ml1m','amazon_kindle'):
        for method in ('ALS L1','ALS L5','P-NSMF'):
            if method=='P-NSMF':
                rs=[x for x in pn if x['dataset']==ds];p=[x for x in rs if x['arm']=='P'];m=[x for x in rs if x['arm']=='M']
                keyed={x['copy']:x for x in m}
                ls=[loss(float(x['stable_ndcg']),float(keyed[x['copy']]['stable_ndcg'])) for x in p]
                if len(p)!=3 or len(m)!=3:raise ValueError('Expected only three P-NSMF pairs')
                priv={name:float(np.mean([float(x['stable_'+name]) for x in p])) for name in ('ndcg','recall')}
                match={name:float(np.mean([float(x['stable_'+name]) for x in m])) for name in ('ndcg','recall')}
                t=dict(dataset=ds,method=method,n_pairs=3,matched={k+'20':v for k,v in match.items()},private={k+'20':v for k,v in priv.items()},paired_loss_pct=float(np.mean([x for x in ls if x!=''])) if any(x!='' for x in ls) else None)
            else:
                x=next(x for x in overview if x['dataset']==ds and x['profile']==method.split()[1])
                t=dict(dataset=ds,method=method,n_pairs=9,matched={k:x[k+'_matched_mean'] for k in ('ndcg20','recall20')},private={k:x[k+'_private_mean'] for k in ('ndcg20','recall20')},paired_loss_pct=x['ndcg20_mean_paired_relative_loss_percent'])
            table9.append(t)
    return dict(paired_losses=pairs,summary_descriptive=summaries,comparison_by_copy=comparison,overview_fixed_copies=overview),table9

def compare_rows(actual, expected, keys, tolerance=1e-10):
    a={tuple(str(x[k]) for k in keys):x for x in actual};b={tuple(str(x[k]) for k in keys):x for x in expected}
    if len(a)!=len(actual) or len(b)!=len(expected) or set(a)!=set(b):raise ValueError('Row identity mismatch: '+str(keys))
    worst=0.
    for key,x in a.items():
        for field,value in x.items():
            original=b[key][field]
            if isinstance(value,(float,int)) and not isinstance(value,bool):
                err=abs(float(original)-value);worst=max(worst,err)
                if not math.isfinite(err) or err>tolerance:raise ValueError('Numeric mismatch: '+str(key)+field)
            elif str(value)!=str(original):raise ValueError('Non-numeric mismatch: '+str(key)+field)
    return worst

def check_ledger(events, summaries, protocol):
    by=defaultdict(list)
    for e in events:by[e['run_id']].append(e)
    profiles={p['id']:p for p in protocol['profiles']};orders=protocol['orders'];out=[]
    if len(by)!=72 or len(summaries)!=72:raise ValueError('Ledger model coverage')
    for s in summaries:
        p=profiles[s['profile']];private=s['arm']=='private';es=by[s['run_id']]
        expected=[('private_count_for_filter_tail','',math.sqrt(p['k']),p['raw_count_sd'],True),('private_count_after_tail','',math.sqrt(p['k']),p['raw_count_sd'],True)]
        for t in range(1,p['T']+1):
            for name,sens,sd in [('local_gram',math.sqrt(p['k'])*p['Gamma']**2,p['training_sigmas'][0]*p['Gamma']**2),('local_rhs',math.sqrt(p['k'])*p['Gamma'],p['training_sigmas'][1]*p['Gamma']),('shared_gram',p['Gamma']**2,p['training_sigmas'][2]*p['Gamma']**2)]:
                expected.append((name,str(t),sens,sd if private else 0.,private))
        if len(es)!=len(expected):raise ValueError('Ledger event count')
        rho=0.
        for e,(name,t,sens,sd,protected) in zip(es,expected):
            if e['name']!=name or e['sweep']!=t or e['protected']!=str(protected) or int(e['repetitions'])!=1:raise ValueError('Event identity')
            if max(abs(float(e['sensitivity'])-sens),abs(float(e['noise_sd'])-sd),abs(float(e['gaussian_multiplier'])-sd/sens))>1e-12:raise ValueError('Event units')
            if private:rho+=sens*sens/(2*sd*sd)
        eps=min(a*rho+math.log1p(-1/a)-math.log(p['delta']*a)/(a-1) for a in orders) if private else None
        err=abs(eps-p['epsilon_library']) if private else 0.
        if private and (err>1e-10 or abs(eps-float(s['epsilon_library']))>1e-10):raise ValueError('Epsilon mismatch')
        if not private and (s['private']!='False' or s['epsilon_library']!=''):raise ValueError('Matched is not DP')
        out.append(dict(run_id=s['run_id'],arm=s['arm'],private=private,events=len(es),rho=rho if private else '',epsilon=eps if private else '',epsilon_target_error=err if private else '',status='pass'))
    return out

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir',type=Path,default=ROOT/'results/r2_private_als')
    ap.add_argument('--protocol',type=Path,default=ROOT/'configs/private_als_r2_portable_v1.json')
    ap.add_argument('--output-dir',type=Path,required=True,help='Fresh directory; source results remain read-only')
    args=ap.parse_args(argv);base=args.results_dir
    metrics=read(base/'metrics.csv');pn=read(base/'pnsmf_reference_metrics.csv');p=json.loads(args.protocol.read_text(encoding='utf-8'))
    frozen=read(ROOT/p['run_manifest']['path']);ids=('order','stage','pair_id','run_id','dataset','copy','label','profile','arm')
    if len(metrics)!=72 or len({r['run_id'] for r in metrics})!=72 or len(pn)!=12:raise ValueError('Incorrect run coverage')
    if [{k:x[k] for k in ids} for x in metrics]!=[{k:x[k] for k in ids} for x in frozen]:raise ValueError('Frozen identities differ')
    tables,t9=derived(metrics,pn)
    # Retain the full per-pair/per-copy reference tables. Aggregated summaries
    # and Table 9 are generated outputs, avoiding redundant archived copies.
    keys=dict(paired_losses=('pair_id',),comparison_by_copy=('als_run_id',))
    errors={name:compare_rows(tables[name],read(base/(name+'.csv')),key) for name,key in keys.items()}
    if len(t9)!=6:raise ValueError('Table9 coverage')
    budgets=check_ledger(read(base/'ledger_events.csv'),read(base/'ledger_summary.csv'),p)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items():write(args.output_dir/(name+'.csv'),table)
    write(args.output_dir/'accounting_recomputed.csv',budgets)
    result=dict(models=72,pairs=36,pnsmf_states=12,comparison_rows=24,private_models=36,matched_models=36,
        absolute_tolerance=1e-10,summary_max_errors=errors,
        generated_only=['summary_descriptive.csv','overview_fixed_copies.csv','table9.json'],
        max_private_epsilon_error=max(x['epsilon_target_error'] for x in budgets if x['private']),
        performed=['CSV summary recomputation','explicit event Gaussian RDP conversion'],
        not_performed=['factor evaluation','training','clean-machine reproduction',
            'comparison against an independently archived Table 9 in this public command'])
    for name,obj in [('table9.json',t9),('checks.json',result)]:
        with (args.output_dir/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

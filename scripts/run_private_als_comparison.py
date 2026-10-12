"""Re-run the frozen two-profile benchmark simulation in a fresh directory.

Importing this module never reads data or starts training. Public seeds and
saved user factors are research artifacts, not a production DP release.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
from src.data.pnsmf_fixed import load_pnsmf_fixed_split
from src.training.private_als import Config, preprocess, solve_users, solve_items, gaussian_draws, restore_catalog
from src.privacy.private_als_accounting import account
from src.evaluation.full_catalog_stable import evaluate_factors

ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def make_config(sizes, profile):
    return Config(sizes['N'],sizes['M'],profile['rank'],profile['k'],
        beta=profile['beta'],radius=profile['Gamma'],ridge=profile['ridge'],gravity=profile['gravity'])

def simulate_pair(histories, profile, sizes, rows, checkpoint=lambda:None):
    """Same scientific flow as the accepted runner; usable with public toy data."""
    if [r['arm'] for r in rows]!=['private','matched_no_training_noise']:
        raise ValueError('Expected private then matched, sharing one pair.')
    if rows[0]['pair_id']!=rows[1]['pair_id'] or any(rows[0][k]!=rows[1][k] for k in ('preprocess_entropy','init_entropy')):
        raise ValueError('Pair identities or shared random streams differ.')
    cfg=make_config(sizes,profile)
    seed_pre=int(np.random.SeedSequence(json.loads(rows[0]['preprocess_entropy'])).generate_state(1,dtype=np.uint64)[0])
    pre=preprocess(histories,cfg,profile['normalized_count_sigma'],seed_pre)
    initial=np.random.default_rng(np.random.SeedSequence(json.loads(rows[0]['init_entropy']))).normal(
        scale=.1,size=(len(pre.frequent_ids),cfg.rank))
    outputs=[]
    for row in rows:
        items=initial.copy();events=[dict(e) for e in pre.events];diagnostics=[]
        rng=np.random.default_rng(np.random.SeedSequence(json.loads(row['training_noise_entropy'])))
        private=row['arm']=='private'
        for sweep in range(profile['T']):
            users,res=solve_users(items,pre,cfg)
            clip=float(np.max(np.linalg.norm(users,axis=1)))
            if res>1e-7 or clip>cfg.radius*(1+1e-10):raise RuntimeError('Raw user solve/clip gate failed')
            z=gaussian_draws(rng,len(items),cfg.rank) if private else {k:None for k in ('gram','rhs','shared')}
            items,step,state=solve_items(users,pre,cfg,profile['training_sigmas'],z,private)
            if state['item_residual']>1e-7 or not np.isfinite(items).all():raise RuntimeError('Item solve/finite gate failed')
            for event in step:event['sweep']=sweep+1
            events.extend(step)
            diagnostics.append(dict(sweep=sweep+1,user_raw_relative_residual=res,
                item_relative_residual=state['item_residual'],clip_max=clip,
                condition_max=float(max(np.linalg.cond(x) for x in state['systems']))))
            checkpoint()
        u,res=solve_users(items,pre,cfg,clip=False);v=restore_catalog(items,pre,cfg)
        if res>1e-7 or not np.isfinite(u).all() or not np.isfinite(v).all():raise RuntimeError('Final projection gate failed')
        budget=account(events,profile['delta'])
        if private and abs(budget['epsilon_library']-profile['epsilon_library'])>1e-7:raise RuntimeError('Budget gate failed')
        if not private and (budget['private'] or any(e['noise_sd']!=0 for e in events[2:])):raise RuntimeError('Matched gate failed')
        outputs.append(dict(row=row,users=u,items=v,events=events,accounting=budget,
            diagnostics=diagnostics,evaluation_user_raw_relative_residual=res,
            initial_items=initial.copy(),preprocess_adapter_seed=seed_pre))
    return pre,outputs

def load_split(data_root, dataset, copy):
    entries={x['split']:Path(data_root)/x['path'] for x in dataset['split_files'] if x['copy']==copy}
    return load_pnsmf_fixed_split(entries['train'],entries['valid'],entries['test'],
        num_users=dataset['N'],num_items=dataset['M'])

def evaluate_saved(path, data, cutoff=20, batch_size=256):
    with np.load(path,allow_pickle=False) as z:
        return evaluate_factors(z['evaluation_user_factors'],z['item_factors'],
            [x+1 for x in data.test_by_user],[x+1 for x in data.train_by_user],
            item_ids=z['item_ids'],cutoff=cutoff,batch_size=batch_size)

class ResourceGate:
    """Original Windows memory units; non-Windows use requires explicit waiver."""
    def __init__(self, allow_unchecked):
        self.pilot_start=time.perf_counter();self.peak=None
        if sys.platform!='win32':
            if not allow_unchecked:raise RuntimeError('Windows memory gate required; other systems need --allow-unchecked-memory')
            self.memory=lambda:None
        else:
            import ctypes
            class PMC(ctypes.Structure):
                _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(x,ctypes.c_size_t) for x in
                    ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                     'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
            kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
            psapi=ctypes.WinDLL('psapi',use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong]
            def memory():
                v=PMC();v.cb=ctypes.sizeof(v)
                if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(v),v.cb):raise ctypes.WinError(ctypes.get_last_error())
                return max(v.PeakWorkingSetSize,v.PeakPagefileUsage)/2**30
            self.memory=memory
    def check(self):
        self.peak=self.memory()
        if self.peak is not None and self.peak>8:raise RuntimeError('8 GiB memory gate failed')
        if self.pilot_start is not None and time.perf_counter()-self.pilot_start>7200:raise RuntimeError('2-hour pilot gate failed')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=ROOT/'configs/private_als_r2_portable_v1.json')
    ap.add_argument('--output-dir',type=Path,required=True,help='Must not exist; no automatic retries/resume')
    ap.add_argument('--allow-unchecked-memory',action='store_true',help='Explicitly waive unvalidated non-Windows memory monitoring')
    args=ap.parse_args(argv)
    from threadpoolctl import threadpool_limits
    import torch
    threadpool_limits(limits=4);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    p=json.loads(args.protocol.read_text(encoding='utf-8'))
    rows=read_csv(ROOT/p['run_manifest']['path'])
    if sha(ROOT/p['run_manifest']['path'])!=p['run_manifest']['sha256']:raise RuntimeError('Manifest hash mismatch')
    if len(rows)!=72 or len({r['run_id'] for r in rows})!=72:raise RuntimeError('Expected exactly 72 models')
    for d in p['datasets']:
        for f in d['split_files']:
            if sha(args.data_root/f['path'])!=f['sha256']:raise RuntimeError('Data hash mismatch: '+f['path'])
    gate=ResourceGate(args.allow_unchecked_memory)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name in ('factors','runs','privacy_ledger'):(args.output_dir/name).mkdir()
    write_json(args.output_dir/'protocol_used.json',p)
    datasets={d['id']:d for d in p['datasets']};profiles={s['id']:s for s in p['profiles']}
    cache={};saved=[];pilot_times=[];start=time.perf_counter()
    for i in range(0,len(rows),2):
        pair=rows[i:i+2];r=pair[0];t=time.perf_counter()
        if r['stage']=='B' and gate.pilot_start is not None:raise RuntimeError('Missing pilot gate')
        key=(r['dataset'],int(r['copy']));d=datasets[key[0]]
        if key not in cache:cache.clear();cache[key]=load_split(args.data_root,d,key[1])
        data=cache[key];pre,outputs=simulate_pair(data.train_by_user,profiles[r['profile']],d,pair,gate.check)
        np.savez_compressed(args.output_dir/'factors'/f"{r['pair_id']}_preprocessing.npz",
            frequent_original_ids=pre.frequent_ids+1,first_noisy_counts=pre.first_counts,
            second_noisy_counts=pre.second_counts,initial_items=outputs[0]['initial_items'],present_mask=pre.present_mask)
        for out in outputs:
            row=out['row'];rid=row['run_id'];fp=args.output_dir/'factors'/f'{rid}.npz'
            np.savez_compressed(fp,item_factors=out['items'],evaluation_user_factors=out['users'],
                item_ids=np.arange(1,d['M']+1),user_ids=np.arange(1,d['N']+1),
                frequent_item_ids=pre.frequent_ids+1,run_id=np.array(rid))
            write_json(args.output_dir/'privacy_ledger'/f'{rid}.json',dict(events=out['events'],accounting=out['accounting']))
            record=dict(**row,status='completed',attempt=1,factor_path=f'factors/{rid}.npz',factor_sha256=sha(fp),
                frequent_items=len(pre.frequent_ids),train_frequent_fraction=sum(map(len,pre.local_histories))/data.train_count,
                zero_item_factor_rate=float(np.mean(np.linalg.norm(out['items'],axis=1)==0)),
                zero_user_factor_rate=float(np.mean(np.linalg.norm(out['users'],axis=1)==0)))
            write_json(args.output_dir/'runs'/f'{rid}.json',dict(**record,diagnostics=out['diagnostics']))
            saved.append(record)
        gate.check()
        if r['stage']=='A':pilot_times.append(time.perf_counter()-t)
        if i==6:
            elapsed=time.perf_counter()-gate.pilot_start;estimate=max(9*sum(pilot_times)*1.5+72*15,elapsed)
            if estimate>86400:raise RuntimeError('24-hour estimate gate failed')
            write_json(args.output_dir/'pilot_gate.json',dict(passed=True,models=8,technical_only=True,
                seconds=elapsed,estimated_seconds=estimate,peak_memory_GiB=gate.peak,
                memory_gate_checked=sys.platform=='win32',test_utility_not_evaluated=True))
            gate.pilot_start=None
        del outputs,pre
    if len(saved)!=72:raise RuntimeError('Incomplete training; evaluation forbidden')
    write_json(args.output_dir/'training_complete.json',dict(models=72,pairs=36,seconds=time.perf_counter()-start))
    write_csv(args.output_dir/'execution_manifest.csv',saved)
    metrics=[]
    for r in saved:
        d=datasets[r['dataset']];key=(r['dataset'],int(r['copy']))
        if key not in cache:cache.clear();cache[key]=load_split(args.data_root,d,key[1])
        out=evaluate_saved(args.output_dir/r['factor_path'],cache[key],p['evaluation']['cutoff'],p['evaluation']['batch_size'])
        m={k:r[k] for k in ('order','stage','pair_id','run_id','dataset','copy','label','profile','arm')}
        m.update(ndcg20=out['ndcg@20'],recall20=out['recall@20'],evaluated_users=out['users'],**out['diagnostics'])
        m['candidate_shortage_users']=json.dumps(m['candidate_shortage_users'])
        for k in ('factor_sha256','frequent_items','train_frequent_fraction','zero_item_factor_rate','zero_user_factor_rate'):m[k]=r[k]
        metrics.append(m);gate.check()
    write_csv(args.output_dir/'metrics.csv',metrics)
    print(json.dumps(dict(models=72,pairs=36,output_dir=str(args.output_dir),
        warning='Public-seed simulation archive; not a joint DP release'),indent=2))

if __name__=='__main__':main()

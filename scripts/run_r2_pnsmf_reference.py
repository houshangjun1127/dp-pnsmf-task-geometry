"""Rebuild only the twelve identified old reference states in a fresh folder."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def commands(data_root,output_dir,states):
    out=[]
    for state in states:
        config=state['config'];rid=state['run_id']
        cmd=[sys.executable,'-m','scripts.run_dp_pnsmf_bgd',
            '--data-dir',str(Path(data_root)/config['data_dir']),
            '--output',str(Path(output_dir)/'runs'/f'{rid}.json'),
            '--factor-output',str(Path(output_dir)/'factors'/state['factor_file_identity']),
            '--torch-threads','4']
        for key,value in config.items():
            if key in ('data_dir','output','factor_output') or value is None:continue
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value:cmd.append(flag)
            else:cmd.extend([flag,str(value)])
        out.append(cmd)
    return out

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--states',type=Path,default=ROOT/'configs/pnsmf_r2_reference_states.json')
    ap.add_argument('--plan-only',action='store_true',help='Print all recorded arguments; do not read data or write output')
    args=ap.parse_args(argv);states=json.loads(args.states.read_text(encoding='utf-8'))['states']
    if len(states)!=12 or len({x['run_id'] for x in states})!=12:raise ValueError('Exactly12 specified references required')
    planned=commands(args.data_root,args.output_dir,states)
    if args.plan_only:
        print(json.dumps(planned,indent=2));return
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name in ('runs','factors'):(args.output_dir/name).mkdir()
    for cmd in planned:subprocess.run(cmd,cwd=ROOT,check=True)
    print('Rebuilt12 all-present research states; apply the frozen stable evaluator to saved factors. This is not a DP archive.')

if __name__=='__main__':main()

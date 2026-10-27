"""Validate matched evaluation settings and summarize saved per-episode benchmarks."""
import argparse
import json
from pathlib import Path
import numpy as np


def report(directory, map_name='MAP01'):
    directory=Path(directory)
    results=[json.loads((directory/f'{map_name}_{policy}.json').read_text())
             for policy in ('random','teacher','quantum','classical')]
    keys=('map','skill','seeds','episodes','max_steps','frame_skip','iwad_sha256')
    settings={k:results[0]['summary'][k] for k in keys}
    for result in results:
        if any(result['summary'][k]!=v for k,v in settings.items()):
            raise ValueError('Benchmark policies have mismatched evaluation settings')
        if [e['seed'] for e in result['episodes']]!=settings['seeds']:
            raise ValueError('Episode seed order differs from benchmark settings')
        if len(result['episodes'])!=settings['episodes']:
            raise ValueError('Incomplete episode results')
    rng=np.random.default_rng(2026)
    differences={}
    random=results[0]['episodes']
    for result in results[1:]:
        policy=result['summary']['policy']; differences[policy]={}
        for metric in ('kills','coverage_cells','items','max_displacement'):
            delta=np.array([a[metric]-b[metric] for a,b in zip(result['episodes'],random)])
            samples=rng.choice(delta,(10000,len(delta)),replace=True).mean(axis=1)
            differences[policy][metric]=dict(paired_mean_difference=float(delta.mean()),
                bootstrap_95_percent_interval=np.quantile(samples,[.025,.975]).tolist())
    runtime=json.loads((directory/'runtime.json').read_text()) if (directory/'runtime.json').exists() else {}
    data=dict(settings=settings,runtime=runtime,results=results,paired_differences_vs_random=differences,
              note='Single-map privileged-state benchmark. Bootstrap intervals reflect these seeds only; no quantum advantage claim. Runtime measured on a shared host with concurrent jobs.')
    (directory/'benchmark.json').write_text(json.dumps(data,indent=2)+'\n')
    print('| Policy | Completions | Kills | Cells | Items | Health | Survival s | Shaped reward | Wall s/ep |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for result in results:
        s=result['summary']
        print(f'| {s["policy"]} | {sum(e["completed"] for e in result["episodes"])}/{s["episodes"]} | '
              f'{s["mean_kills"]:.2f} | {s["mean_coverage_cells"]:.2f} | {s["mean_items"]:.2f} | '
              f'{s["mean_health"]:.2f} | {s["mean_survival_seconds"]:.2f} | '
              f'{s["mean_shaped_reward"]:.2f} | {s["mean_wall_seconds"]:.2f} |')
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/full_game'))
    parser.add_argument('--map',default='MAP01')
    args=parser.parse_args(); report(args.output_dir,args.map)

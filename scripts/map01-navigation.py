"""Reproduce the bounded navigation study inside the existing project container."""
from pathlib import Path
import sys
from corridor_qnn.__main__ import parser
from corridor_qnn.full_cli import evaluate, json_write

ROOT=Path('outputs/map01_navigation')
phase=sys.argv[1]
if phase=='baseline':
    for policy in ('random','teacher','quantum','classical'):
        argv=['full-eval','--policy',policy,'--navigation','legacy','--teacher-version','legacy',
              '--episodes','20','--seed','800','--output-dir',str(ROOT/'baseline'/policy)]
        if policy in ('quantum','classical'): argv+=['--checkpoint',f'outputs/full_game/models/{policy}.pt']
        args=parser().parse_args(argv)
        if (args.output_dir/f'MAP01_{policy}.json').exists(): raise FileExistsError(args.output_dir)
        evaluate(args)
elif phase=='teacher':
    for policy in ('teacher','quantum','classical'):
        args=parser().parse_args(['full-eval','--policy',policy,'--episodes','20','--seed','800',
            '--output-dir',str(ROOT/'improved_navigation'/policy),
            *(['--checkpoint',f'outputs/full_game/models/{policy}.pt'] if policy!='teacher' else [])])
        if (args.output_dir/f'MAP01_{policy}.json').exists(): raise FileExistsError(args.output_dir)
        evaluate(args)
elif phase=='candidate':
    checkpoint='outputs/full_game/models/quantum_map01_dagger.pt'
    for name,flags in (('candidate',[]),('quantum_disabled',['--quantum-disabled'])):
        args=parser().parse_args(['full-eval','--policy','quantum','--episodes','20','--seed','800',
            '--checkpoint',checkpoint,'--output-dir',str(ROOT/name),*flags])
        evaluate(args)
    # Record a known successful seed, then verify recording reproduces a genuine exit.
    import json
    result=json.loads((ROOT/'candidate/MAP01_quantum.json').read_text())
    successes=[row['seed'] for row in result['episodes'] if row['completed']]
    if successes:
        seed=successes[0]
        args=parser().parse_args(['full-eval','--policy','quantum','--episodes','1','--seed',str(seed),
            '--checkpoint',checkpoint,'--output-dir',str(ROOT/'recording'),
            '--record',str(ROOT/f'recording/qnn_exit_{seed}.gif'),
            '--demo',str(ROOT/f'recording/qnn_exit_{seed}.lmp'),
            '--trace-dir',str(ROOT/'recording/traces')])
        recorded=evaluate(args)
        assert recorded['episodes'][0]['completed'],'Recording did not reproduce the exit'
    result=json.loads((ROOT/'improved_navigation/quantum/MAP01_quantum.json').read_text())
    seed=next(row['seed'] for row in result['episodes'] if row['completed'])
    args=parser().parse_args(['full-eval','--policy','quantum','--episodes','1','--seed',str(seed),
        '--checkpoint','outputs/full_game/models/quantum.pt',
        '--output-dir',str(ROOT/'recommended_recording'),
        '--record',str(ROOT/f'recommended_recording/qnn_original_weights_new_navigation_{seed}.gif'),
        '--demo',str(ROOT/f'recommended_recording/qnn_original_weights_new_navigation_{seed}.lmp')])
    assert evaluate(args)['episodes'][0]['completed']
else:
    raise ValueError(phase)

"""Build a checked manifest and measured table; no simulation or training."""
from pathlib import Path
import hashlib
import json
import subprocess

root=Path('outputs/map01_navigation')
if (root/'manifest.json').exists(): raise FileExistsError(root/'manifest.json')
configs=[('Random, original planner','baseline/random/MAP01_random.json'),
         ('Original teacher','baseline/teacher/MAP01_teacher.json'),
         ('Improved teacher','improved_navigation/teacher/MAP01_teacher.json'),
         ('Original QNN, original planner','baseline/quantum/MAP01_quantum.json'),
         ('Original QNN, improved planner','improved_navigation/quantum/MAP01_quantum.json'),
         ('DAgger QNN, improved planner','candidate/MAP01_quantum.json'),
         ('DAgger QNN, quantum disabled','quantum_disabled/MAP01_quantum.json'),
         ('Classical, original planner','baseline/classical/MAP01_classical.json'),
         ('Classical, improved planner','improved_navigation/classical/MAP01_classical.json')]
results={name:json.loads((root/path).read_text()) for name,path in configs}
for name,result in results.items():
    s=result['summary']
    assert s['seeds']==list(range(800,820)) and s['episodes']==20
    assert s['skill']==1 and s['max_steps']==1000 and s['frame_skip']==4 and s['map']=='MAP01'
    assert s['iwad_sha256']=='a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b'
    assert all(e['completed']==(e['outcome']=='completed') for e in result['episodes'])
# Verify reproduction of all historical gameplay metrics, ignoring wall times.
for policy in ('random','teacher','quantum','classical'):
    old=json.loads(Path(f'outputs/full_game/MAP01_{policy}.json').read_text())['episodes']
    new=json.loads((root/f'baseline/{policy}/MAP01_{policy}.json').read_text())['episodes']
    for a,b in zip(old,new):
        for key in ('completed','kills','coverage_cells','items','pickup_events','health','steps'):
            assert a[key]==b[key],(policy,a['seed'],key,a[key],b[key])
lines=['| Policy | Exits | Kills | Cells | Items | Pickup events | Health | Deaths | Wall s (20 eps) |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,result in results.items():
    s=result['summary']; e=result['episodes']
    lines.append(f'| {name} | {sum(x["completed"] for x in e)}/20 | {s["mean_kills"]:.2f} | {s["mean_coverage_cells"]:.2f} | {s["mean_items"]:.2f} | {s["mean_pickup_events"]:.2f} | {s["mean_health"]:.2f} | {sum(x["death"] for x in e)} | {s["total_wall_seconds"]:.2f} |')
table='\n'.join(lines)+'\n'
(root/'benchmark_table.md').write_text(table)
(root/'benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
training=json.loads((root/'dagger/training.json').read_text())
assert training['collected_decisions']==4000
artifacts=[Path('outputs/full_game/models/quantum.pt'),Path('outputs/full_game/models/quantum_map01_dagger.pt'),Path('outputs/full_game/models/classical.pt'),root/'dagger/corrections_100.npz',root/'dagger/aggregate_100.npz']
artifacts+=list((root/'recording').glob('*.gif'))+list((root/'recording').glob('*.lmp'))
artifacts+=list((root/'recommended_recording').glob('*.gif'))+list((root/'recommended_recording').glob('*.lmp'))
source_files=list(Path('corridor_qnn').glob('full_*.py'))+list(Path('scripts').glob('*navigation*'))+[Path('scripts/map01-dagger.py'),Path('scripts/map01-report.py'),Path('run-nixos.sh')]
def identity(path): return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
paired={}
candidate=results['DAgger QNN, improved planner']['episodes']
for name in ('Original QNN, original planner','Original QNN, improved planner','DAgger QNN, quantum disabled'):
    comparison=results[name]['episodes']
    paired[name]=dict(candidate_only_exits=[a['seed'] for a,b in zip(candidate,comparison) if a['completed'] and not b['completed']],
        comparison_only_exits=[a['seed'] for a,b in zip(candidate,comparison) if b['completed'] and not a['completed']],
        mean_kills_delta=sum(a['kills']-b['kills'] for a,b in zip(candidate,comparison))/20)
manifest=dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
    source_worktree=str(Path.cwd()),
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    implementation_iterations=2,additional_training_decisions=4000,finetuning_stages=1,
    evaluation_seeds=list(range(800,820)),collection_seeds=[r['seed'] for r in training['collection']],
    model_selection='four prespecified epochs; no held-out gameplay selection',
    original_benchmarks_reproduced_exactly=True,
    recommended_policy='original checkpoint with improved navigation; candidate fine-tuning did not improve on the planner-only change',
    artifacts={str(p):identity(p) for p in artifacts},sources={str(p):identity(p) for p in source_files},
    paired_comparisons=paired,
    timing_note='Measured wall time under concurrent bounded experiment jobs; not an isolated throughput comparison.',
    quantum_note='Setting quantum_gain=0 removes the quantum residual from action logits. The circuit is still evaluated; this is a gameplay ablation, not a speed benchmark or evidence of quantum advantage.')
assert manifest['branch']=='experiment/map01-navigation'
assert manifest['artifacts']['outputs/full_game/models/quantum.pt']['sha256']=='461d98644bbfd378588212cf4bcfb4fa724946853ea9037d1e342e01eb4a5f0a'
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(table)
print('All original metrics reproduced; settings, budgets, baseline hash and branch verified.')

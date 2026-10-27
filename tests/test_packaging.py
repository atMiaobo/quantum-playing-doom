"""Public checkpoint compatibility and path behavior."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from corridor_qnn.__main__ import parser
from corridor_qnn import training
from corridor_qnn.training import load_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['quantum_skill3_candidate.pt', 'quantum_skill1_baseline.pt'])
def test_pretrained_integrity_and_forward(name):
    path = ROOT / 'pretrained' / name
    manifest = json.loads((ROOT / 'pretrained/manifest.json').read_text())[name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256']
    model = load_model(path, kind='quantum').eval()
    assert (model.qubits, model.layers) == (6, 2)
    with torch.no_grad():
        logits, value = model(torch.zeros((2, 64, 96), dtype=torch.uint8), torch.zeros((2, 20)))
    assert logits.shape == (2, 10) and value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_initializer_preserves_existing_weights(tmp_path):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'pretrained').mkdir()
    shutil.copy2(ROOT / 'scripts/init-checkpoints.py', tmp_path / 'scripts')
    source = ROOT / 'pretrained/quantum_skill3_candidate.pt'
    shutil.copy2(source, tmp_path / 'pretrained')
    command = [sys.executable, str(tmp_path / 'scripts/init-checkpoints.py')]
    subprocess.run(command, cwd=tmp_path.parent, check=True, capture_output=True)
    target = tmp_path / 'outputs/models/quantum.pt'
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b'existing training run')
    subprocess.run(command, cwd=tmp_path.parent, check=True, capture_output=True)
    assert target.read_bytes() == b'existing training run'


def test_imitation_loads_explicit_input_and_saves_separately(tmp_path, monkeypatch):
    source = ROOT / 'pretrained/quantum_skill1_baseline.pt'
    target = tmp_path / 'run.pt'
    args = parser().parse_args(['imitate', '--checkpoint', str(source),
                               '--output-checkpoint', str(target), '--teacher-prob', '0.5'])
    expected = load_model(source).state_dict()
    samples = (None, None, np.zeros(2, dtype=np.int64), [{'won': False}])

    def collect(*unused, model, **kwargs):
        for key, value in expected.items():
            assert torch.equal(model.state_dict()[key], value)
        return samples

    def fit(model, dataset, epochs, batch_size, lr, seed, checkpoint):
        assert dataset is samples and checkpoint == target
        training.save_model(checkpoint, model)
        return 0.5

    before = source.read_bytes()
    monkeypatch.setattr(training, 'collect_demonstrations', collect)
    monkeypatch.setattr(training, 'fit_imitation', fit)
    training.train_imitation(args)
    assert target.exists()
    assert source.read_bytes() == before


def test_explicit_missing_input_does_not_start_random_training(tmp_path):
    missing = str(tmp_path / 'missing.pt')
    for command in ['imitate', 'ppo']:
        args = parser().parse_args([command, '--checkpoint', missing])
        fn = training.train_imitation if command == 'imitate' else training.train_ppo
        with pytest.raises(FileNotFoundError):
            fn(args)


class SyntheticCorridor:
    """One-step episodes to exercise I/O without launching the game engine."""
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def reset(self):
        from types import SimpleNamespace
        return SimpleNamespace(pixels=np.zeros((64, 96), dtype=np.uint8),
                               features=np.zeros(20, dtype=np.float32),
                               frame=np.zeros((240, 320, 3), dtype=np.uint8),
                               context={'kills': 0})

    def step(self, action):
        assert 0 <= action < 10
        return None, 1.0, True, dict(won=False, kills=0, health=100,
                                    episode_raw_reward=1.0, steps=1)


def test_evaluation_recording_and_comparison_with_synthetic_environment(tmp_path, monkeypatch):
    from PIL import Image
    from corridor_qnn import __main__ as cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, 'DeadlyCorridor', SyntheticCorridor)
    checkpoint = ROOT / 'pretrained/quantum_skill3_candidate.pt'
    result = cli.eval_one('quantum', 2, 3, 4, 4, 800,
                          record='outputs/game.gif', checkpoint=checkpoint)
    assert result['episodes'] == 2 and result['mean_raw_reward'] == 1.0
    with Image.open('outputs/game.gif') as gif:
        assert gif.format == 'GIF' and gif.size == (480, 360)
    args = parser().parse_args(['compare', '--episodes', '2',
                               '--quantum-checkpoint', str(checkpoint)])
    cli.compare(args)
    summary = json.loads(Path('outputs/comparison.json').read_text())
    assert [r['policy'] for r in summary] == ['teacher', 'quantum']


def test_ppo_explicit_checkpoint_with_synthetic_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, 'DeadlyCorridor', SyntheticCorridor)
    source = ROOT / 'pretrained/quantum_skill3_candidate.pt'
    before = source.read_bytes()
    target = tmp_path / 'outputs/models/ppo-smoke.pt'
    args = parser().parse_args(['ppo', '--checkpoint', str(source),
                               '--output-checkpoint', str(target), '--steps', '4',
                               '--rollout', '4', '--epochs', '1', '--batch-size', '2'])
    training.train_ppo(args)
    result = torch.load(target, map_location='cpu', weights_only=True)
    assert result['metadata']['training'] == 'PPO'
    assert result['metadata']['steps'] == 4
    assert Path('outputs/ppo_history.csv').exists()
    assert source.read_bytes() == before
    load_model(target, kind='quantum')


def test_launcher_mounts_checkout_and_passes_cli_arguments(tmp_path, monkeypatch):
    import os
    checkout = tmp_path / 'checkout with spaces'
    checkout.mkdir()
    launcher = checkout / 'run-nixos.sh'
    shutil.copy2(ROOT / 'run-nixos.sh', launcher)
    engine = tmp_path / 'podman'
    log = tmp_path / 'engine.jsonl'
    engine.write_text(f'''#!{sys.executable}
import json, os, sys
with open(os.environ['TASK_ENGINE_LOG'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
''')
    engine.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('DOOM_CORRIDOR_ENGINE', 'podman')
    monkeypatch.setenv('TASK_ENGINE_LOG', str(log))
    subprocess.run([str(launcher), 'eval', '--checkpoint',
                    'pretrained/quantum_skill3_candidate.pt'], cwd=tmp_path, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    run = calls[-1]
    assert calls[:2] == [['info'], ['image', 'inspect', 'localhost/doom-corridor-qnn:latest']]
    assert run[0:2] == ['run', '--rm']
    assert run[run.index('-v') + 1] == f'{checkout}:/app'
    assert '--userns=keep-id' in run
    assert run[-3:] == ['eval', '--checkpoint', 'pretrained/quantum_skill3_candidate.pt']


@pytest.mark.parametrize('equals',[False,True])
def test_launcher_external_iwad_is_single_readonly_file(tmp_path,monkeypatch,equals):
    import os
    checkout=tmp_path/'checkout'; checkout.mkdir()
    launcher=checkout/'run-nixos.sh'; shutil.copy2(ROOT/'run-nixos.sh',launcher)
    wad=tmp_path/'licensed game.wad'; wad.write_bytes(b'IWAD')
    engine=tmp_path/'podman'; log=tmp_path/'engine.jsonl'
    engine.write_text(f'''#!{sys.executable}
import json, os, sys
with open(os.environ['TASK_ENGINE_LOG'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
''')
    engine.chmod(0o755)
    monkeypatch.setenv('PATH',str(tmp_path)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('DOOM_CORRIDOR_ENGINE','podman')
    monkeypatch.setenv('TASK_ENGINE_LOG',str(log))
    flag=['--iwad='+str(wad)] if equals else ['--iwad',str(wad)]
    subprocess.run([str(launcher),'full-doctor',*flag],check=True)
    run=json.loads(log.read_text().splitlines()[-1])
    assert f'{wad}:/tmp/full-game-external.wad:ro' in run
    assert str(tmp_path)+':/tmp' not in run
    assert ('--iwad=/tmp/full-game-external.wad' in run if equals else run[-1]=='/tmp/full-game-external.wad')

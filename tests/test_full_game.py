"""Full-game contract tests; engine checks are in scripts/full-integration.py."""
import json
import struct
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from corridor_qnn.full_actions import ACTIONS, BUTTONS, action_vectors
from corridor_qnn.full_environment import FEATURE_DIM, FEATURE_NAMES, terminal_status, shape_reward
from corridor_qnn.full_navigation import map_name, read_map, Navigator
from corridor_qnn.full_models import FullActorCritic, save_full, load_full, transfer_corridor
from corridor_qnn.full_teacher import teacher_action
from corridor_qnn.__main__ import parser


def write_wad(path, lumps):
    body=b''; directory=b''
    for name, data in lumps:
        directory+=struct.pack('<ii8s',12+len(body),len(data),name.encode())
        body+=data
    path.write_bytes(struct.pack('<4sii',b'PWAD',len(lumps),12+len(body))+body+directory)
    return path


@pytest.fixture
def small_wad(tmp_path):
    vertices=[(-256,-256),(256,-256),(256,256),(-256,256)]
    return write_wad(tmp_path/'test.wad',[
        ('MAP01',b''),('THINGS',b''),
        ('LINEDEFS',b''.join(struct.pack('<7H',i,(i+1)%4,1,11 if i==0 else 0,0,0,65535) for i in range(4))),
        ('SIDEDEFS',struct.pack('<hh8s8s8sH',0,0,b'-',b'-',b'STONE',0)),
        ('VERTEXES',b''.join(struct.pack('<hh',*v) for v in vertices)),
        ('SECTORS',struct.pack('<hh8s8shhh',0,128,b'FLOOR',b'CEIL',160,0,0)),
    ])


def test_full_actions_order_and_combinations():
    buttons=list(reversed(BUTTONS)); vectors=action_vectors(buttons)
    assert len(vectors)==21 and len(set(ACTIONS))==21
    for macro, vector in zip(ACTIONS,vectors):
        assert {b for b,v in zip(buttons,vector) if v}==set(macro)
    assert ('USE',) in ACTIONS and ('MOVE_FORWARD','USE') in ACTIONS
    assert any('SELECT_NEXT_WEAPON' in a for a in ACTIONS)
    with pytest.raises(ValueError): action_vectors(['ATTACK'])


@pytest.mark.parametrize('name', ['MAP01','MAP02','MAP30'])
def test_maps(name):
    assert map_name(name.lower())==name


def test_secret_maps_explicit():
    for name in ('MAP00','MAP31','E1M1','MAP1','MAP33'):
        with pytest.raises(ValueError): map_name(name)
    assert map_name('MAP32',True)=='MAP32'


@pytest.mark.parametrize('inputs,expected',[
    ((False,False,False,False,False),'running'),
    ((True,True,False,False,False),'death'),
    ((True,False,True,False,False),'timeout'),
    ((False,False,False,True,False),'timeout'),
    ((True,False,False,False,True),'completed'),
    ((True,False,False,False,False),'unknown_terminal'),
    ((False,False,False,False,True),'running'),
    ((True,True,False,False,True),'death'),
])
def test_terminal_state(inputs,expected):
    assert terminal_status(*inputs)==expected


def test_reward_deltas():
    before=dict(KILLCOUNT=2,ITEMCOUNT=3,HEALTH=80,ARMOR=0,HITCOUNT=3)
    after=dict(KILLCOUNT=3,ITEMCOUNT=4,HEALTH=70,ARMOR=10,HITCOUNT=5)
    assert shape_reward(before,after,1,'running')==pytest.approx(3.545)
    assert shape_reward(before,after,1,'completed')==pytest.approx(53.545)
    assert shape_reward(before,before,0,'running')==-.005


def test_geometry_exit_is_read_from_wad(small_wad):
    verts,lines,_,_=read_map(small_wad,'MAP01')
    assert len(verts)==4 and lines[0][3]==11
    nav=Navigator(small_wad,'MAP01')
    assert np.allclose(nav.exits[0],[0,-256])
    target,interaction=nav.update(np.array([0.,0.]))
    assert np.isfinite(target).all() and interaction==256
    assert nav.reachable>1
    assert target[1]<0
    nav.reset(); assert not nav.path and not nav.visits.any()
    with pytest.raises(ValueError): read_map(small_wad,'MAP02')


def test_full_observation_contract():
    assert len(set(FEATURE_NAMES))==FEATURE_DIM==54
    f=np.zeros(FEATURE_DIM,dtype=np.float32); f[8]=1; f[11]=4
    assert teacher_action(f)==0
    f[9]=1; assert teacher_action(f)==2
    f[9]=-1; assert teacher_action(f)==3
    f[15]=1; f[1]=1; f[12]=1; f[13]=.3
    assert teacher_action(f)==8
    f[13]=-.3; assert teacher_action(f)==9


@pytest.mark.parametrize('kind',['quantum','classical'])
def test_full_model_checkpoint_gradients_and_transfer(tmp_path,kind):
    model=FullActorCritic(kind)
    x=torch.zeros(2,64,96,dtype=torch.uint8); f=torch.randn(2,FEATURE_DIM)
    logits,values=model(x,f)
    assert logits.shape==(2,21) and values.shape==(2,)
    torch.nn.functional.cross_entropy(logits,torch.tensor([0,2])).backward()
    if kind=='quantum':
        assert model.core.weights.grad is not None
        assert model.core.weights.grad.abs().sum()>0
    path=tmp_path/'full.pt'; save_full(path,model,{'test':True})
    other=load_full(path,kind)
    assert torch.equal(other(x,f)[0],logits)
    with pytest.raises(ValueError): load_full(path,'classical' if kind=='quantum' else 'quantum')
    legacy=__import__('pathlib').Path(__file__).resolve().parents[1]/'pretrained/quantum_skill1_baseline.pt'
    with pytest.raises(ValueError): load_full(legacy)
    if kind=='quantum':
        fusion=model.fusion[0].weight.detach().clone()
        keys=transfer_corridor(model,legacy)
        assert 'core.weights' in keys and 'vision.net.0.weight' in keys
        assert 'actor.weight' not in keys and 'fusion.0.weight' not in keys
        assert torch.equal(fusion,model.fusion[0].weight)


def test_full_cli_and_legacy_coexist():
    for cmd in ('doctor','eval','imitate','ppo','compare','full-doctor','full-eval','full-train','full-campaign'):
        args=parser().parse_args([cmd]); assert args.cmd==cmd
    args=parser().parse_args(['full-campaign','--start-map','MAP02','--end-map','MAP03'])
    assert args.start_map=='MAP02'


def test_full_eval_records_policy_actions(tmp_path,monkeypatch):
    from corridor_qnn import full_cli
    from PIL import Image
    calls=[]
    class Environment:
        iwad='fake'; iwad_hash='synthetic'
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def reset(self,seed=None,demo=None):
            calls.append(('seed',seed))
            return SimpleNamespace(frame=np.zeros((240,320,3),dtype=np.uint8))
        def step(self,action):
            calls.append(('action',action))
            return None,0,True,dict(completed=False,death=False,kills=0,survival_seconds=1,
                coverage_cells=1,items=0,pickup_events=0,health=100,raw_reward=0,
                shaped_reward=0,steps=1,max_displacement=0)
    monkeypatch.setattr(full_cli,'environment',lambda args:Environment())
    monkeypatch.setattr(full_cli,'infer',lambda model,obs,deterministic:7)
    args=parser().parse_args(['full-eval','--episodes','2','--record',str(tmp_path/'qnn.gif'),
                             '--output-dir',str(tmp_path)])
    result=full_cli.evaluate(args,model=FullActorCritic())
    assert calls==[('seed',800),('action',7),('seed',801),('action',7)]
    assert result['summary']['completion_rate']==0
    assert json.loads((tmp_path/'MAP01_quantum.json').read_text())['summary']['seeds']==[800,801]
    with Image.open(tmp_path/'qnn.gif') as image: assert image.format=='GIF'


def synthetic_step_environment(before,after,raw=0,finished=False,dead=False,engine_timeout=False):
    from corridor_qnn.full_environment import FullDoom
    env=FullDoom.__new__(FullDoom)
    env.done=False; env.v=before; env.steps=0; env.max_steps=100
    env.actions=action_vectors(list(BUTTONS)); env.frame_skip=4
    env.map='MAP01'; env.skill=1; env.start=np.array([0.,0.])
    env.peak_distance=0.; env.visited={(0,0)}; env.pickups=0
    env.reward=0.; env.native_reward=0.
    ticks=iter([140,144,148,152])
    env.game=SimpleNamespace(make_action=lambda *args:raw,is_episode_finished=lambda:finished,
        is_player_dead=lambda:dead,is_episode_timeout_reached=lambda:engine_timeout,
        get_episode_time=lambda:next(ticks) if not finished else 0)
    env.variables=lambda:after
    env.observe=lambda:None
    return env


def full_variables():
    from corridor_qnn.full_environment import VARIABLES
    v=dict.fromkeys(VARIABLES,0.)
    v.update(HEALTH=70.,KILLCOUNT=5.,ITEMCOUNT=3.,SELECTED_WEAPON=2.,SELECTED_WEAPON_AMMO=40.)
    return v


def test_exit_preserves_preterminal_counters():
    before=full_variables(); reset=full_variables()
    reset.update(HEALTH=100.,KILLCOUNT=0.,ITEMCOUNT=0.)
    env=synthetic_step_environment(before,reset,raw=1,finished=True)
    _,_,done,info=env.step(12)
    assert done and info['completed'] and not info['death']
    assert info['kills']==5 and info['items']==3 and info['health']==70
    assert env.pickups==0
    with pytest.raises(RuntimeError): env.step(0)


def test_weapon_switch_does_not_count_as_pickup():
    before=full_variables(); after=dict(before)
    after.update(SELECTED_WEAPON=3.,SELECTED_WEAPON_AMMO=80.)
    env=synthetic_step_environment(before,after)
    _,_,done,info=env.step(17)
    assert not done and info['pickup_events']==0
    after=dict(after); after['AMMO2']=10
    env.variables=lambda:after
    _,_,_,info=env.step(0)
    assert info['pickup_events']==1


def test_campaign_progresses_only_after_real_completion(tmp_path,monkeypatch):
    from corridor_qnn import full_cli
    visited=[]
    def evaluate(args):
        visited.append(args.map)
        return dict(summary=dict(completion_rate=1. if args.map=='MAP01' else 0.))
    monkeypatch.setattr(full_cli,'evaluate',evaluate)
    args=parser().parse_args(['full-campaign','--policy','teacher','--start-map','MAP01',
                             '--end-map','MAP03','--output-dir',str(tmp_path)])
    full_cli.campaign(args)
    assert visited==['MAP01','MAP02']
    result=json.loads((tmp_path/'campaign.json').read_text())
    assert result['mode']=='independent_maps_fresh_inventory'


def test_nonadvancing_engine_cannot_produce_benchmark_metrics():
    before=full_variables()
    env=synthetic_step_environment(before,dict(before))
    env.game.get_episode_time=lambda:140
    with pytest.raises(RuntimeError,match='failed to advance'):
        env.step(0)

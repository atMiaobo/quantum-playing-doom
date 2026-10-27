from types import SimpleNamespace
import numpy as np
import pytest
import torch
from corridor_qnn.actions import ACTION_SETS, action_vectors
from corridor_qnn.features import (FEATURE_DIM, PIXEL_H, PIXEL_W, image_to_gray,
                                   compose_observation, bearing_to, wrap_pi)
from corridor_qnn.teacher import teacher_action
from corridor_qnn.models import ActorCritic, pack
from corridor_qnn.training import gae_advantages, save_model, load_model, fit_imitation
from corridor_qnn.quantum import VariationalQuantumCircuit, ry, rz


def var():
    return dict(POSITION_X=0,POSITION_Y=0,ANGLE=0,HEALTH=100,
                SELECTED_WEAPON_AMMO=50,KILLCOUNT=0)


def obj(name='ShotgunGuy',category='Monster', x=160.,y=70.,id=7):
    return SimpleNamespace(name=name,category=category,position_x=x,position_y=y,id=id)


def observe(objs=None,labels=None):
    return compose_observation(frame=np.zeros((240,320,3),dtype=np.uint8),
         variables=var(),objects=objs or [],labels=labels or [],
         start_x=0,start_y=0,last_health=100,previous_reward=0,
         steps=0,max_steps=450)


def test_actions_are_combinations():
    b=['MOVE_LEFT','MOVE_RIGHT','ATTACK','MOVE_FORWARD','MOVE_BACKWARD','TURN_LEFT','TURN_RIGHT']
    v=action_vectors(b)
    assert len(v)==10
    assert v[1]==[0,0,1,1,0,0,0]
    assert v[2]==[0,0,1,0,0,1,0]
    with pytest.raises(ValueError):action_vectors(['ATTACK'])


def test_image_resize_and_format():
    b=np.ones((240,320,3),dtype=np.uint8)*120
    assert image_to_gray(b).shape==(PIXEL_H, PIXEL_W)
    assert image_to_gray(b.transpose(2,0,1)).shape==(PIXEL_H,PIXEL_W)
    assert abs(float(image_to_gray(b).mean())-120)<2


def test_bearing_and_wrap():
    b, d=bearing_to(0,0,0,0,100)
    assert abs(b-np.pi/2)<1e-5
    assert d==100
    assert -np.pi<=wrap_pi(6)<np.pi


def test_features_objects_and_labels():
    monster=obj()
    label=SimpleNamespace(object_id=7,object_category='Monster',object_name='ShotgunGuy')
    obs=observe([monster,obj(name='GreenArmor',category='Armor',x=900,y=0,id=10)],[label])
    assert obs.pixels.shape == (64,96)
    assert obs.features.shape == (FEATURE_DIM,)
    assert obs.context['enemy_visible']
    assert obs.context['goal_visible_in_objects']
    assert obs.context['enemy_count']==1
    assert obs.context['goal_distance']==900
    assert obs.features[13]==1


def test_missing_objects_graceful():
    obs=observe()
    assert np.isfinite(obs.features).all()
    assert not obs.context['goal_visible_in_objects']
    assert obs.features[14]==0
    assert teacher_action(obs.features)==0


def test_teacher_tracks_target_left_and_right():
    f=observe([obj()], [SimpleNamespace(object_id=7,object_category='Monster')]).features
    assert teacher_action(f)==2 # target to left relative to yaw
    f[11]=-f[11]
    assert teacher_action(f)==3


def test_teacher_goes_to_goal():
    o=observe([obj(name='GreenArmor',category='Armor',x=20,y=200,id=55)])
    assert teacher_action(o.features)==8


def test_quantum_state_normalized():
    m=VariationalQuantumCircuit(qubits=6,layers=2)
    x=torch.randn(5,6,requires_grad=True)
    z=m(x)
    assert z.shape==(5,6)
    assert torch.isfinite(z).all()
    assert z.abs().max() <= 1.001
    z.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m.weights.grad is not None


@pytest.mark.parametrize('kind',('quantum','classical'))
def test_actor_critic_shapes_and_gradients(kind):
    m=ActorCritic(kind=kind,qubits=4,layers=1)
    img=torch.randint(0,255,(3,64,96),dtype=torch.uint8)
    f=torch.randn(3,FEATURE_DIM)
    logits,value=m(img,f)
    assert logits.shape==(3,10) and value.shape==(3,)
    loss=logits.square().sum()+value.square().sum()
    loss.backward()
    if kind=='quantum':assert m.core.weights.grad is not None
    assert torch.isfinite(logits).all()


def test_batch_pack():
    obs=observe()
    a,b=pack(obs)
    assert a.shape==(1,64,96) and b.shape==(1,20)
    a,b=pack([obs,obs])
    assert a.shape[0]==2 and b.shape[0]==2


def test_checkpoint_roundtrip(tmp_path):
    m=ActorCritic('quantum',qubits=4,layers=1)
    p=tmp_path/'model.pt'
    save_model(p,m,{'source':'unit-test'})
    loaded=load_model(p,kind='quantum')
    for a,b in zip(m.parameters(),loaded.parameters()):
        assert torch.allclose(a,b)
    with pytest.raises(ValueError): load_model(p,kind='classical')


def test_gaes_reset_at_done():
    a,r=gae_advantages([1,1,1],[0,0,0],[False,True,False],next_value=2,gamma=1,lam=1)
    assert np.allclose(a,[2,1,3])
    assert np.allclose(r,a)


def test_fit_small_synthetic_dataset(tmp_path):
    torch.manual_seed(42)
    m=ActorCritic('quantum',qubits=2,layers=1)
    n=18
    images=np.random.default_rng(3).integers(0,255,(n,64,96),dtype=np.uint8)
    features=np.random.default_rng(4).normal(size=(n,FEATURE_DIM)).astype(np.float32)
    labels=np.random.default_rng(5).integers(0,10,n)
    result=fit_imitation(m,(images,features,labels,[]),epochs=1,batch_size=6,
                         checkpoint=tmp_path/'trained.pt')
    assert 0<=result<=1
    assert (tmp_path/'trained.pt').exists()

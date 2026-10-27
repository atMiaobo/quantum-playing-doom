"""CNN + 6-qubit VQC + classical residual actor-critic; pure CNN ablation."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.distributions import Categorical
from .quantum import VariationalQuantumCircuit
from .features import FEATURE_DIM
from .actions import NUM_ACTIONS


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=3), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 4 * 7, 48), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.float().unsqueeze(1)/255.0)


class ActorCritic(nn.Module):
    def __init__(self, kind="quantum", qubits=6, layers=2):
        super().__init__()
        if kind not in ("quantum", "classical"):
            raise ValueError("kind must be quantum or classical")
        self.kind, self.qubits, self.layers = kind, qubits, layers
        self.vision = VisionEncoder()
        self.fusion = nn.Sequential(nn.Linear(48+FEATURE_DIM, 64), nn.Tanh(),
                                    nn.Linear(64, 48), nn.Tanh())
        self.angles = nn.Sequential(nn.Linear(48, qubits), nn.Tanh())
        if kind == "quantum":
            self.core = VariationalQuantumCircuit(qubits, layers)
        else:
            self.core = nn.Sequential(nn.Linear(qubits, 32), nn.Tanh(),
                                      nn.Linear(32, qubits), nn.Tanh())
        # A classical residual makes training stable and gives quantum branch a
        # meaningful learnable contribution, without bottlenecking all decisions.
        self.actor = nn.Linear(qubits, NUM_ACTIONS)
        self.classical_actor = nn.Linear(48, NUM_ACTIONS)
        self.quantum_gain = nn.Parameter(torch.tensor(0.7))
        self.critic = nn.Sequential(nn.Linear(48+qubits, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, pixels: torch.Tensor, features: torch.Tensor):
        encoded = self.vision(pixels)
        fused = self.fusion(torch.cat([encoded, features.float()], dim=-1))
        a = self.angles(fused)
        measured = self.core(a*math.pi) if self.kind == "quantum" else self.core(a)
        logits = self.classical_actor(fused) + self.quantum_gain * self.actor(measured)
        value = self.critic(torch.cat([fused, measured], dim=-1)).squeeze(-1)
        return logits, value

    def act(self, pixels, features, deterministic=False):
        logits, value = self(pixels, features)
        dist = Categorical(logits=logits)
        actions = logits.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), value

    def params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pack(observations, device="cpu"):
    import numpy as np
    if not isinstance(observations, (tuple, list)):
        observations = [observations]
    images = torch.as_tensor(np.stack([o.pixels for o in observations]), device=device)
    features = torch.as_tensor(np.stack([o.features for o in observations]), device=device)
    return images, features

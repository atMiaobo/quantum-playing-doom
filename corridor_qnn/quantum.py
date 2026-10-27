"""Differentiable *actual quantum gate* state-vector simulator written with PyTorch.

This uses complex amplitudes, unitary RY/RZ and CNOT gates, and Pauli-Z
measurements. No quantum hardware or external quantum library is required.
"""
from __future__ import annotations


import torch
from torch import nn


def ry(angles: torch.Tensor) -> torch.Tensor:
    """Batched real-valued RY rotations, output (batch, 2, 2)."""
    c = torch.cos(angles / 2)
    s = torch.sin(angles / 2)
    return torch.stack((torch.stack((c, -s), -1),
                        torch.stack((s, c), -1)), -2).to(torch.complex64)


def rz(angles: torch.Tensor) -> torch.Tensor:
    """Batched RZ rotations, output (batch, 2, 2)."""
    phase = angles.to(torch.complex64) * (0.5j)
    a = torch.exp(-phase)
    b = torch.exp(phase)
    zeros = torch.zeros_like(a)
    return torch.stack((torch.stack((a, zeros), -1),
                        torch.stack((zeros, b), -1)), -2)


def apply_single_qubit(state: torch.Tensor, gates: torch.Tensor,
                       wire: int, qubits: int) -> torch.Tensor:
    """Apply B independent 2x2 unitary gates to a B x 2**qubits state."""
    batch = state.shape[0]
    tensor = state.reshape(batch, *([2] * qubits)).movedim(wire + 1, -1)
    tensor = tensor.reshape(batch, -1, 2)
    tensor = torch.matmul(tensor, gates.transpose(-2, -1))
    tensor = tensor.reshape(batch, *([2] * (qubits - 1)), 2)
    return tensor.movedim(-1, wire + 1).reshape(batch, 2**qubits)


def cnot_permutation(qubits: int, control: int, target: int) -> torch.Tensor:
    if control == target:
        raise ValueError("CNOT requires different control and target")
    indices = torch.arange(2**qubits)
    c_bit = 1 << (qubits - control - 1)
    t_bit = 1 << (qubits - target - 1)
    return indices ^ (((indices & c_bit) != 0).long() * t_bit)


class VariationalQuantumCircuit(nn.Module):
    """Angle embedding + trainable RY/RZ + CNOT ring + <Z> measurements.

    Variational parameters: (layers, qubits, 2), one RY and one RZ per layer
    per qubit. This is an ideal noise-free state-vector simulation.
    """

    def __init__(self, qubits: int = 4, layers: int = 2):
        super().__init__()
        if not 2 <= qubits <= 8:
            raise ValueError("Supported range: 2 to 8 qubits")
        if layers < 1:
            raise ValueError("At least one layer is needed")
        self.qubits = qubits
        self.layers = layers
        self.weights = nn.Parameter(0.15 * torch.randn(layers, qubits, 2))
        for control in range(qubits):
            self.register_buffer(
                f"cnot_{control}",
                cnot_permutation(qubits, control, (control + 1) % qubits),
                persistent=False,
            )
        basis = torch.arange(2**qubits)
        for wire in range(qubits):
            signs = 1 - 2 * ((basis >> (qubits - wire - 1)) & 1)
            self.register_buffer(f"z_signs_{wire}", signs.float(), persistent=False)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.ndim != 2 or angles.shape[1] != self.qubits:
            raise ValueError(f"Expected (batch, {self.qubits}) angles, got {tuple(angles.shape)}")
        batch = angles.shape[0]
        state = torch.zeros(batch, 2**self.qubits,
                            dtype=torch.complex64, device=angles.device)
        state[:, 0] = 1.0 + 0.0j  # |000...0>

        for wire in range(self.qubits):
            state = apply_single_qubit(state, ry(angles[:, wire]), wire, self.qubits)
        for layer in range(self.layers):
            for wire in range(self.qubits):
                y = self.weights[layer, wire, 0].expand(batch)
                z = self.weights[layer, wire, 1].expand(batch)
                state = apply_single_qubit(state, ry(y), wire, self.qubits)
                state = apply_single_qubit(state, rz(z), wire, self.qubits)
            # The last layer has entanglement too: a circular CNOT ladder.
            for control in range(self.qubits):
                state = state[:, getattr(self, f"cnot_{control}")]
        probabilities = state.abs().square()
        return torch.stack(
            [probabilities @ getattr(self, f"z_signs_{i}")
             for i in range(self.qubits)], dim=-1
        )

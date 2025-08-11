from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from graph_model import GraphModel, NeuronModel, ConnectionModel


@dataclass
class SimulationConfig:
    dt_ms: float = 1.0  # Euler step in ms
    input_current: float = 10.0  # baseline input (pA arbitrary)
    synaptic_scale: float = 5.0  # scale for weighted inputs


class IzhikevichSimulator:
    def __init__(self, model: GraphModel, config: SimulationConfig | None = None) -> None:
        self.model = model
        self.config = config or SimulationConfig()
        self.step_count: int = 0

    def advance_steps(self, steps: int) -> None:
        for _ in range(max(0, steps)):
            self._step_euler()
            self.step_count += 1

    def _step_euler(self) -> None:
        dt = self.config.dt_ms

        # Compute synaptic input for each neuron from spikes in previous step
        incoming: Dict[int, float] = {n.id: 0.0 for n in self.model.neurons}
        for c in self.model.connections:
            src = self.model.find_neuron(c.source_id)
            if src is None:
                continue
            if src.spiked:
                incoming[c.target_id] = incoming.get(c.target_id, 0.0) + c.weight * self.config.synaptic_scale

        # Update all neurons using Euler method
        for n in self.model.neurons:
            I = self.config.input_current + incoming.get(n.id, 0.0)

            # Izhikevich model differential equations (ms scale)
            # dv/dt = 0.04 v^2 + 5 v + 140 - u + I
            # du/dt = a (b v - u)
            dv = 0.04 * n.v * n.v + 5.0 * n.v + 140.0 - n.u + I
            du = n.a * (n.b * n.v - n.u)

            # Euler update
            n.v += dt * dv
            n.u += dt * du

            # Spike condition
            if n.v >= 30.0:
                n.v = n.c
                n.u += n.d
                n.spiked = True
            else:
                n.spiked = False



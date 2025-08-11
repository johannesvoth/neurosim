from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from graph_model import GraphModel, NeuronModel, ConnectionModel


@dataclass
class SimulationConfig:
    dt_ms: float = 1.0  # Euler step in ms
    input_current: float = 1.0  # baseline input (pA arbitrary)
    synaptic_scale: float = 20.0  # scale for weighted inputs


class IzhikevichSimulator:
    def __init__(self, model: GraphModel, config: SimulationConfig | None = None) -> None:
        self.model = model
        self.config = config or SimulationConfig()
        self.step_count: int = 0
        # One-step per-neuron current injections to be applied on the next step
        self._pending_injection: Dict[int, float] = {}

    def advance_steps(self, steps: int) -> None:
        for _ in range(max(0, steps)):
            self._step_euler()
            self.step_count += 1

    def inject_current(self, neuron_id: int, amount: float) -> None:
        """Queue a one-step current injection for the given neuron (applied on next step)."""
        self._pending_injection[neuron_id] = self._pending_injection.get(neuron_id, 0.0) + amount

    def _step_euler(self) -> None:
        dt = self.config.dt_ms

        # Compute synaptic input for each neuron from spikes in previous step
        incoming: Dict[int, float] = {n.id: 0.0 for n in self.model.neurons} # make a dict of all neurons with 0.0 as the value 
        for c in self.model.connections: # for each connection
            src = self.model.find_neuron(c.source_id) # find the source neuron (only one neuron ever)
            #if src is None: 
                #continue # commented out because I think it might be redundant, not sure yet.
            if src.spiked: # if the source neuron spiked bool is set
                incoming[c.target_id] = incoming.get(c.target_id, 0.0) + c.weight * self.config.synaptic_scale 
                # dict stores a list of neurons and their incoming synaptic input.
                # incoming.get(c.target_id) is the synaptic input for the target neuron.
                # we do this so if a neuron has multiple other neurons connection from it (multiple sources/connections that result in a spike) we can add them all up.
                # So usually it will be 0, and we simply do the weight * synaptic_scale for a spike, makes sense.
                # we end up with a list of neurons and their incoming synaptic input.

        # Update all neurons using Euler method
        for n in self.model.neurons: # for each neuron. One run of this function loops through all neurons.
            # I is global baseline + synaptic + any one-step injection queued
            I = (
                self.config.input_current
                + incoming.get(n.id, 0.0)
                + self._pending_injection.get(n.id, 0.0)
                + n.i_baseline
            ) # I is a global input current to all neurons. # for a specifc neuron we add  the synaptic input. Again, usually 0.

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
                n.spiked = True # even if we noticed that a neuron spikes in this time step, the calculation the other neurons is still done independently just like this neuron.
                # the spike information is only put into consideration for the next time we run the euler step and 
                # resets naturally. So a spike could technically be ongoing for a few steps if we make it that way. But naturally it should only last a single time step.
            else:
                n.spiked = False

        # Clear consumed injections after applying a single step
        self._pending_injection.clear()



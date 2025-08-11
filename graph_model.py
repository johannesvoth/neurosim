from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NeuronModel:
    id: int
    x: float
    y: float
    selected: bool = False
    # Izhikevich parameters and state
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    v: float = -65.0
    u: float = -13.0  # typically b*v
    spiked: bool = False


@dataclass
class ConnectionModel:
    source_id: int
    target_id: int
    weight: float = 0.0
    selected: bool = False


@dataclass
class GraphModel:
    neurons: List[NeuronModel] = field(default_factory=list)
    connections: List[ConnectionModel] = field(default_factory=list)
    _next_neuron_id: int = 1

    def add_neuron(self, x: float, y: float) -> NeuronModel:
        neuron = NeuronModel(id=self._next_neuron_id, x=x, y=y)
        self._next_neuron_id += 1
        self.neurons.append(neuron)
        return neuron

    def find_neuron(self, neuron_id: int) -> Optional[NeuronModel]:
        for n in self.neurons:
            if n.id == neuron_id:
                return n
        return None

    def add_connection(self, source_id: int, target_id: int) -> Tuple[bool, str]:
        if source_id == target_id:
            return False, "no connection"
        for c in self.connections:
            if c.source_id == source_id and c.target_id == target_id:
                return False, "connection already exists"
        self.connections.append(ConnectionModel(source_id=source_id, target_id=target_id, weight=0.0))
        return True, "connection made"

    def delete_selected(self) -> None:
        selected_neuron_ids = {n.id for n in self.neurons if n.selected}
        if selected_neuron_ids:
            self.neurons = [n for n in self.neurons if n.id not in selected_neuron_ids]
            self.connections = [
                c for c in self.connections if c.source_id not in selected_neuron_ids and c.target_id not in selected_neuron_ids
            ]
            return
        # delete selected connection if any
        self.connections = [c for c in self.connections if not c.selected]

    def clear_selection(self) -> None:
        for n in self.neurons:
            n.selected = False
        for c in self.connections:
            c.selected = False



from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, ClassVar, Set


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
    type_name: str = "Custom"
    i_baseline: float = 0.0


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
    _edge_set: Set[Tuple[int, int]] = field(default_factory=set)

    def add_neuron(self, x: float, y: float) -> NeuronModel:
        neuron = NeuronModel(id=self._next_neuron_id, x=x, y=y)
        self._next_neuron_id += 1
        self.neurons.append(neuron)
        return neuron

    # Common Izhikevich neuron type presets (class-level constant)
    NEURON_PRESETS: ClassVar[Dict[str, Tuple[float, float, float, float]]] = {
        # name: (a, b, c, d)
        "RS": (0.02, 0.20, -65.0, 8.0),   # Regular spiking
        "FS": (0.10, 0.20, -65.0, 2.0),   # Fast spiking
        "IB": (0.02, 0.20, -55.0, 4.0),   # Intrinsically bursting
        "CH": (0.02, 0.20, -50.0, 2.0),   # Chattering
        "LTS": (0.02, 0.25, -65.0, 2.0),  # Low-threshold spiking
        "I": (0.02, 0.20, -65.0, 8.0),    # Tonic spiking base; uses baseline current
    }

    def add_neuron_of_type(self, type_name: str, x: float, y: float) -> NeuronModel:
        a, b, c, d = self.NEURON_PRESETS.get(type_name, (0.02, 0.20, -65.0, 8.0))
        v0 = c
        u0 = b * v0
        i_baseline = 10.0 if type_name == "I" else 0.0
        neuron = NeuronModel(
            id=self._next_neuron_id,
            x=x,
            y=y,
            selected=False,
            a=a,
            b=b,
            c=c,
            d=d,
            v=v0,
            u=u0,
            spiked=False,
            type_name=type_name,
            i_baseline=i_baseline,
        )
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
        if (source_id, target_id) in self._edge_set:
            return False, "connection already exists"
        self.connections.append(ConnectionModel(source_id=source_id, target_id=target_id, weight=0.0))
        self._edge_set.add((source_id, target_id))
        return True, "connection made"

    def add_connection_fast(self, source_id: int, target_id: int, weight: float = 0.0) -> bool:
        """Fast path without duplicate scan. Returns True if added, False if skipped.
        Skips self-loop and duplicates using an internal edge set.
        """
        if source_id == target_id:
            return False
        if (source_id, target_id) in self._edge_set:
            return False
        self.connections.append(ConnectionModel(source_id=source_id, target_id=target_id, weight=weight))
        self._edge_set.add((source_id, target_id))
        return True

    def delete_selected(self) -> None:
        selected_neuron_ids = {n.id for n in self.neurons if n.selected}
        if selected_neuron_ids:
            self.neurons = [n for n in self.neurons if n.id not in selected_neuron_ids]
            self.connections = [
                c for c in self.connections if c.source_id not in selected_neuron_ids and c.target_id not in selected_neuron_ids
            ]
            # Rebuild edge set after deletions
            self._edge_set = {(c.source_id, c.target_id) for c in self.connections}
            return
        # delete selected connection if any
        removed = {(c.source_id, c.target_id) for c in self.connections if c.selected}
        self.connections = [c for c in self.connections if not c.selected]
        if removed:
            self._edge_set.difference_update(removed)

    def clear_selection(self) -> None:
        for n in self.neurons:
            n.selected = False
        for c in self.connections:
            c.selected = False



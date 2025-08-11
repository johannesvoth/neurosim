from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

from graph_model import GraphModel, NeuronModel
from simulator import IzhikevichSimulator


Number = Union[int, float]
PixelGrid = Sequence[Sequence[Union[Number, bool]]]
Point = Tuple[float, float]
LabeledPoint = Tuple[float, float, int]


# --- Column inputs from label lists ---

def generate_binary_labels(num_points: int, p_one: float = 0.5, seed: int | None = None) -> List[int]:
    import random
    if seed is not None:
        random.seed(seed)
    return [1 if random.random() < p_one else 0 for _ in range(max(0, num_points))]


def create_input_column_from_labels(
    model: GraphModel,
    labels: Sequence[int],
    origin: Tuple[float, float] = (260.0, 60.0),
    spacing: float = 7.0,
    type_for_label1: str = "I",
    type_for_label0: str = "FS",
) -> List[int]:
    """
    Create a vertical column of input neurons, one per label in `labels`.
    label 1 -> `type_for_label1` (default I), label 0 -> `type_for_label0` (default FS).
    Returns the neuron IDs in order.
    """
    ox, oy = origin
    ids: List[int] = []
    for i, lbl in enumerate(labels):
        x = ox
        y = oy + i * spacing
        tname = type_for_label1 if int(lbl) == 1 else type_for_label0
        if tname in model.NEURON_PRESETS:
            n = model.add_neuron_of_type(tname, x, y)
        else:
            n = model.add_neuron(x, y)
        ids.append(n.id)
    return ids


def create_input_column_count(
    model: GraphModel,
    num_points: int,
    p_one: float = 0.5,
    origin: Tuple[float, float] = (260.0, 60.0),
    spacing: float = 7.0,
    type_for_label1: str = "I",
    type_for_label0: str = "FS",
    seed: int | None = None,
) -> List[int]:
    labels = generate_binary_labels(num_points, p_one=p_one, seed=seed)
    return create_input_column_from_labels(
        model,
        labels,
        origin=origin,
        spacing=spacing,
        type_for_label1=type_for_label1,
        type_for_label0=type_for_label0,
    )


# --- Simple samplers for visualization-only datasets (normalized to [-1,1]) ---

def sample_random_points(num_points: int = 100, seed: int | None = None) -> List[LabeledPoint]:
    import random
    if seed is not None:
        random.seed(seed)
    pts: List[LabeledPoint] = []
    for _ in range(max(0, num_points)):
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        label = 1 if random.random() < 0.5 else 0
        pts.append((x, y, label))
    return pts


def sample_circle_dataset(
    n_inner: int = 50,
    n_outer: int = 50,
    r_inner: float = 0.35,
    r_outer: float = 0.75,
    noise: float = 0.03,
    seed: int | None = None,
) -> List[LabeledPoint]:
    import math, random
    if seed is not None:
        random.seed(seed)
    pts: List[LabeledPoint] = []
    for i in range(max(0, n_inner)):
        t = 2 * math.pi * i / max(1, n_inner)
        ri = max(0.0, r_inner + random.gauss(0.0, noise))
        x = ri * math.cos(t)
        y = ri * math.sin(t)
        pts.append((x, y, 0))
    for i in range(max(0, n_outer)):
        t = 2 * math.pi * i / max(1, n_outer)
        ro = max(0.0, r_outer + random.gauss(0.0, noise))
        x = ro * math.cos(t)
        y = ro * math.sin(t)
        pts.append((x, y, 1))
    return pts


# --- Grid-based samplers and helpers ---

def sample_random_grid(width: int = 30, height: int = 30, p_one: float = 0.5, seed: int | None = None) -> List[List[int]]:
    import random
    if seed is not None:
        random.seed(seed)
    grid: List[List[int]] = []
    for r in range(height):
        row: List[int] = []
        for c in range(width):
            row.append(1 if random.random() < p_one else 0)
        grid.append(row)
    return grid


def sample_circle_grid(width: int = 30, height: int = 30, radius: float = 0.6) -> List[List[int]]:
    # radius in normalized units relative to half-size; center at grid center
    grid: List[List[int]] = []
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    norm = max(cx, cy) or 1.0
    for r in range(height):
        row: List[int] = []
        for c in range(width):
            x = (c - cx) / norm
            y = (r - cy) / norm
            row.append(1 if (x * x + y * y) ** 0.5 <= radius else 0)
        grid.append(row)
    return grid


def flatten_grid_to_labels(grid: PixelGrid) -> List[int]:
    labels: List[int] = []
    for row in grid:
        for v in row:
            labels.append(1 if _to_float01(v) > 0.5 else 0)
    return labels


@dataclass
class InputLayer:
    neuron_ids: List[List[int]]
    origin: Tuple[float, float]
    spacing: Tuple[float, float]

    def shape(self) -> Tuple[int, int]:
        rows = len(self.neuron_ids)
        cols = len(self.neuron_ids[0]) if rows > 0 else 0
        return rows, cols


def _to_float01(value: Union[Number, bool]) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        v = float(value)
    except Exception:
        return 0.0
    # If values look like 0..255 scale, map to 0..1
    if v > 1.0:
        return max(0.0, min(1.0, v / 255.0))
    return max(0.0, min(1.0, v))


def create_input_layer_from_grid(
    model: GraphModel,
    grid: PixelGrid,
    origin: Tuple[float, float] = (260.0, 60.0),
    spacing: Tuple[float, float] = (18.0, 18.0),
    threshold: float = 0.5,
    white_type: str = "I",
    black_type: str = "FS",
) -> InputLayer:
    """
    Create a grid of input neurons mapped from a pixel grid.

    - Pixels > threshold become white_type neurons (default I) which spike due to baseline current.
    - Pixels <= threshold become black_type neurons (default FS) which stay silent without input.

    Returns an InputLayer with neuron IDs laid out as grid[row][col].
    """
    ox, oy = origin
    sx, sy = spacing
    neuron_ids: List[List[int]] = []
    for r, row in enumerate(grid):
        id_row: List[int] = []
        for c, pix in enumerate(row):
            v = _to_float01(pix)
            x = ox + c * sx
            y = oy + r * sy
            tname = white_type if v > threshold else black_type
            if tname in model.NEURON_PRESETS:
                n = model.add_neuron_of_type(tname, x, y)
            else:
                n = model.add_neuron(x, y)
            id_row.append(n.id)
        neuron_ids.append(id_row)
    return InputLayer(neuron_ids=neuron_ids, origin=origin, spacing=spacing)


def present_grid_as_pulses(
    sim: IzhikevichSimulator,
    input_layer: InputLayer,
    grid: PixelGrid,
    amount_white: float = 10.0,
    amount_black: float = 0.0,
    threshold: float = 0.5,
) -> None:
    """
    Present the given grid for a single simulation step as current pulses:
    - For each input neuron, inject +amount_white if pixel > threshold, else +amount_black.
    - Call sim.advance_steps(k) after this to step the network.
    """
    rows = len(input_layer.neuron_ids)
    cols = len(input_layer.neuron_ids[0]) if rows > 0 else 0
    for r in range(rows):
        for c in range(cols):
            nid = input_layer.neuron_ids[r][c]
            v = _to_float01(grid[r][c])
            amt = amount_white if v > threshold else amount_black
            if amt != 0.0:
                sim.inject_current(nid, amt)


def demo_make_checkerboard(model: GraphModel, size: int = 8) -> PixelGrid:
    grid: List[List[int]] = []
    for r in range(size):
        row: List[int] = []
        for c in range(size):
            row.append(1 if (r + c) % 2 == 0 else 0)
        grid.append(row)
    return grid


def build_input_layer_for_image(
    model: GraphModel,
    image: PixelGrid,
    origin: Tuple[float, float] = (260.0, 60.0),
    spacing: Tuple[float, float] = (18.0, 18.0),
    threshold: float = 0.5,
    white_type: str = "I",
    black_type: str = "FS",
) -> InputLayer:
    """Alias for create_input_layer_from_grid for semantic clarity."""
    return create_input_layer_from_grid(
        model,
        image,
        origin=origin,
        spacing=spacing,
        threshold=threshold,
        white_type=white_type,
        black_type=black_type,
    )

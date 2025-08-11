from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

from graph_model import GraphModel, NeuronModel, ConnectionModel
from simulator import IzhikevichSimulator, SimulationConfig
from training import (
    sample_random_grid,
    sample_circle_grid,
    flatten_grid_to_labels,
    create_input_column_from_labels,
)


SIDEBAR_WIDTH = 200
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
FPS = 60
MAX_DRAW_CONNECTIONS = 1000000


@dataclass
class Button:
    label: str
    rect: pygame.Rect
    color: Tuple[int, int, int]
    hover_color: Tuple[int, int, int]

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, mouse_pos: Tuple[int, int]) -> None:
        is_hover = self.rect.collidepoint(mouse_pos)
        pygame.draw.rect(surface, self.hover_color if is_hover else self.color, self.rect, border_radius=6)
        text_surface = font.render(self.label, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


@dataclass
class Message:
    text: str
    created_ms: int


class PygameGraphApp:
    def __init__(self, model: GraphModel) -> None:
        pygame.init()
        pygame.display.set_caption("Vibe NeuroSim")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)

        self.sidebar_color = (30, 30, 35)
        self.canvas_bg = (245, 245, 248)
        self.grid_color = (230, 230, 235)
        self.right_panel_color = (36, 36, 42)
        self.right_panel_width = 220
        self.output_panel_height = 180
        self.zoom: float = 1.0
        # Right panel scrolling
        self.right_scroll_offset: int = 0
        self._right_content_height: int = 0

        # UI state
        self.buttons: List[Button] = []
        self.dragging_palette_item: Optional[str] = None  # currently only "Neuron"
        self.drag_position = pygame.Vector2(0, 0)

        # Canvas state
        self.model = model
        self.sim = IzhikevichSimulator(self.model)
        self.camera_offset = pygame.Vector2(0, 0)
        self.is_panning = False
        self.pan_start_mouse = pygame.Vector2(0, 0)
        self.pan_start_camera = pygame.Vector2(0, 0)

        self.dragging_existing_neuron: Optional[NeuronModel] = None
        self.drag_offset = pygame.Vector2(0, 0)

        self.dragging_connection_from: Optional[NeuronModel] = None
        self.dragging_connection_pos = pygame.Vector2(0, 0)

        self.selected_connection: Optional[ConnectionModel] = None
        self.messages: List[Message] = []
        self._drawn_connections_last: int = 0

        # Sim UI state
        self._steps_per_click: int = 10
        self._dt_step: float = 0.5
        # Hidden layer generation UI state
        self._hidden_layer_count: int = 1
        self._hidden_neurons_per_layer: int = 16
        # Output rate classification and error metric
        self._rate_threshold: float = 0.5
        self._last_mse: Optional[float] = None
        # Rolling MSE over recent steps
        self._mse_window_size: int = 40
        self._mse_window: List[float] = []
        self._mse_window_avg: Optional[float] = None
        # Auto-perturbation controls/state
        self._auto_perturb_enabled: bool = False
        self._auto_perturb_interval: int = 40
        self._auto_perturb_delta: float = 0.2
        self._auto_perturb_state: str = "idle"  # 'idle' or 'evaluating'
        self._auto_perturb_steps: int = 0
        self._auto_perturb_snapshot: dict[tuple[int, int], float] = {}
        self._auto_perturb_pre_metric: Optional[float] = None
        self._auto_perturb_manual_active: bool = False
        # Placeholders for input/output metadata for wiring
        self._input_grid_shape: Optional[Tuple[int, int]] = None
        self._input_ids_flat: Optional[List[int]] = None
        self._input_origin: Optional[Tuple[float, float]] = None
        self._input_spacing: Optional[Tuple[float, float]] = None
        self._output_origin: Optional[Tuple[float, float]] = None
        self._output_spacing: Optional[Tuple[float, float]] = None
        self._hidden_layer_ids: List[List[int]] = []
        # Placeholders for rects populated in _build_ui
        self.sim_dt_minus_rect = pygame.Rect(0, 0, 0, 0)
        self.sim_dt_plus_rect = pygame.Rect(0, 0, 0, 0)
        self.sim_steps_minus_rect = pygame.Rect(0, 0, 0, 0)
        self.sim_steps_plus_rect = pygame.Rect(0, 0, 0, 0)
        self.sim_advance_rect = pygame.Rect(0, 0, 0, 0)
        # Neuron overview panel rect placeholder
        self.neuron_overview_rect = pygame.Rect(0, 0, 0, 0)

        self._build_ui()

    def _build_ui(self) -> None:
        padding = 12
        button_height = 40
        x = 12
        y = 12

        def make_button(label: str) -> Button:
            nonlocal y
            btn = Button(
                label=label,
                rect=pygame.Rect(x, y, SIDEBAR_WIDTH - 2 * padding, button_height),
                color=(70, 70, 80),
                hover_color=(90, 90, 110),
            )
            y += button_height + 10
            return btn

        self.buttons = [
            make_button("RS Neuron"),  # draggable palette item
            make_button("FS Neuron"),
            make_button("I Neuron"),
            # make_button("IB Neuron"),
            # make_button("CH Neuron"),
            # make_button("LTS Neuron"),
            make_button("Delete"),
        ]

        # Simulate panel layout under buttons
        sim_top = y + 6
        # dt row
        self.sim_dt_minus_rect = pygame.Rect(12, sim_top, 22, 22)
        self.sim_dt_plus_rect = pygame.Rect(12 + 22 + 120, sim_top, 22, 22)
        # steps row
        steps_top = sim_top + 26
        self.sim_steps_minus_rect = pygame.Rect(12, steps_top, 22, 22)
        self.sim_steps_plus_rect = pygame.Rect(12 + 22 + 120, steps_top, 22, 22)
        # advance button
        self.sim_advance_rect = pygame.Rect(12, steps_top + 30, SIDEBAR_WIDTH - 24, 32)
        # Neuron overview panel below simulate controls
        overview_top = self.sim_advance_rect.bottom + 12
        self.neuron_overview_rect = pygame.Rect(12, overview_top, SIDEBAR_WIDTH - 24, 180)

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                self._handle_event(event)

            self._draw()
            pygame.display.flip()
            self.clock.tick(FPS)

    def _handle_event(self, event: pygame.event.Event) -> None:
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos_v = pygame.Vector2(mouse_pos)
        canvas_rect = pygame.Rect(
            SIDEBAR_WIDTH,
            0,
            WINDOW_WIDTH - SIDEBAR_WIDTH - self.right_panel_width,
            WINDOW_HEIGHT,
        )

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Simulate controls first
            if self.sim_advance_rect.collidepoint(mouse_pos):
                # Track output neuron spike rate while advancing
                def on_step(_model: GraphModel) -> None:
                    if hasattr(self, "_output_grid_shape") and hasattr(self, "_output_rates"):
                        h, w = self._output_grid_shape
                        total = h * w
                        if total > 0 and len(self.model.neurons) >= total:
                            # Cache output neuron ids if needed (assumes outputs were created last)
                            if not hasattr(self, "_output_ids") or len(self._output_ids) != total:
                                self._output_ids = [n.id for n in self.model.neurons[-total:]]
                            for idx, nid in enumerate(self._output_ids):
                                n = self.model.find_neuron(nid)
                                if n is None:
                                    continue
                                # Simple, sensitive indicator: latch to 1.0 on spike, otherwise decay
                                if n.spiked:
                                    self._output_rates[idx] = 1.0
                                else:
                                    self._output_rates[idx] *= 0.9
                            # Compute classification-based MSE vs sampled grid
                            grid = getattr(self, "_sampled_grid", None)
                            if grid is not None:
                                h = min(h, len(grid))
                                w = min(w, len(grid[0]) if len(grid) > 0 else 0)
                                if h > 0 and w > 0:
                                    err_sum = 0.0
                                    cnt = 0
                                    thr = getattr(self, "_rate_threshold", 0.5)
                                    for r in range(h):
                                        for c in range(w):
                                            idx = r * (self._output_grid_shape[1]) + c
                                            if idx >= len(self._output_rates):
                                                continue
                                            pred = 1 if self._output_rates[idx] >= thr else 0
                                            target = 1 if grid[r][c] else 0
                                            d = float(pred - target)
                                            err_sum += d * d
                                            cnt += 1
                                    if cnt > 0:
                                        self._last_mse = err_sum / float(cnt)
                                        # Update rolling window
                                        self._mse_window.append(self._last_mse)
                                        if len(self._mse_window) > self._mse_window_size:
                                            self._mse_window.pop(0)
                                        if self._mse_window:
                                            self._mse_window_avg = sum(self._mse_window) / float(len(self._mse_window))

                # Integrate auto-perturbation cycle management
                def managed_on_step(m: GraphModel) -> None:
                    on_step(m)
                    self._maybe_run_auto_perturbation_step()

                self.sim.advance_steps(self._steps_per_click, on_step=managed_on_step)
                self._add_message(f"advanced {self._steps_per_click} steps")
                return
            # Training panel buttons
            if hasattr(self, "_train_btn_random") and self._train_btn_random.collidepoint(mouse_pos):
                self._sampled_grid = sample_random_grid(3, 3, p_one=0.5)
                self._add_message("sampled random grid 3x3")
                return
            if hasattr(self, "_train_btn_circle") and self._train_btn_circle.collidepoint(mouse_pos):
                self._sampled_grid = sample_circle_grid(3, 3, radius=0.6)
                self._add_message("sampled circle grid 3x3")
                return
            # Create input grid from current sampled grid
            if hasattr(self, "_train_btn_create") and self._train_btn_create.collidepoint(mouse_pos):
                if getattr(self, "_sampled_grid", None) is not None:
                    grid = self._sampled_grid
                    h = len(grid)
                    w = len(grid[0]) if h > 0 else 0
                    if h == 0 or w == 0:
                        self._add_message("empty grid")
                        return
                    # Increased left offset and spacing for input grid
                    ox = SIDEBAR_WIDTH + 140
                    oy = 100
                    sx = 24.0
                    sy = 24.0
                    input_ids: List[int] = []
                    for r in range(h):
                        for c in range(w):
                            v = 1 if grid[r][c] else 0
                            x = ox + c * sx
                            y = oy + r * sy
                            tname = "I" if v == 1 else "FS"
                            n = self.model.add_neuron_of_type(tname, x, y)
                            input_ids.append(n.id)
                            # Color neurons to match preview palette
                            color = (240, 160, 90) if v == 1 else (90, 160, 240)
                            # Store color on the neuron instance by duck-typing (or use an external map)
                            setattr(n, "_viz_color", color)
                    # Store metadata for wiring
                    self._input_grid_shape = (h, w)
                    self._input_ids_flat = input_ids
                    self._input_origin = (ox, oy)
                    self._input_spacing = (sx, sy)
                    self._add_message(f"created input grid ({w}x{h})")
                else:
                    self._add_message("no sampled grid")
                return
            # Create output neurons based on sampled grid dimensions
            if hasattr(self, "_train_btn_output") and self._train_btn_output.collidepoint(mouse_pos):
                grid = getattr(self, "_sampled_grid", None)
                if grid is None:
                    self._add_message("no sampled grid")
                    return
                h = len(grid)
                w = len(grid[0]) if h > 0 else 0
                if h == 0 or w == 0:
                    self._add_message("empty grid")
                    return
                # Create h*w basic RS neurons packed as a grid to the right of inputs
                # Use larger spacing and a wider gap from input grid
                in_ox = SIDEBAR_WIDTH + 140
                in_sx =24.0
                gap = 220.0
                ox = in_ox + w * in_sx + gap
                oy = 100
                sx = 24.0
                sy = 24.0
                new_ids: list[int] = []
                for r in range(h):
                    for c in range(w):
                        x = ox + c * sx
                        y = oy + r * sy
                        n = self.model.add_neuron_of_type("RS", x, y)
                        new_ids.append(n.id)
                # Initialize output rate buffer and shape
                self._output_grid_shape = (h, w)
                self._output_rates = [0.0 for _ in range(h * w)]
                self._output_ids = new_ids
                # Store metadata for wiring
                self._output_origin = (ox, oy)
                self._output_spacing = (sx, sy)
                self._add_message(f"created output neurons ({w}x{h})")
                return
            # Hidden layer controls
            if hasattr(self, "_hidden_layers_minus_rect") and self._hidden_layers_minus_rect.collidepoint(mouse_pos):
                self._hidden_layer_count = max(1, self._hidden_layer_count - 1)
                return
            if hasattr(self, "_hidden_layers_plus_rect") and self._hidden_layers_plus_rect.collidepoint(mouse_pos):
                self._hidden_layer_count = min(8, self._hidden_layer_count + 1)
                return
            if hasattr(self, "_hidden_neurons_minus_rect") and self._hidden_neurons_minus_rect.collidepoint(mouse_pos):
                self._hidden_neurons_per_layer = max(1, self._hidden_neurons_per_layer - 1)
                return
            if hasattr(self, "_hidden_neurons_plus_rect") and self._hidden_neurons_plus_rect.collidepoint(mouse_pos):
                self._hidden_neurons_per_layer = min(200, self._hidden_neurons_per_layer + 1)
                return
            if hasattr(self, "_hidden_generate_rect") and self._hidden_generate_rect.collidepoint(mouse_pos):
                try:
                    # Validate prerequisites
                    if not getattr(self, "_input_ids_flat", None) or not getattr(self, "_output_ids", None):
                        self._add_message("need input and output first")
                        return
                    h_in, w_in = self._input_grid_shape if self._input_grid_shape else (0, 0)
                    if h_in == 0 or w_in == 0:
                        self._add_message("invalid input grid")
                        return
                    in_ox, in_oy = self._input_origin if self._input_origin else (SIDEBAR_WIDTH + 140, 100)
                    in_sx, in_sy = self._input_spacing if self._input_spacing else (24.0, 24.0)
                    out_ox, out_oy = self._output_origin if self._output_origin else (in_ox + w_in * in_sx + 220.0, in_oy)
                    # Compute x positions across columns between input-right edge and output-left edge
                    input_right_x = in_ox + w_in * in_sx
                    total_layers = self._hidden_layer_count
                    if total_layers <= 0:
                        self._add_message("no hidden layers to generate")
                        return
                    span = max(60.0, (out_ox - input_right_x))
                    step_x = span / float(total_layers + 1)
                    neurons_per_layer = self._hidden_neurons_per_layer
                    mid_input_y = in_oy + (h_in - 1) * in_sy * 0.5
                    # Build layers
                    self._hidden_layer_ids = []
                    for li in range(total_layers):
                        col_x = input_right_x + step_x * (li + 1)
                        top_y = mid_input_y - (neurons_per_layer - 1) * in_sy * 0.5
                        layer_ids: List[int] = []
                        for r in range(neurons_per_layer):
                            n = self.model.add_neuron_of_type("RS", col_x, top_y + r * in_sy)
                            setattr(n, "_viz_color", (130, 200, 140))
                            layer_ids.append(n.id)
                        self._hidden_layer_ids.append(layer_ids)
                    # Connect fully: input -> L1 (fixed 1.0), hidden -> hidden (random init), last hidden -> output (fixed 1.0)
                    def _connect_fixed(src_ids: List[int], dst_ids: List[int], w: float) -> None:
                        add_fast = getattr(self.model, "add_connection_fast", None)
                        if callable(add_fast):
                            for sid in src_ids:
                                for tid in dst_ids:
                                    add_fast(sid, tid, weight=w)
                        else:
                            for sid in src_ids:
                                for tid in dst_ids:
                                    ok, _msg = self.model.add_connection(sid, tid)
                                    if ok:
                                        for c in reversed(self.model.connections):
                                            if c.source_id == sid and c.target_id == tid:
                                                c.weight = w
                                                break

                    def _connect_random(src_ids: List[int], dst_ids: List[int]) -> None:
                        import random
                        add_fast = getattr(self.model, "add_connection_fast", None)
                        if callable(add_fast):
                            for sid in src_ids:
                                for tid in dst_ids:
                                    add_fast(sid, tid, weight=random.uniform(-1.0, 1.0))
                        else:
                            for sid in src_ids:
                                for tid in dst_ids:
                                    ok, _msg = self.model.add_connection(sid, tid)
                                    if ok:
                                        w = random.uniform(-1.0, 1.0)
                                        for c in reversed(self.model.connections):
                                            if c.source_id == sid and c.target_id == tid:
                                                c.weight = w
                                                break

                    # Input -> first hidden (fixed)
                    prev_ids = list(self._input_ids_flat or [])
                    if self._hidden_layer_ids:
                        _connect_fixed(prev_ids, self._hidden_layer_ids[0], 1.0)
                        prev_ids = self._hidden_layer_ids[0]
                    # Hidden -> hidden (random)
                    for i in range(1, len(self._hidden_layer_ids)):
                        _connect_random(self._hidden_layer_ids[i - 1], self._hidden_layer_ids[i])
                        prev_ids = self._hidden_layer_ids[i]
                    # Last hidden -> output (fixed)
                    if prev_ids and getattr(self, "_output_ids", None):
                        _connect_fixed(prev_ids, list(self._output_ids or []), 1.0)
                    self._add_message(
                        f"generated {total_layers} layer(s) x {neurons_per_layer} neurons"
                    )
                except Exception as e:
                    self._add_message(f"error: {type(e).__name__}")
                return
            # Auto-perturbation controls
            if hasattr(self, "_ap_toggle_rect") and self._ap_toggle_rect.collidepoint(mouse_pos):
                self._auto_perturb_enabled = not self._auto_perturb_enabled
                self._auto_perturb_state = "idle"
                self._auto_perturb_steps = 0
                self._add_message(f"auto-perturb {'ON' if self._auto_perturb_enabled else 'OFF'}")
                return
            if hasattr(self, "_ap_interval_minus_rect") and self._ap_interval_minus_rect.collidepoint(mouse_pos):
                self._auto_perturb_interval = max(1, self._auto_perturb_interval - 1)
                return
            if hasattr(self, "_ap_interval_plus_rect") and self._ap_interval_plus_rect.collidepoint(mouse_pos):
                self._auto_perturb_interval = min(10000, self._auto_perturb_interval + 1)
                return
            if hasattr(self, "_ap_delta_minus_rect") and self._ap_delta_minus_rect.collidepoint(mouse_pos):
                self._auto_perturb_delta = max(0.0, round(self._auto_perturb_delta - 0.05, 3))
                return
            if hasattr(self, "_ap_delta_plus_rect") and self._ap_delta_plus_rect.collidepoint(mouse_pos):
                self._auto_perturb_delta = min(1.0, round(self._auto_perturb_delta + 0.05, 3))
                return
            if hasattr(self, "_ap_now_rect") and self._ap_now_rect.collidepoint(mouse_pos):
                # Apply a one-shot perturbation cycle (does not keep auto ON)
                self._begin_perturbation_cycle(manual=True)
                return
            if self.sim_dt_minus_rect.collidepoint(mouse_pos):
                self.sim.config.dt_ms = max(0.1, self.sim.config.dt_ms - self._dt_step)
                return
            if self.sim_dt_plus_rect.collidepoint(mouse_pos):
                self.sim.config.dt_ms = min(10.0, self.sim.config.dt_ms + self._dt_step)
                return
            if self.sim_steps_minus_rect.collidepoint(mouse_pos):
                self._steps_per_click = max(1, self._steps_per_click - 1)
                return
            if self.sim_steps_plus_rect.collidepoint(mouse_pos):
                self._steps_per_click = min(10000, self._steps_per_click + 1)
                return
            # Neuron one-step injection controls
            sel = self._get_selected_neuron()
            if sel is not None:
                if hasattr(self, "_neuron_inj_plus_rect") and self._neuron_inj_plus_rect.collidepoint(mouse_pos):
                    self.sim.inject_current(sel.id, +10.0)
                    self._add_message(f"inject +10 to neuron {sel.id}")
                    return
                if hasattr(self, "_neuron_inj_minus_rect") and self._neuron_inj_minus_rect.collidepoint(mouse_pos):
                    self.sim.inject_current(sel.id, -10.0)
                    self._add_message(f"inject -10 to neuron {sel.id}")
                    return
            # Weight buttons for selected connection
            for label, rect in self._get_selected_weight_button_rects():
                if rect.collidepoint(mouse_pos) and self.selected_connection is not None:
                    new_weight = -1.0 if label == "-1" else (0.0 if label == "0" else 1.0)
                    self.selected_connection.weight = max(-1.0, min(1.0, new_weight))
                    self._add_message(f"weight set to {int(self.selected_connection.weight)}")
                    return

            # Start dragging neuron from palette if clicked on a preset button
            for idx in self._preset_button_indices():
                if self.buttons[idx].rect.collidepoint(mouse_pos):
                    self.dragging_palette_item = self.buttons[idx].label
                    self.drag_position = mouse_pos_v
                    return

            # No dataset creation on left panel

            # Delete button acts on selected items
            del_idx = self._delete_button_index()
            if del_idx is not None and self.buttons[del_idx].rect.collidepoint(mouse_pos):
                self.model.delete_selected()
                # Clear any stale connection selection after deletion
                self.selected_connection = None
                return

            # Otherwise, canvas interactions
            if canvas_rect.collidepoint(mouse_pos):
                world_mouse = self.screen_to_world(mouse_pos_v)
                clicked_any = False
                # Neuron hit-test first
                for n in reversed(self.model.neurons):
                    if self._neuron_hit_test(n, world_mouse):
                        # Select this neuron; deselect others
                        self.model.clear_selection()
                        n.selected = True
                        # Start dragging neuron
                        self.dragging_existing_neuron = n
                        self.drag_offset = pygame.Vector2(n.x, n.y) - world_mouse
                        clicked_any = True
                        break
                if not clicked_any:
                    # Try selecting a connection
                    conn = self._find_connection_under_mouse(mouse_pos)
                    if conn is not None:
                        self.model.clear_selection()
                        if self.selected_connection is not None:
                            self.selected_connection.selected = False
                        self.selected_connection = conn
                        conn.selected = True
                        return
                    # Start panning
                    self.model.clear_selection()
                    if self.selected_connection is not None:
                        self.selected_connection.selected = False
                        self.selected_connection = None
                    self.is_panning = True
                    self.pan_start_mouse = mouse_pos_v
                    self.pan_start_camera = self.camera_offset.copy()

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            # Drop palette item onto canvas
            if self.dragging_palette_item:
                if canvas_rect.collidepoint(mouse_pos):
                    world_mouse = self.screen_to_world(mouse_pos_v)
                    label = self.dragging_palette_item
                    type_name = label.split()[0] if label else "RS"
                    if type_name in self.model.NEURON_PRESETS:
                        self.model.add_neuron_of_type(type_name, world_mouse.x, world_mouse.y)
                    else:
                        self.model.add_neuron(world_mouse.x, world_mouse.y)
                self.dragging_palette_item = None

            # Stop dragging existing neuron
            if self.dragging_existing_neuron is not None:
                self.dragging_existing_neuron = None

            # Stop panning
            if self.is_panning:
                self.is_panning = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            # Begin creating a connection with right-click drag from a neuron
            if canvas_rect.collidepoint(mouse_pos):
                world_mouse = self.screen_to_world(mouse_pos_v)
                for n in reversed(self.model.neurons):
                    if self._neuron_hit_test(n, world_mouse):
                        self.dragging_connection_from = n
                        self.dragging_connection_pos = world_mouse
                        break

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
            # Finish creating a connection if dropped over a neuron
            if self.dragging_connection_from is not None:
                world_mouse = self.screen_to_world(mouse_pos_v)
                made_any = False
                for n in reversed(self.model.neurons):
                    if n is self.dragging_connection_from:
                        continue
                    if self._neuron_hit_test(n, world_mouse):
                        ok, msg = self.model.add_connection(self.dragging_connection_from.id, n.id)
                        self._add_message(msg)
                        made_any = True
                        break
                if not made_any:
                    self._add_message("no connection")
                self.dragging_connection_from = None

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging_palette_item:
                self.drag_position = mouse_pos_v
            if self.dragging_existing_neuron is not None:
                world_mouse = self.screen_to_world(mouse_pos_v)
                self.dragging_existing_neuron.x = world_mouse.x + self.drag_offset.x
                self.dragging_existing_neuron.y = world_mouse.y + self.drag_offset.y
            elif self.is_panning:
                delta = mouse_pos_v - self.pan_start_mouse
                self.camera_offset = self.pan_start_camera + delta
            if self.dragging_connection_from is not None:
                self.dragging_connection_pos = self.screen_to_world(mouse_pos_v)

        elif event.type == pygame.MOUSEWHEEL:
            # Wheel: zoom on canvas, scroll on right panel
            right_x = WINDOW_WIDTH - self.right_panel_width
            canvas_rect = pygame.Rect(
                SIDEBAR_WIDTH,
                0,
                WINDOW_WIDTH - SIDEBAR_WIDTH - self.right_panel_width,
                WINDOW_HEIGHT,
            )
            right_rect = pygame.Rect(right_x, 0, self.right_panel_width, WINDOW_HEIGHT)
            if right_rect.collidepoint(mouse_pos):
                # Scroll right panel
                self.right_scroll_offset = max(
                    0,
                    min(
                        max(0, self._right_content_height - WINDOW_HEIGHT),
                        self.right_scroll_offset - event.y * 24,
                    ),
                )
            elif canvas_rect.collidepoint(mouse_pos):
                # Zoom canvas
                zoom_factor = 1.1 ** event.y
                self._apply_zoom(zoom_factor, mouse_pos_v)

        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                self.model.delete_selected()
                if self.selected_connection is not None and self.selected_connection.selected:
                    self.selected_connection = None

    def _neuron_hit_test(self, neuron: NeuronModel, point: pygame.Vector2) -> bool:
        radius = 18
        dx = neuron.x - point.x
        dy = neuron.y - point.y
        return (dx * dx + dy * dy) <= radius * radius

    def _draw(self) -> None:
        self.screen.fill(self.canvas_bg)

        # Sidebar (left)
        pygame.draw.rect(self.screen, self.sidebar_color, (0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        title_surface = self.font.render("Palette", True, (200, 200, 210))
        self.screen.blit(title_surface, (16, 10))
        # Right training panel with scrolling
        right_x = WINDOW_WIDTH - self.right_panel_width
        panel_rect = pygame.Rect(right_x, 0, self.right_panel_width, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, self.right_panel_color, panel_rect)
        # Scroll area clipping
        content_offset = -self.right_scroll_offset
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(panel_rect)
        tr_title = self.font.render("Training", True, (200, 200, 210))
        self.screen.blit(tr_title, (right_x + 12, 10 + content_offset))
        # Draw subpanels with y-offset
        self._draw_training_panel(right_x, content_offset)
        # Output visualization panel
        self._draw_output_panel(right_x, content_offset)
        # Restore clip
        self.screen.set_clip(prev_clip)

        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(self.screen, self.font, mouse_pos)
        # Simulate panel UI
        self._draw_simulate_panel()
        # Neuron overview panel (if a neuron is selected)
        self._draw_neuron_overview()

        # Grid and canvas (clip to canvas so sidebars stay on top)
        canvas_rect = pygame.Rect(
            SIDEBAR_WIDTH,
            0,
            WINDOW_WIDTH - SIDEBAR_WIDTH - self.right_panel_width,
            WINDOW_HEIGHT,
        )
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(canvas_rect)
        self._draw_grid()
        self._draw_connections()
        self._draw_neurons()
        self.screen.set_clip(prev_clip)

        # Palette drag preview
        if self.dragging_palette_item:
            preview_color = (120, 180, 220)
            pygame.draw.circle(self.screen, preview_color, (int(self.drag_position.x), int(self.drag_position.y)), 18)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(self.drag_position.x), int(self.drag_position.y)), 18, 2)
            # Label preview
            lbl = self.font.render(self.dragging_palette_item, True, (230, 230, 240))
            self.screen.blit(lbl, (self.drag_position.x + 16, self.drag_position.y - 10))

        # Connection drag preview (clip to canvas as well)
        if self.dragging_connection_from is not None:
            prev_clip2 = self.screen.get_clip()
            self.screen.set_clip(canvas_rect)
            start = self.world_to_screen(pygame.Vector2(self.dragging_connection_from.x, self.dragging_connection_from.y))
            end = self.world_to_screen(self.dragging_connection_pos)
            pygame.draw.line(self.screen, (120, 120, 130), (int(start.x), int(start.y)), (int(end.x), int(end.y)), 2)
            self.screen.set_clip(prev_clip2)

        # Clear invalid connection selection
        if self.selected_connection is not None and not self._selected_connection_is_valid():
            self.selected_connection = None

        # Messages and selected connection controls
        self._draw_stats()
        self._draw_messages()
        self._draw_selected_weight_buttons()

    def _draw_training_panel(self, right_x: int, y_offset: int = 0) -> None:
        # Buttons for sampling datasets (visual only)
        y = 40 + y_offset
        btn_w = self.right_panel_width - 24
        buttons = [
            ("Sample Random Grid", pygame.Rect(right_x + 12, y, btn_w, 24)),
            ("Sample Circle Grid", pygame.Rect(right_x + 12, y + 28, btn_w, 24)),
            ("Create Input Grid", pygame.Rect(right_x + 12, y + 28 + 28, btn_w, 24)),
            ("Create Output Neurons", pygame.Rect(right_x + 12, y + 28 + 28 + 28, btn_w, 24)),
        ]
        mouse_pos = pygame.mouse.get_pos()
        for label, rect in buttons:
            is_hover = rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, (60, 60, 70) if not is_hover else (85, 85, 100), rect, border_radius=4)
            t = self.font.render(label, True, (230, 230, 240))
            tr = t.get_rect(center=rect.center)
            self.screen.blit(t, tr)

        # Remember rects for click detection
        self._train_btn_random = buttons[0][1]
        self._train_btn_circle = buttons[1][1]
        self._train_btn_create = buttons[2][1]
        self._train_btn_output = buttons[3][1]

        # Preview sampled points
        preview_rect = pygame.Rect(right_x + 12, buttons[-1][1].bottom + 12 + y_offset, btn_w, 120)
        pygame.draw.rect(self.screen, (32, 32, 38), preview_rect, border_radius=6)
        pygame.draw.rect(self.screen, (55, 55, 64), preview_rect, width=1, border_radius=6)

        if not hasattr(self, "_sampled_grid"):
            self._sampled_grid = None
        # draw grid in preview
        if self._sampled_grid is not None:
            grid = self._sampled_grid
            h = len(grid)
            w = len(grid[0]) if h > 0 else 0
            if w > 0 and h > 0:
                cell_w = preview_rect.width / w
                cell_h = preview_rect.height / h
                for r in range(h):
                    for c in range(w):
                        v = 1 if grid[r][c] else 0
                        color = (240, 160, 90) if v == 1 else (90, 160, 240)
                        rx = int(preview_rect.left + c * cell_w)
                        ry = int(preview_rect.top + r * cell_h)
                        rw = int(cell_w + 0.999)
                        rh = int(cell_h + 0.999)
                        pygame.draw.rect(self.screen, color, (rx, ry, rw, rh))
        self._train_preview_rect = preview_rect

        # Hidden layer controls below preview
        ctl_y = preview_rect.bottom + 12
        # Hidden layers count row
        self._hidden_layers_minus_rect = pygame.Rect(right_x + 12, ctl_y, 22, 22)
        self._hidden_layers_plus_rect = pygame.Rect(right_x + 12 + 22 + 120, ctl_y, 22, 22)
        self._draw_small_btn(self._hidden_layers_minus_rect, "-")
        text_layers = self.font.render(f"Hidden layers: {self._hidden_layer_count}", True, (230, 230, 240))
        self.screen.blit(text_layers, (self._hidden_layers_minus_rect.right + 6, self._hidden_layers_minus_rect.top + 3))
        self._draw_small_btn(self._hidden_layers_plus_rect, "+")

        # Neurons per layer row
        ctl_y2 = ctl_y + 28
        self._hidden_neurons_minus_rect = pygame.Rect(right_x + 12, ctl_y2, 22, 22)
        self._hidden_neurons_plus_rect = pygame.Rect(right_x + 12 + 22 + 120, ctl_y2, 22, 22)
        self._draw_small_btn(self._hidden_neurons_minus_rect, "-")
        text_neurons = self.font.render(f"Neurons/layer: {self._hidden_neurons_per_layer}", True, (230, 230, 240))
        self.screen.blit(text_neurons, (self._hidden_neurons_minus_rect.right + 6, self._hidden_neurons_minus_rect.top + 3))
        self._draw_small_btn(self._hidden_neurons_plus_rect, "+")

        # Generate button
        ctl_y3 = ctl_y2 + 28
        self._hidden_generate_rect = pygame.Rect(right_x + 12, ctl_y3, btn_w, 28)
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self._hidden_generate_rect.collidepoint(mouse_pos)
        pygame.draw.rect(self.screen, (60, 60, 70) if not is_hover else (85, 85, 100), self._hidden_generate_rect, border_radius=4)
        tgen = self.font.render("Generate Hidden Layers", True, (230, 230, 240))
        tgenr = tgen.get_rect(center=self._hidden_generate_rect.center)
        self.screen.blit(tgen, tgenr)

        # Auto-perturbation controls
        ap_y = self._hidden_generate_rect.bottom + 10
        # Toggle
        self._ap_toggle_rect = pygame.Rect(right_x + 12, ap_y, btn_w, 24)
        is_hover = self._ap_toggle_rect.collidepoint(mouse_pos)
        pygame.draw.rect(self.screen, (60, 60, 70) if not is_hover else (85, 85, 100), self._ap_toggle_rect, border_radius=4)
        tgl = self.font.render(f"Auto-perturb: {'ON' if self._auto_perturb_enabled else 'OFF'}", True, (230, 230, 240))
        self.screen.blit(tgl, self._ap_toggle_rect.move(8, 4))

        # Interval row
        ap_y += 24
        self._ap_interval_minus_rect = pygame.Rect(right_x + 12, ap_y, 22, 22)
        self._ap_interval_plus_rect = pygame.Rect(right_x + 12 + 22 + 120, ap_y, 22, 22)
        self._draw_small_btn(self._ap_interval_minus_rect, "-")
        txt = self.font.render(f"Interval: {self._auto_perturb_interval}", True, (230, 230, 240))
        self.screen.blit(txt, (self._ap_interval_minus_rect.right + 6, self._ap_interval_minus_rect.top + 3))
        self._draw_small_btn(self._ap_interval_plus_rect, "+")

        # Delta row
        ap_y += 22
        self._ap_delta_minus_rect = pygame.Rect(right_x + 12, ap_y, 22, 22)
        self._ap_delta_plus_rect = pygame.Rect(right_x + 12 + 22 + 120, ap_y, 22, 22)
        self._draw_small_btn(self._ap_delta_minus_rect, "-")
        txt = self.font.render(f"Delta: ±{self._auto_perturb_delta:.2f}", True, (230, 230, 240))
        self.screen.blit(txt, (self._ap_delta_minus_rect.right + 6, self._ap_delta_minus_rect.top + 3))
        self._draw_small_btn(self._ap_delta_plus_rect, "+")

        # Trigger now
        ap_y += 24
        # Track total content height for scrolling
        self._right_content_height = max(self._right_content_height, ap_y + 40 - y_offset)
        self._ap_now_rect = pygame.Rect(right_x + 12, ap_y, btn_w, 24)
        is_hover = self._ap_now_rect.collidepoint(mouse_pos)
        pygame.draw.rect(self.screen, (60, 60, 70) if not is_hover else (85, 85, 100), self._ap_now_rect, border_radius=4)
        nowt = self.font.render("Perturb Now", True, (230, 230, 240))
        self.screen.blit(nowt, self._ap_now_rect.move(8, 4))

    def _draw_output_panel(self, right_x: int, y_offset: int = 0) -> None:
        # Panel for visualizing output neuron activity
        panel_y = WINDOW_HEIGHT - self.output_panel_height - 12 + y_offset
        rect = pygame.Rect(right_x + 12, panel_y, self.right_panel_width - 24, self.output_panel_height)
        pygame.draw.rect(self.screen, (32, 32, 38), rect, border_radius=6)
        pygame.draw.rect(self.screen, (55, 55, 64), rect, width=1, border_radius=6)
        title = self.font.render("Output", True, (200, 200, 210))
        self.screen.blit(title, (rect.left + 8, rect.top + 6))
        # Show metrics in a column
        thr = getattr(self, "_rate_threshold", 0.5)
        mse_val = getattr(self, "_last_mse", None)
        mse_window_avg = getattr(self, "_mse_window_avg", None)
        filled = len(getattr(self, "_mse_window", []))
        window_size = getattr(self, "_mse_window_size", 100)
        lines = [
            f"thr={thr:.2f}",
            f"MSE={mse_val:.3f}" if mse_val is not None else "MSE=--",
            f"EL={mse_window_avg:.3f}" if mse_window_avg is not None else f"EL=-- ({filled}/{window_size})",
        ]
        info_y = rect.top + 20
        line_h = 16
        for txt in lines:
            surf = self.font.render(txt, True, (190, 190, 200))
            self.screen.blit(surf, (rect.left + 8, info_y))
            info_y += line_h

        grid = getattr(self, "_output_grid_shape", None)
        rates = getattr(self, "_output_rates", None)
        if grid is None or rates is None:
            return
        h, w = grid
        if w <= 0 or h <= 0:
            return
        # Leave room for the info column above
        area_top = info_y + 4
        area = pygame.Rect(rect.left + 8, area_top, rect.width - 16, rect.height - (area_top - rect.top) - 12)
        cell_w = area.width / w
        cell_h = area.height / h
        for r in range(h):
            for c in range(w):
                idx = r * w + c
                rate = rates[idx] if idx < len(rates) else 0.0
                # map rate to color (0 -> blue, high -> yellow)
                t = max(0.0, min(1.0, rate))
                color = (
                    int(90 + t * (240 - 90)),
                    int(160 + t * (230 - 160)),
                    int(240 - t * (240 - 90)),
                )
                rx = int(area.left + c * cell_w)
                ry = int(area.top + r * cell_h)
                rw = int(cell_w + 0.999)
                rh = int(cell_h + 0.999)
                pygame.draw.rect(self.screen, color, (rx, ry, rw, rh))

        # Also color output neurons on canvas according to the same rate mapping
        if hasattr(self, "_output_ids"):
            for idx, nid in enumerate(self._output_ids):
                rate = rates[idx] if idx < len(rates) else 0.0
                t = max(0.0, min(1.0, rate))
                color = (
                    int(90 + t * (240 - 90)),
                    int(160 + t * (230 - 160)),
                    int(240 - t * (240 - 90)),
                )
                n = self.model.find_neuron(nid)
                if n is not None:
                    setattr(n, "_viz_color", color)

    def _draw_simulate_panel(self) -> None:
        # Title
        title_y = self.sim_dt_minus_rect.top - 18
        title = self.font.render("Simulate", True, (200, 200, 210))
        self.screen.blit(title, (16, title_y))

        # dt row: [-]  dt: X ms  [+]
        def draw_small_btn(rect: pygame.Rect, label: str) -> None:
            mouse_pos = pygame.mouse.get_pos()
            is_hover = rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, (60, 60, 70) if not is_hover else (85, 85, 100), rect, border_radius=4)
            t = self.font.render(label, True, (230, 230, 240))
            tr = t.get_rect(center=rect.center)
            self.screen.blit(t, tr)

        draw_small_btn(self.sim_dt_minus_rect, "-")
        dt_text = self.font.render(f"dt: {self.sim.config.dt_ms:.1f} ms", True, (230, 230, 240))
        self.screen.blit(dt_text, (self.sim_dt_minus_rect.right + 6, self.sim_dt_minus_rect.top + 3))
        draw_small_btn(self.sim_dt_plus_rect, "+")

        # steps row: [-] steps: N  [+]
        draw_small_btn(self.sim_steps_minus_rect, "-")
        steps_text = self.font.render(f"steps: {self._steps_per_click}", True, (230, 230, 240))
        self.screen.blit(steps_text, (self.sim_steps_minus_rect.right + 6, self.sim_steps_minus_rect.top + 3))
        draw_small_btn(self.sim_steps_plus_rect, "+")

        # Advance button with step counter
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self.sim_advance_rect.collidepoint(mouse_pos)
        pygame.draw.rect(self.screen, (70, 70, 80) if not is_hover else (90, 90, 110), self.sim_advance_rect, border_radius=6)
        adv_text = self.font.render(f"Advance ({self.sim.step_count})", True, (255, 255, 255))
        adv_rect = adv_text.get_rect(center=self.sim_advance_rect.center)
        self.screen.blit(adv_text, adv_rect)

    def _get_selected_neuron(self) -> Optional[NeuronModel]:
        for n in self.model.neurons:
            if n.selected:
                return n
        return None

    def _preset_button_indices(self) -> List[int]:
        return [i for i, b in enumerate(self.buttons) if b.label.endswith("Neuron")]

    def _delete_button_index(self) -> Optional[int]:
        for i, b in enumerate(self.buttons):
            if b.label == "Delete":
                return i
        return None

    # helper removed

    def _draw_neuron_overview(self) -> None:
        neuron = self._get_selected_neuron()
        if neuron is None:
            return
        rect = self.neuron_overview_rect
        # Panel background
        pygame.draw.rect(self.screen, (40, 40, 48), rect, border_radius=6)
        pygame.draw.rect(self.screen, (60, 60, 70), rect, width=1, border_radius=6)

        # Title and params
        title = self.font.render(f"Neuron {neuron.id} ({neuron.type_name})", True, (210, 210, 220))
        self.screen.blit(title, (rect.left + 8, rect.top + 6))
        param_y = rect.top + 24
        params = f"a={neuron.a:.2f}  b={neuron.b:.2f}  c={neuron.c:.0f}  d={neuron.d:.0f}"
        p_text = self.font.render(params, True, (190, 190, 200))
        self.screen.blit(p_text, (rect.left + 8, param_y))

        # One-step current injection controls
        inj_plus = pygame.Rect(rect.right - 70, rect.top + 6, 26, 20)
        inj_minus = pygame.Rect(rect.right - 38, rect.top + 6, 26, 20)
        self._neuron_inj_plus_rect = inj_plus
        self._neuron_inj_minus_rect = inj_minus
        self._draw_small_btn(inj_plus, "+")
        self._draw_small_btn(inj_minus, "-")

        # Phase plane area
        plot_margin = 8
        plot_rect = pygame.Rect(rect.left + plot_margin, param_y + 18, rect.width - 2 * plot_margin, rect.height - (param_y - rect.top) - 26)
        pygame.draw.rect(self.screen, (32, 32, 38), plot_rect)
        pygame.draw.rect(self.screen, (55, 55, 64), plot_rect, width=1)

        # Axes ranges
        v_min, v_max = -90.0, 40.0
        u_min, u_max = -30.0, 30.0

        def vu_to_px(v: float, u: float) -> Tuple[int, int]:
            x = plot_rect.left + int((v - v_min) / (v_max - v_min) * (plot_rect.width - 1))
            # y: top= u_max
            y = plot_rect.top + int((u_max - u) / (u_max - u_min) * (plot_rect.height - 1))
            return x, y

        # Constrain drawing within the plot area
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(plot_rect)

        # Axes inside plot: u=0 (horizontal) and v=0 (vertical)
        # Draw border already exists; we add axis lines and ticks with numbers
        def draw_axes_and_ticks() -> None:
            # Axes lines
            # u = 0 horizontal
            if u_min <= 0.0 <= u_max:
                _, y0 = vu_to_px(v_min, 0.0)
                pygame.draw.line(self.screen, (80, 80, 95), (plot_rect.left, y0), (plot_rect.right - 1, y0), 1)
            # v = 0 vertical
            if v_min <= 0.0 <= v_max:
                x0, _ = vu_to_px(0.0, u_min)
                pygame.draw.line(self.screen, (80, 80, 95), (x0, plot_rect.top), (x0, plot_rect.bottom - 1), 1)

            # Ticks and numeric labels (draw inside the plot bounds)
            tick_color = (120, 120, 135)

            # v-axis ticks along bottom
            num_v_ticks = 5
            for i in range(num_v_ticks + 1):
                v = v_min + (v_max - v_min) * i / num_v_ticks
                x, y = vu_to_px(v, u_min)
                pygame.draw.line(self.screen, tick_color, (x, plot_rect.bottom - 6), (x, plot_rect.bottom - 1), 1)

            # u-axis ticks along left side
            num_u_ticks = 4
            for i in range(num_u_ticks + 1):
                u = u_min + (u_max - u_min) * i / num_u_ticks
                x, y = vu_to_px(v_min, u)
                pygame.draw.line(self.screen, tick_color, (plot_rect.left, y), (plot_rect.left + 5, y), 1)

        draw_axes_and_ticks()

        # Nullclines
        I = self.sim.config.input_current
        # u = b v (du/dt = 0)
        v_samples = [v_min + (v_max - v_min) * i / 60.0 for i in range(61)]
        points_du0 = [vu_to_px(v, neuron.b * v) for v in v_samples]
        pygame.draw.lines(self.screen, (140, 200, 230), False, points_du0, 1)
        # u = 0.04 v^2 + 5 v + 140 + I (dv/dt = 0)
        points_dv0 = [vu_to_px(v, 0.04 * v * v + 5 * v + 140.0 + I) for v in v_samples]
        pygame.draw.lines(self.screen, (230, 140, 140), False, points_dv0, 1)

        # Spike threshold line at v = 30 mV (reset trigger)
        v_thresh = 30.0
        x_thr = plot_rect.left + int((v_thresh - v_min) / (v_max - v_min) * (plot_rect.width - 1))
        if plot_rect.left <= x_thr < plot_rect.right:
            pygame.draw.line(
                self.screen,
                (240, 180, 120),
                (x_thr, plot_rect.top),
                (x_thr, plot_rect.bottom - 1),
                2,
            )

        # Current state point
        px, py = vu_to_px(neuron.v, neuron.u)
        # Clamp point to plot bounds
        px = max(plot_rect.left, min(plot_rect.right - 1, px))
        py = max(plot_rect.top, min(plot_rect.bottom - 1, py))
        color = (240, 230, 140) if neuron.spiked else (200, 220, 120)
        pygame.draw.circle(self.screen, color, (px, py), 3)
        # Restore clip
        self.screen.set_clip(prev_clip)

    def _draw_small_btn(self, rect: pygame.Rect, label: str) -> None:
        mouse_pos = pygame.mouse.get_pos()
        is_hover = rect.collidepoint(mouse_pos)
        pygame.draw.rect(self.screen, (60, 60, 70) if not is_hover else (85, 85, 100), rect, border_radius=4)
        t = self.font.render(label, True, (230, 230, 240))
        tr = t.get_rect(center=rect.center)
        self.screen.blit(t, tr)

    # --- Auto-perturbation logic ---
    def _begin_perturbation_cycle(self, manual: bool = False) -> None:
        if manual:
            self._auto_perturb_manual_active = True
        elif not self._auto_perturb_enabled:
            self._auto_perturb_enabled = True
        # Snapshot current weights
        self._auto_perturb_snapshot = {(c.source_id, c.target_id): c.weight for c in self.model.connections}
        # Record current long-term metric as baseline
        self._auto_perturb_pre_metric = self._mse_window_avg if self._mse_window_avg is not None else self._last_mse
        self._auto_perturb_steps = 0
        self._auto_perturb_state = "evaluating"
        # Apply random perturbations to all connections
        try:
            import random
            delta = self._auto_perturb_delta
            for c in self.model.connections:
                c.weight = max(-1.0, min(1.0, c.weight + random.uniform(-delta, delta)))
        except Exception:
            pass
        self._add_message("perturb applied; evaluating…")

    def _maybe_run_auto_perturbation_step(self) -> None:
        # When OFF, only proceed if a manual cycle is active
        if not self._auto_perturb_enabled and not self._auto_perturb_manual_active:
            return
        # If not in evaluation, count towards next cycle
        if self._auto_perturb_state == "idle":
            self._auto_perturb_steps += 1
            if self._auto_perturb_steps >= self._auto_perturb_interval:
                self._begin_perturbation_cycle()
            return
        # In evaluation phase
        if self._auto_perturb_state == "evaluating":
            self._auto_perturb_steps += 1
            if self._auto_perturb_steps >= self._auto_perturb_interval:
                # Compare performance
                current_metric = self._mse_window_avg if self._mse_window_avg is not None else self._last_mse
                pre = self._auto_perturb_pre_metric
                improved = False
                # Lower MSE is better
                if current_metric is not None and pre is not None:
                    improved = current_metric < pre
                elif current_metric is not None and pre is None:
                    improved = True
                # Keep or revert
                if improved:
                    self._add_message("kept perturbation (improved)")
                else:
                    # Revert weights from snapshot
                    snap = self._auto_perturb_snapshot
                    for c in self.model.connections:
                        key = (c.source_id, c.target_id)
                        if key in snap:
                            c.weight = snap[key]
                    self._add_message("reverted perturbation (worse)")
                # Reset cycle
                self._auto_perturb_snapshot = {}
                self._auto_perturb_steps = 0
                self._auto_perturb_state = "idle"
                # End manual cycle if it was manual
                if self._auto_perturb_manual_active and not self._auto_perturb_enabled:
                    self._auto_perturb_manual_active = False

    def _draw_neurons(self) -> None:
        # Scale neuron radius with zoom for proportional view
        base_r = 18.0
        r = max(4, int(base_r * self.zoom))
        r_sel_outer = max(6, int((base_r + 6) * self.zoom))
        r_sel_inner = max(4, int(base_r * self.zoom))
        for n in self.model.neurons:
            pos = self.world_to_screen(pygame.Vector2(n.x, n.y))
            color = getattr(n, "_viz_color", (70, 130, 180))
            if n.selected:
                highlight_color = (255, 220, 120)
                pygame.draw.circle(self.screen, highlight_color, (int(pos.x), int(pos.y)), r_sel_outer, 4)
                inner = (min(color[0] + 60, 255), min(color[1] + 60, 255), min(color[2] + 60, 255))
                pygame.draw.circle(self.screen, inner, (int(pos.x), int(pos.y)), r_sel_inner)
            else:
                pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y)), r)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(pos.x), int(pos.y)), r, 2)

    def _draw_grid(self) -> None:
        grid_size = 24
        left = SIDEBAR_WIDTH
        canvas_right = WINDOW_WIDTH - self.right_panel_width
        pixel_spacing = int(grid_size * self.zoom)
        if pixel_spacing <= 0:
            pixel_spacing = 1
        offset_x = int(self.camera_offset.x) % pixel_spacing
        offset_y = int(self.camera_offset.y) % pixel_spacing
        start_x = left + offset_x
        start_y = 0 + offset_y
        for x in range(start_x, canvas_right, pixel_spacing):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(start_y, WINDOW_HEIGHT, pixel_spacing):
            pygame.draw.line(self.screen, self.grid_color, (left, y), (canvas_right, y))

    def _draw_connections(self) -> None:
        # Visualize all connections (input↔hidden↔output)
        conns: List[ConnectionModel] = list(self.model.connections)
        total = len(conns)
        if total == 0:
            self._drawn_connections_last = 0
            return
        # For very large graphs, draw a representative subset to avoid freezing
        if total > MAX_DRAW_CONNECTIONS:
            step = max(1, total // MAX_DRAW_CONNECTIONS)
        else:
            step = 1
        shown = 0
        for idx, conn in enumerate(conns):
            if step > 1 and (idx % step) != 0:
                continue
            p = pygame.Vector2(*self._neuron_screen_pos(conn.source_id))
            q = pygame.Vector2(*self._neuron_screen_pos(conn.target_id))
            # Compute arc offset based on multiplicity
            same_dir = [c for c in conns if c.source_id == conn.source_id and c.target_id == conn.target_id]
            try:
                i = same_dir.index(conn)
            except ValueError:
                i = 0
            n = len(same_dir)
            vec = q - p
            length = vec.length()
            if length == 0:
                continue
            perp = pygame.Vector2(-vec.y, vec.x)
            if perp.length() == 0:
                continue
            perp = perp.normalize()
            if n % 2 == 1:
                centered = i - (n // 2)
            else:
                centered = (i - (n / 2 - 0.5))
            base_offset = 14.0
            offset = perp * base_offset * centered
            opp_dir = [c for c in conns if c.source_id == conn.target_id and c.target_id == conn.source_id]
            if len(opp_dir) > 0:
                offset += perp * 10.0
            control = (p + q) * 0.5 + offset

            color = self._color_for_weight(conn.weight)
            if conn.selected:
                self._draw_quadratic_curve(self.screen, (220, 220, 230), p, control, q, width=6)
            # For hidden-layer visualization, enforce convention: 0->grey, 1->green
            # Map weight in [-1,1] to color scale where 0 is grey and positive moves to green
            if conn.weight >= 0.0:
                t = max(0.0, min(1.0, conn.weight))
                color = (int(120 + (70 - 120) * t), int(120 + (200 - 120) * t), int(120 + (100 - 120) * t))
            else:
                # Negative weights: fade towards a muted grey-red if needed
                t = max(0.0, min(1.0, -conn.weight))
                color = (int(120 + (220 - 120) * t), int(120 + (120 - 120) * t), int(120 + (120 - 120) * t))
            self._draw_quadratic_curve(self.screen, color, p, control, q, width=3)
            # Arrowhead near target to indicate direction
            tip_t = 0.92
            tip = self._bezier_point(p, control, q, tip_t)
            tangent = self._bezier_tangent(p, control, q, tip_t)
            if tangent.length_squared() > 0.0001:
                self._draw_arrowhead(self.screen, color, tip, tangent)
            shown += 1
        self._drawn_connections_last = shown

    def _neuron_screen_pos(self, neuron_id: int) -> Tuple[int, int]:
        n = self.model.find_neuron(neuron_id)
        if n is None:
            return 0, 0
        pos = self.world_to_screen(pygame.Vector2(n.x, n.y))
        return int(pos.x), int(pos.y)

    def world_to_screen(self, world: pygame.Vector2) -> pygame.Vector2:
        return world * self.zoom + self.camera_offset

    def screen_to_world(self, screen: pygame.Vector2) -> pygame.Vector2:
        if self.zoom == 0:
            return pygame.Vector2(screen)
        return (screen - self.camera_offset) / self.zoom

    def _apply_zoom(self, factor: float, anchor_screen: pygame.Vector2) -> None:
        old_zoom = self.zoom
        new_zoom = max(0.25, min(4.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        # Keep the world point under the mouse stationary in screen space
        anchor_world = self.screen_to_world(anchor_screen)
        self.zoom = new_zoom
        self.camera_offset = anchor_screen - anchor_world * self.zoom

    def _draw_quadratic_curve(
        self,
        surface: pygame.Surface,
        color: Tuple[int, int, int],
        p0: pygame.Vector2,
        p1: pygame.Vector2,
        p2: pygame.Vector2,
        width: int = 2,
        segments: int = 32,
    ) -> None:
        last = p0
        for s in range(1, segments + 1):
            t = s / segments
            one_t = 1.0 - t
            point = one_t * one_t * p0 + 2 * one_t * t * p1 + t * t * p2
            pygame.draw.line(surface, color, (int(last.x), int(last.y)), (int(point.x), int(point.y)), width)
            last = point

    def _bezier_point(
        self,
        p0: pygame.Vector2,
        p1: pygame.Vector2,
        p2: pygame.Vector2,
        t: float,
    ) -> pygame.Vector2:
        one_t = 1.0 - t
        return one_t * one_t * p0 + 2 * one_t * t * p1 + t * t * p2

    def _bezier_tangent(
        self,
        p0: pygame.Vector2,
        p1: pygame.Vector2,
        p2: pygame.Vector2,
        t: float,
    ) -> pygame.Vector2:
        # Derivative of quadratic Bezier: 2(1-t)(p1-p0) + 2t(p2-p1)
        return 2 * (1.0 - t) * (p1 - p0) + 2 * t * (p2 - p1)

    def _draw_arrowhead(
        self,
        surface: pygame.Surface,
        color: Tuple[int, int, int],
        tip: pygame.Vector2,
        direction: pygame.Vector2,
        size: float = 10.0,
    ) -> None:
        dir_norm = direction.normalize()
        base = tip - dir_norm * size
        # Perpendicular vector for wings
        perp = pygame.Vector2(-dir_norm.y, dir_norm.x)
        wing = perp * (size * 0.45)
        left = base + wing
        right = base - wing
        points = [(int(tip.x), int(tip.y)), (int(left.x), int(left.y)), (int(right.x), int(right.y))]
        pygame.draw.polygon(surface, color, points)

    def _color_for_weight(self, weight: float) -> Tuple[int, int, int]:
        w = max(-1.0, min(1.0, weight))
        if w < 0:
            t = (w + 1.0)
            r = int(220 * (1.0 - t) + 150 * t)
            g = int(70 * (1.0 - t) + 150 * t)
            b = int(70 * (1.0 - t) + 160 * t)
            return (r, g, b)
        else:
            t = w
            r = int(150 * (1.0 - t) + 70 * t)
            g = int(150 * (1.0 - t) + 200 * t)
            b = int(160 * (1.0 - t) + 100 * t)
            return (r, g, b)

    def _find_connection_under_mouse(self, mouse_pos: Tuple[int, int]) -> Optional[ConnectionModel]:
        # Hit-test among all connections to match visualization
        conns: List[ConnectionModel] = list(self.model.connections)
        total = len(conns)
        if total == 0:
            return None
        # Sample for performance if needed
        step = max(1, total // MAX_DRAW_CONNECTIONS) if total > MAX_DRAW_CONNECTIONS else 1
        # Iterate reverse for top-most
        for idx in range(len(conns) - 1, -1, -1):
            if step > 1 and (idx % step) != 0:
                continue
            conn = conns[idx]
            p = pygame.Vector2(*self._neuron_screen_pos(conn.source_id))
            q = pygame.Vector2(*self._neuron_screen_pos(conn.target_id))
            vec = q - p
            length = vec.length()
            if length == 0:
                continue
            perp = pygame.Vector2(-vec.y, vec.x)
            if perp.length() == 0:
                continue
            perp = perp.normalize()

            same_dir = [c for c in conns if c.source_id == conn.source_id and c.target_id == conn.target_id]
            try:
                i = same_dir.index(conn)
            except ValueError:
                i = 0
            n = len(same_dir)
            if n % 2 == 1:
                centered = i - (n // 2)
            else:
                centered = (i - (n / 2 - 0.5))
            base_offset = 14.0
            offset = perp * base_offset * centered
            opp_dir = [c for c in conns if c.source_id == conn.target_id and c.target_id == conn.source_id]
            if len(opp_dir) > 0:
                offset += perp * 10.0
            control = (p + q) * 0.5 + offset
            if self._is_mouse_near_curve(mouse_pos, p, control, q):
                return conn
        return None

    def _is_mouse_near_curve(
        self,
        mouse_pos: Tuple[int, int],
        p0: pygame.Vector2,
        p1: pygame.Vector2,
        p2: pygame.Vector2,
        threshold: float = 6.0,
    ) -> bool:
        last = p0
        segments = 40
        mouse = pygame.Vector2(mouse_pos)
        for s in range(1, segments + 1):
            t = s / segments
            one_t = 1.0 - t
            point = one_t * one_t * p0 + 2 * one_t * t * p1 + t * t * p2
            if self._point_to_segment_distance(mouse, last, point) <= threshold:
                return True
            last = point
        return False

    def _point_to_segment_distance(
        self,
        p: pygame.Vector2,
        a: pygame.Vector2,
        b: pygame.Vector2,
    ) -> float:
        ab = b - a
        if ab.length_squared() == 0:
            return p.distance_to(a)
        t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared()))
        proj = a + t * ab
        return p.distance_to(proj)

    def _get_selected_weight_button_rects(self) -> List[Tuple[str, pygame.Rect]]:
        if self.selected_connection is None or not self._selected_connection_is_valid():
            return []
        p = pygame.Vector2(*self._neuron_screen_pos(self.selected_connection.source_id))
        q = pygame.Vector2(*self._neuron_screen_pos(self.selected_connection.target_id))
        mid = (p + q) * 0.5
        labels = ["-1", "0", "1"]
        size = pygame.Vector2(28, 20)
        spacing = 6
        total_w = size.x * 3 + spacing * 2
        start = pygame.Vector2(mid.x - total_w / 2, mid.y - 40)
        rects: List[Tuple[str, pygame.Rect]] = []
        for i, label in enumerate(labels):
            r = pygame.Rect(int(start.x + i * (size.x + spacing)), int(start.y), int(size.x), int(size.y))
            rects.append((label, r))
        return rects

    def _add_message(self, text: str) -> None:
        self.messages.append(Message(text=text, created_ms=pygame.time.get_ticks()))

    def _draw_messages(self) -> None:
        now = pygame.time.get_ticks()
        lifetime_ms = 2000
        fade_ms = 600
        self.messages = [m for m in self.messages if now - m.created_ms < lifetime_ms]
        draw_y = WINDOW_HEIGHT - 20
        for m in reversed(self.messages[-3:]):
            age = now - m.created_ms
            alpha_scale = 1.0
            if age > lifetime_ms - fade_ms:
                alpha_scale = max(0.0, (lifetime_ms - age) / fade_ms)
            base_color = (220, 220, 230)
            color = tuple(int(c * alpha_scale) for c in base_color)
            text_surface = self.font.render(m.text, True, color)
            self.screen.blit(text_surface, (12, draw_y))
            draw_y -= 18

    def _draw_stats(self) -> None:
        # Sidebar stats at bottom above messages area
        max_msgs = 3
        message_lines = min(len(self.messages), max_msgs)
        message_block_h = message_lines * 18 + 24  # messages area + padding
        bottom = WINDOW_HEIGHT - message_block_h
        y = bottom - 8  # small padding above messages

        # Collect stats
        num_neurons = len(self.model.neurons)
        num_connections = len(self.model.connections)
        selected_neurons = sum(1 for n in self.model.neurons if n.selected)
        selected_conns = sum(1 for c in self.model.connections if c.selected)

        lines = [
            "Stats",
            f"Neurons: {num_neurons}",
            f"Connections: {num_connections} (shown: {getattr(self, '_drawn_connections_last', 0)})",
            f"Selected N/C: {selected_neurons}/{selected_conns}",
        ]

        # Measure height
        line_h = 18
        total_h = len(lines) * line_h + 8
        top = y - total_h

        # Background panel
        panel_rect = pygame.Rect(8, max(40, int(top)), SIDEBAR_WIDTH - 16, int(total_h))
        pygame.draw.rect(self.screen, (40, 40, 48), panel_rect, border_radius=6)
        pygame.draw.rect(self.screen, (60, 60, 70), panel_rect, width=1, border_radius=6)

        # Render lines
        text_y = panel_rect.top + 6
        for i, text in enumerate(lines):
            color = (210, 210, 220) if i == 0 else (190, 190, 200)
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (panel_rect.left + 8, text_y))
            text_y += line_h

    def _draw_selected_weight_buttons(self) -> None:
        rects = self._get_selected_weight_button_rects()
        if not rects:
            return
        mouse_pos = pygame.mouse.get_pos()
        for label, rect in rects:
            is_hover = rect.collidepoint(mouse_pos)
            bg = (60, 60, 70) if not is_hover else (85, 85, 100)
            pygame.draw.rect(self.screen, bg, rect, border_radius=4)
            text = self.font.render(label, True, (230, 230, 240))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def _selected_connection_is_valid(self) -> bool:
        conn = self.selected_connection
        if conn is None:
            return False
        if conn not in self.model.connections:
            return False
        if self.model.find_neuron(conn.source_id) is None:
            return False
        if self.model.find_neuron(conn.target_id) is None:
            return False
        return True



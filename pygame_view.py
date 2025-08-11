from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

from graph_model import GraphModel, NeuronModel, ConnectionModel
from simulator import IzhikevichSimulator, SimulationConfig


SIDEBAR_WIDTH = 200
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
FPS = 60


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

        # Sim UI state
        self._steps_per_click: int = 10
        self._dt_step: float = 0.5
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
            make_button("Neuron"),  # draggable palette item
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
        canvas_rect = pygame.Rect(SIDEBAR_WIDTH, 0, WINDOW_WIDTH - SIDEBAR_WIDTH, WINDOW_HEIGHT)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Simulate controls first
            if self.sim_advance_rect.collidepoint(mouse_pos):
                self.sim.advance_steps(self._steps_per_click)
                self._add_message(f"advanced {self._steps_per_click} steps")
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
            # Weight buttons for selected connection
            for label, rect in self._get_selected_weight_button_rects():
                if rect.collidepoint(mouse_pos) and self.selected_connection is not None:
                    new_weight = -1.0 if label == "-1" else (0.0 if label == "0" else 1.0)
                    self.selected_connection.weight = max(-1.0, min(1.0, new_weight))
                    self._add_message(f"weight set to {int(self.selected_connection.weight)}")
                    return

            # Start dragging neuron from palette if clicked on the neuron button
            neuron_button = self.buttons[0]
            if neuron_button.rect.collidepoint(mouse_pos):
                self.dragging_palette_item = "Neuron"
                self.drag_position = mouse_pos_v
                return

            # Delete button acts on selected items
            delete_button = self.buttons[1]
            if delete_button.rect.collidepoint(mouse_pos):
                self.model.delete_selected()
                if self.selected_connection is not None and self.selected_connection.selected:
                    self.selected_connection = None
                return

            # Otherwise, canvas interactions
            if canvas_rect.collidepoint(mouse_pos):
                world_mouse = mouse_pos_v - self.camera_offset
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
            if self.dragging_palette_item == "Neuron":
                if canvas_rect.collidepoint(mouse_pos):
                    world_mouse = mouse_pos_v - self.camera_offset
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
                world_mouse = mouse_pos_v - self.camera_offset
                for n in reversed(self.model.neurons):
                    if self._neuron_hit_test(n, world_mouse):
                        self.dragging_connection_from = n
                        self.dragging_connection_pos = world_mouse
                        break

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
            # Finish creating a connection if dropped over a neuron
            if self.dragging_connection_from is not None:
                world_mouse = mouse_pos_v - self.camera_offset
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
            if self.dragging_palette_item == "Neuron":
                self.drag_position = mouse_pos_v
            if self.dragging_existing_neuron is not None:
                world_mouse = mouse_pos_v - self.camera_offset
                self.dragging_existing_neuron.x = world_mouse.x + self.drag_offset.x
                self.dragging_existing_neuron.y = world_mouse.y + self.drag_offset.y
            elif self.is_panning:
                delta = mouse_pos_v - self.pan_start_mouse
                self.camera_offset = self.pan_start_camera + delta
            if self.dragging_connection_from is not None:
                self.dragging_connection_pos = mouse_pos_v - self.camera_offset

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

        # Sidebar
        pygame.draw.rect(self.screen, self.sidebar_color, (0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        title_surface = self.font.render("Palette", True, (200, 200, 210))
        self.screen.blit(title_surface, (16, 10))

        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(self.screen, self.font, mouse_pos)
        # Simulate panel UI
        self._draw_simulate_panel()
        # Neuron overview panel (if a neuron is selected)
        self._draw_neuron_overview()

        # Grid and canvas
        self._draw_grid()
        self._draw_connections()
        self._draw_neurons()

        # Palette drag preview
        if self.dragging_palette_item == "Neuron":
            preview_color = (120, 180, 220)
            pygame.draw.circle(self.screen, preview_color, (int(self.drag_position.x), int(self.drag_position.y)), 18)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(self.drag_position.x), int(self.drag_position.y)), 18, 2)

        # Connection drag preview
        if self.dragging_connection_from is not None:
            start = pygame.Vector2(self.dragging_connection_from.x, self.dragging_connection_from.y) + self.camera_offset
            end = self.dragging_connection_pos + self.camera_offset
            pygame.draw.line(self.screen, (120, 120, 130), (int(start.x), int(start.y)), (int(end.x), int(end.y)), 2)

        # Messages and selected connection controls
        self._draw_stats()
        self._draw_messages()
        self._draw_selected_weight_buttons()

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

    def _draw_neuron_overview(self) -> None:
        neuron = self._get_selected_neuron()
        if neuron is None:
            return
        rect = self.neuron_overview_rect
        # Panel background
        pygame.draw.rect(self.screen, (40, 40, 48), rect, border_radius=6)
        pygame.draw.rect(self.screen, (60, 60, 70), rect, width=1, border_radius=6)

        # Title and params
        title = self.font.render(f"Neuron {neuron.id}", True, (210, 210, 220))
        self.screen.blit(title, (rect.left + 8, rect.top + 6))
        param_y = rect.top + 24
        params = f"a={neuron.a:.2f}  b={neuron.b:.2f}  c={neuron.c:.0f}  d={neuron.d:.0f}"
        p_text = self.font.render(params, True, (190, 190, 200))
        self.screen.blit(p_text, (rect.left + 8, param_y))

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

        # Nullclines
        I = self.sim.config.input_current
        # u = b v (du/dt = 0)
        v_samples = [v_min + (v_max - v_min) * i / 60.0 for i in range(61)]
        points_du0 = [vu_to_px(v, neuron.b * v) for v in v_samples]
        pygame.draw.lines(self.screen, (140, 200, 230), False, points_du0, 1)
        # u = 0.04 v^2 + 5 v + 140 + I (dv/dt = 0)
        points_dv0 = [vu_to_px(v, 0.04 * v * v + 5 * v + 140.0 + I) for v in v_samples]
        pygame.draw.lines(self.screen, (230, 140, 140), False, points_dv0, 1)

        # Vector field (sparse)
        grid_cols, grid_rows = 12, 8
        for gi in range(grid_cols + 1):
            for gj in range(grid_rows + 1):
                v = v_min + (v_max - v_min) * gi / grid_cols
                u = u_min + (u_max - u_min) * gj / grid_rows
                dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
                du = neuron.a * (neuron.b * v - u)
                # Normalize
                mag = (dv * dv + du * du) ** 0.5
                if mag == 0:
                    continue
                dvn, dun = dv / mag, du / mag
                cx, cy = vu_to_px(v, u)
                length = 8
                ex = int(cx + dvn * length)
                ey = int(cy - dun * length)  # minus because screen y increases downward
                pygame.draw.line(self.screen, (120, 120, 130), (cx, cy), (ex, ey), 1)

        # Current state point
        px, py = vu_to_px(neuron.v, neuron.u)
        color = (240, 230, 140) if neuron.spiked else (200, 220, 120)
        pygame.draw.circle(self.screen, color, (px, py), 3)

    def _draw_neurons(self) -> None:
        for n in self.model.neurons:
            pos = pygame.Vector2(n.x, n.y) + self.camera_offset
            color = (70, 130, 180)
            if n.selected:
                highlight_color = (255, 220, 120)
                pygame.draw.circle(self.screen, highlight_color, (int(pos.x), int(pos.y)), 24, 4)
                inner = (min(color[0] + 60, 255), min(color[1] + 60, 255), min(color[2] + 60, 255))
                pygame.draw.circle(self.screen, inner, (int(pos.x), int(pos.y)), 18)
            else:
                pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y)), 18)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(pos.x), int(pos.y)), 18, 2)

    def _draw_grid(self) -> None:
        grid_size = 24
        left = SIDEBAR_WIDTH
        offset_x = int(self.camera_offset.x) % grid_size
        offset_y = int(self.camera_offset.y) % grid_size
        start_x = left + offset_x
        start_y = 0 + offset_y
        for x in range(start_x, WINDOW_WIDTH, grid_size):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(start_y, WINDOW_HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.grid_color, (left, y), (WINDOW_WIDTH, y))

    def _draw_connections(self) -> None:
        for conn in self.model.connections:
            p = pygame.Vector2(*self._neuron_screen_pos(conn.source_id))
            q = pygame.Vector2(*self._neuron_screen_pos(conn.target_id))
            # Compute arc offset based on multiplicity
            same_dir = [c for c in self.model.connections if c.source_id == conn.source_id and c.target_id == conn.target_id]
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
            opp_dir = [c for c in self.model.connections if c.source_id == conn.target_id and c.target_id == conn.source_id]
            if len(opp_dir) > 0:
                offset += perp * 10.0
            control = (p + q) * 0.5 + offset

            color = self._color_for_weight(conn.weight)
            if conn.selected:
                self._draw_quadratic_curve(self.screen, (220, 220, 230), p, control, q, width=6)
            self._draw_quadratic_curve(self.screen, color, p, control, q, width=3)

    def _neuron_screen_pos(self, neuron_id: int) -> Tuple[int, int]:
        n = self.model.find_neuron(neuron_id)
        assert n is not None
        pos = pygame.Vector2(n.x, n.y) + self.camera_offset
        return int(pos.x), int(pos.y)

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
        # Iterate reverse for top-most
        for conn in reversed(self.model.connections):
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

            same_dir = [c for c in self.model.connections if c.source_id == conn.source_id and c.target_id == conn.target_id]
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
            opp_dir = [c for c in self.model.connections if c.source_id == conn.target_id and c.target_id == conn.source_id]
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
        if self.selected_connection is None:
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
            f"Connections: {num_connections}",
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



"""
skeleton_3d.py — interactive 3D skeleton viewer backed by pyqtgraph/OpenGL.

Displays COCO-17 pose keypoints in 3D for the current video frame.
Supports two simultaneous 3D sources (e.g. 4D-Humans + TRAM) with
distinct colour schemes and an on/off toggle for the alt skeleton.

Primary source (green/red) — loaded via load_player().
Alt source     (cyan/orange) — loaded via load_alt_player(); toggle with the
                                "Alt" button or programmatically via set_alt_visible().

Data source: any Ravens pose JSON with keypoints_3d field.

Coordinate convention (input data):
  - Y is DOWN in camera space (image convention: Y increases downward,
    head at small Y, feet at large Y). Stored keypoints_3d are Y-down.
  - Units: metres (typical range ±0.5 m around the hip root)

Rendering pipeline (applied in _render per frame):
  1. Centre on mid-hip (COCO 11=L-hip, 12=R-hip).
  2. Negate Y  →  Y-up world space (head +Y, feet −Y).
  3. If Sagittal view: swap X↔Z so depth becomes horizontal.
  4. Swap Y↔Z  →  pyqtgraph Z-up display space (pyqtgraph's GLGridItem
     lies in the XY plane with Z as the vertical axis).

Camera presets use elevation=0, azimuth=−90 for the front/side views.
pyqtgraph viewMatrix = R_{−Z}(azimuth+90) · Rx(elevation−90), so
  elevation=0, azimuth=−90  →  camera looks along +pg_Y (world depth),
  screen_right = pg_X (world X), screen_up = pg_Z (world Y after swap).
This matches the 2D image: X right, Y up, Z into screen.

Usage:
    panel = Skeleton3DPanel()
    panel.load_player(frames_3d)      # dict[video_frame_idx → np.ndarray (17,3)]
    panel.update_frame(frame_idx)     # called on every seek / playback tick
"""

import math
import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
)

try:
    import pyqtgraph.opengl as gl
    _GL_OK = True
except ImportError:
    _GL_OK = False

from constants import COCO_SKELETON, LEFT_KPS, RIGHT_KPS

# ── Colour palette (RGBA float32, mirrors 2D editor conventions) ───────────────

_C_LEFT   = (0.20, 0.85, 0.20, 1.0)   # green  — left side  (primary)
_C_RIGHT  = (0.85, 0.20, 0.20, 1.0)   # red    — right side (primary)
_C_CENTER = (0.85, 0.80, 0.20, 1.0)   # amber  — centre     (primary)
_C_GRID   = (0.31, 0.31, 0.31, 0.70)

# Alt skeleton colours (cyan/orange scheme)
_C_ALT_LEFT   = (0.10, 0.80, 0.90, 0.85)   # cyan
_C_ALT_RIGHT  = (0.95, 0.55, 0.10, 0.85)   # orange
_C_ALT_CENTER = (0.85, 0.85, 0.85, 0.85)   # light grey


def _joint_colors() -> np.ndarray:
    """Return (17, 4) float32 RGBA per joint."""
    c = np.zeros((17, 4), dtype=np.float32)
    for i in range(17):
        c[i] = _C_LEFT if i in LEFT_KPS else (_C_RIGHT if i in RIGHT_KPS else _C_CENTER)
    return c


def _bone_color(a: int, b: int) -> tuple:
    if a in LEFT_KPS and b in LEFT_KPS:
        return _C_LEFT
    if a in RIGHT_KPS and b in RIGHT_KPS:
        return _C_RIGHT
    return _C_CENTER


def _build_bone_colors() -> np.ndarray:
    """Return (len(COCO_SKELETON)*2, 4) colour array for GLLinePlotItem pairs."""
    n = len(COCO_SKELETON)
    c = np.zeros((n * 2, 4), dtype=np.float32)
    for i, (a, b) in enumerate(COCO_SKELETON):
        col = _bone_color(a, b)
        c[i * 2] = c[i * 2 + 1] = col
    return c


def _alt_joint_colors() -> np.ndarray:
    """Return (17, 4) float32 RGBA per joint for the alt skeleton."""
    c = np.zeros((17, 4), dtype=np.float32)
    for i in range(17):
        c[i] = _C_ALT_LEFT if i in LEFT_KPS else (
            _C_ALT_RIGHT if i in RIGHT_KPS else _C_ALT_CENTER)
    return c


def _alt_bone_color(a: int, b: int) -> tuple:
    if a in LEFT_KPS and b in LEFT_KPS:
        return _C_ALT_LEFT
    if a in RIGHT_KPS and b in RIGHT_KPS:
        return _C_ALT_RIGHT
    return _C_ALT_CENTER


def _build_alt_bone_colors() -> np.ndarray:
    n = len(COCO_SKELETON)
    c = np.zeros((n * 2, 4), dtype=np.float32)
    for i, (a, b) in enumerate(COCO_SKELETON):
        col = _alt_bone_color(a, b)
        c[i * 2] = c[i * 2 + 1] = col
    return c


# Precomputed constants — built once at import time
_JOINT_COLORS     = _joint_colors()
_BONE_COLORS      = _build_bone_colors()
_ALT_JOINT_COLORS = _alt_joint_colors()
_ALT_BONE_COLORS  = _build_alt_bone_colors()
_N_BONES          = len(COCO_SKELETON)


# ── AxisGizmo ──────────────────────────────────────────────────────────────────

class AxisGizmo(QWidget):
    """Small widget showing projected XYZ axis directions via QPainter."""
    SIZE = 70

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self._az = -90.0
        self._el = 0.0
        self.setAttribute(Qt.WA_TranslucentBackground)

    def update_camera(self, az: float, el: float):
        if az != self._az or el != self._el:
            self._az, self._el = az, el
            self.update()

    def _axis_screen_dirs(self):
        """(3,2) float32 — 2D screen direction of each world axis (X,Y,Z)."""
        az = math.radians(self._az + 90)
        el = math.radians(self._el - 90)
        # pyqtgraph viewMatrix = Rx(el-90) · R_{-Z}(az+90)
        # R_{-Z}(az+90) ≡ Rz(-az)  where az = az_deg+90
        Rz = np.array([[ math.cos(az),  math.sin(az), 0],
                       [-math.sin(az),  math.cos(az), 0],
                       [ 0,             0,            1]], dtype=np.float64)
        Rx = np.array([[1, 0,              0           ],
                       [0, math.cos(el), -math.sin(el)],
                       [0, math.sin(el),  math.cos(el)]], dtype=np.float64)
        R = Rx @ Rz
        cam = (R @ np.eye(3).T).T           # (3,3): row i = camera-space unit axis i
        return np.column_stack([cam[:, 0], -cam[:, 1]]).astype(np.float32)  # flip Y for widget

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx = cy = self.SIZE // 2
        r = cx - 10

        # Background circle
        painter.setBrush(QBrush(QColor(18, 18, 18, 210)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx - r - 6, cy - r - 6, (r + 6) * 2, (r + 6) * 2)

        dirs = self._axis_screen_dirs()   # (3,2)
        labels  = ['X', 'Y', 'Z']
        colors  = [QColor(220, 80, 80), QColor(80, 210, 80), QColor(80, 130, 220)]

        # Compute camera-depth of each axis tip to draw back→front
        az = math.radians(self._az + 90)
        el = math.radians(self._el - 90)
        Rz = np.array([[ math.cos(az), math.sin(az), 0],
                       [-math.sin(az), math.cos(az), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, math.cos(el), -math.sin(el)],
                       [0, math.sin(el),  math.cos(el)]])
        R  = Rx @ Rz
        depths = [(R @ np.eye(3)[i])[2] for i in range(3)]
        order  = np.argsort(depths)   # back→front

        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        for i in order:
            dx, dy = float(dirs[i, 0]), float(dirs[i, 1])
            ex = int(cx + dx * r)
            ey = int(cy + dy * r)
            alpha = int(255 * max(0.3, (depths[i] + 1.0) / 2.0))
            col = QColor(colors[i])
            col.setAlpha(alpha)

            painter.setPen(QPen(col, 2))
            painter.drawLine(cx, cy, ex, ey)

            # Dot at tip
            painter.setBrush(QBrush(col))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(ex - 3, ey - 3, 6, 6)

            # Label
            painter.setPen(QPen(col, 1))
            painter.drawText(ex + int(dx * 7) - 4, ey + int(dy * 7) + 4, labels[i])


# ── Skeleton3DPanel ────────────────────────────────────────────────────────────

class Skeleton3DPanel(QWidget):
    """
    Interactive 3D skeleton panel using pyqtgraph.opengl.GLViewWidget.

    The panel maintains a single GLScatterPlotItem (joints) and a single
    GLLinePlotItem (bones as interleaved vertex pairs).  On each frame seek
    only setData() is called — no GL items are created or destroyed — keeping
    update time in the low-millisecond range even during live playback.

    Camera presets
    --------------
    Isometric  — default 3/4 view; bones are easy to distinguish
    Sagittal   — side view (Z-Y plane): depth vs height
    Frontal    — front view (X-Y plane): lateral spread vs height
    Top        — bird's eye (X-Z plane): lateral vs depth
    Free       — user rotates with mouse drag (no preset applied)
    """

    # Camera preset: (distance_m, elevation_deg, azimuth_deg, fov_deg)
    #
    # pyqtgraph viewMatrix = R_{-Z}(azimuth+90) · Rx(elevation-90)
    #   After the Y↔Z swap in _render, world-Y becomes pg_Z (up) and
    #   world-Z becomes pg_Y (depth).
    #
    #   elevation=0, azimuth=-90:
    #     cam looks along +pg_Y (= world depth Z), screen_right=pg_X (=world X),
    #     screen_up=pg_Z (=world Y).  This matches the 2D image exactly.
    #
    #   elevation=90, azimuth=0:
    #     cam looks straight down along -pg_Z  → top / bird's-eye view.
    #
    # Sagittal swaps X↔Z in the data (before the Y↔Z swap) so the depth axis
    # is shown on the horizontal screen axis.
    #
    # Frontal/Sagittal use large distance + narrow FOV (telephoto ≈ orthographic).
    # This makes the frontal projection match the 2D image positions: at d→∞
    # with proportionally small FOV, perspective distortion → 0, and the
    # XZ plane projection becomes identical to the original camera's flat image.
    # fov=None → use _compute_ortho_fov(dist) which preserves apparent skeleton size.
    _PRESETS = {
        "Frontal":   (30.0,  0.0, -90.0, None),  # orthographic-like: matches 2D image
        "Sagittal":  (30.0,  0.0, -90.0, None),  # same camera; X↔Z swapped in data
        "Isometric": ( 2.5, 30.0,  45.0, 60.0),  # normal perspective
        "Top":       ( 2.5, 90.0,   0.0, 60.0),  # bird's-eye
        "Free":      None,
    }

    # Trunk + limb joints used for camera auto-scaling.
    # Head joints (0–4: nose, eyes, ears) are excluded because 3D lifting
    # often produces large depth errors for face keypoints from side-on cameras.
    _SCALE_JOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames_3d:     dict[int, np.ndarray] = {}
        self._frames_3d_alt: dict[int, np.ndarray] = {}
        self._alt_visible:   bool = True
        self._current_frame: int = -1
        self._bone_pts     = np.zeros((_N_BONES * 2, 3), dtype=np.float32)
        self._bone_pts_alt = np.zeros((_N_BONES * 2, 3), dtype=np.float32)
        self._label_items: list = []
        self._current_view: str = "Frontal"

        # Animation state
        self._auto_dist: float = 2.0
        # X-axis scale correction: lift_poses_3d.py normalises x by W/2 and y by H/2
        # independently, but MotionAGFormer expects both normalised by the same value
        # (W/2 for landscape video).  This makes y 1.78× too large for 1280×720 video,
        # compressing the skeleton horizontally.  Multiply X by W/H to restore proportions.
        self._x_scale_corr: float = 1280.0 / 720.0   # = W / H for Ravens 1280×720 source
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_step)
        self._anim_start  = (2.0,  0.0, -90.0, 60.0)  # (dist, el, az, fov)
        self._anim_target = (2.0,  0.0, -90.0, 60.0)  # (dist, el, az, fov)
        self._anim_t: float = 0.0
        self._anim_pending_view: str | None = None

        self._setup_ui()

        # Camera polling timer (Feature A + C)
        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self._poll_camera)
        self._cam_timer.start(80)

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar (28 px)
        header = QWidget()
        header.setFixedHeight(28)
        header.setStyleSheet("background:#1e1e1e; border-bottom:1px solid #444;")
        h = QHBoxLayout(header)
        h.setContentsMargins(8, 0, 8, 0)

        lbl = QLabel("3D Skeleton")
        lbl.setStyleSheet("color:#d4d4d4; font-size:12px; font-weight:bold;")
        h.addWidget(lbl)

        # Alt source legend dots
        self._legend_lbl = QLabel(
            '<span style="color:#33cccc;">■</span> Primary &nbsp;'
            '<span style="color:#f28a20;">■</span> Alt'
        )
        self._legend_lbl.setStyleSheet("color:#aaa; font-size:11px; padding-left:8px;")
        self._legend_lbl.setVisible(False)
        h.addWidget(self._legend_lbl)

        h.addStretch()

        # Alt toggle button
        self._alt_btn = QPushButton("Alt: ON")
        self._alt_btn.setFixedHeight(20)
        self._alt_btn.setCheckable(True)
        self._alt_btn.setChecked(True)
        self._alt_btn.setVisible(False)
        self._alt_btn.setStyleSheet(
            "background:#0e639c; color:#fff; border:none; border-radius:2px; "
            "padding:0 6px; font-size:11px;"
        )
        self._alt_btn.clicked.connect(self._on_alt_toggle)
        h.addWidget(self._alt_btn)

        # Feature A — Az/El text readout
        self._angle_label = QLabel("Az  +0°  El  +0°")
        self._angle_label.setStyleSheet(
            "color:#888; font-size:11px; font-family:monospace;"
        )
        self._angle_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(self._angle_label)

        layout.addWidget(header)

        # Feature B — Preset button bar (74 px)
        btn_bar = QWidget()
        btn_bar.setFixedHeight(74)
        btn_bar.setStyleSheet("background:#252526; border-bottom:1px solid #333;")
        bb = QHBoxLayout(btn_bar)
        bb.setContentsMargins(8, 0, 8, 0)
        bb.setSpacing(4)

        self._preset_buttons: dict[str, QPushButton] = {}
        for name in self._PRESETS:
            btn = QPushButton(name)
            btn.setFixedHeight(28)
            btn.setCheckable(False)
            btn.clicked.connect(lambda checked, n=name: self._on_preset_btn(n))
            self._preset_buttons[name] = btn
            bb.addWidget(btn)

        bb.addStretch()

        # Feature C — Axis gizmo (right end of button bar)
        self._gizmo = AxisGizmo()
        bb.addWidget(self._gizmo)

        layout.addWidget(btn_bar)

        # Style the initial active button
        self._set_active_button("Frontal")

        if not _GL_OK:
            err = QLabel(
                "pyqtgraph.opengl not available.\n"
                "Install with:  pip install pyqtgraph PyOpenGL PyOpenGL_accelerate")
            err.setAlignment(Qt.AlignCenter)
            err.setStyleSheet("color:#f84; font-size:12px; padding:16px;")
            layout.addWidget(err)
            return

        # No-data placeholder (shown when player has no 3D poses)
        self._no_data = QLabel(
            "No 3D keypoints for this player.\n"
            "Select a 3D model (e.g. poses_3d_motionagformer, poses_3d_4dhumans)\n"
            "or configure a 3D Poses Folder via File \u2192 New Session.\n"
            "4D-Humans: run src/pose_extraction/convert_4dhumans.py first.")
        self._no_data.setAlignment(Qt.AlignCenter)
        self._no_data.setStyleSheet("color:#555; font-size:12px; padding:20px;")
        layout.addWidget(self._no_data)

        # OpenGL viewport
        self._gl = gl.GLViewWidget()
        self._gl.setBackgroundColor((30, 30, 30, 255))
        self._gl.setMinimumHeight(480)
        self._gl.hide()
        layout.addWidget(self._gl, 1)

        # Scene items
        grid = gl.GLGridItem()
        grid.setSize(3, 3)
        grid.setSpacing(0.5, 0.5)
        grid.setColor(_C_GRID)
        self._gl.addItem(grid)

        axes = gl.GLAxisItem()
        axes.setSize(0.25, 0.25, 0.25)
        self._gl.addItem(axes)

        # Feature D — Axis labels in GL scene
        for text, pos in [("X", (0.30, 0, 0)), ("Y", (0, 0.30, 0)), ("Z", (0, 0, 0.30))]:
            t = gl.GLTextItem(pos=np.array(pos, dtype=np.float32), text=text,
                              color=(200, 200, 200, 180))
            self._gl.addItem(t)

        self._joint_item = gl.GLScatterPlotItem(
            pos=np.zeros((17, 3), dtype=np.float32),
            color=_JOINT_COLORS,
            size=8.0,
            pxMode=True,
        )
        self._gl.addItem(self._joint_item)

        # All bones as interleaved pairs in one GLLinePlotItem (mode='lines')
        self._bone_item = gl.GLLinePlotItem(
            pos=self._bone_pts.copy(),
            color=_BONE_COLORS,
            width=2.0,
            antialias=True,
            mode='lines',
        )
        self._gl.addItem(self._bone_item)

        # ── Alt skeleton GL items (hidden until load_alt_player() called) ──────
        self._alt_joint_item = gl.GLScatterPlotItem(
            pos=np.zeros((17, 3), dtype=np.float32),
            color=_ALT_JOINT_COLORS,
            size=6.0,
            pxMode=True,
        )
        self._alt_joint_item.setVisible(False)
        self._gl.addItem(self._alt_joint_item)

        self._alt_bone_item = gl.GLLinePlotItem(
            pos=self._bone_pts_alt.copy(),
            color=_ALT_BONE_COLORS,
            width=1.5,
            antialias=True,
            mode='lines',
        )
        self._alt_bone_item.setVisible(False)
        self._gl.addItem(self._alt_bone_item)

        # Per-joint index labels
        self._label_items: list = []
        for i in range(17):
            t = gl.GLTextItem(
                pos=np.array([0.0, 0.0, 0.0]),
                text=str(i),
                color=(220, 220, 220, 255),
            )
            self._gl.addItem(t)
            self._label_items.append(t)

        self._current_view = "Frontal"
        self._apply_preset("Frontal")

    # ── Button styling ─────────────────────────────────────────────────────────

    _STYLE_ACTIVE = (
        "background:#0e639c; color:#fff; border:none; border-radius:2px; "
        "padding:0 8px; font-size:11px;"
    )
    _STYLE_INACTIVE = (
        "background:#3c3c3c; color:#ccc; border:none; border-radius:2px; "
        "padding:0 8px; font-size:11px;"
    )

    def _set_active_button(self, name: str):
        for btn_name, btn in self._preset_buttons.items():
            btn.setStyleSheet(
                self._STYLE_ACTIVE if btn_name == name else self._STYLE_INACTIVE
            )

    # ── Data API ──────────────────────────────────────────────────────────────

    def set_labels_visible(self, v: bool):
        for lbl in self._label_items:
            lbl.setVisible(v)

    def load_player(self, frames_3d: dict):
        """
        Set 3D frame data for a player.

        Parameters
        ----------
        frames_3d : dict mapping video_frame_index (int) → np.ndarray (17, 3)
                    Coordinates in metres, Y-down camera space (head at small Y,
                    feet at large Y). _render negates Y for display.
                    Pass None or empty dict to show the no-data placeholder.
        """
        self._frames_3d    = frames_3d or {}
        self._current_frame = -1

        if not _GL_OK:
            return

        if self._frames_3d:
            self._no_data.hide()
            self._gl.show()
            # Auto-scale camera using trunk/limb joints only (head joints
            # excluded — face keypoints have unreliable depth from lifting)
            sample  = next(iter(self._frames_3d.values())).astype(np.float32)
            root    = (sample[11] + sample[12]) / 2.0
            trunk   = sample[self._SCALE_JOINTS] - root
            extent  = float(np.max(np.abs(trunk))) * 2.0
            dist    = float(np.clip(extent * 2.0, 1.0, 4.0))
            self._auto_dist = dist
            preset = self._PRESETS.get(self._current_view)
            if preset:
                dist_raw, p_el, p_az, fov_raw = preset
                fov = self._compute_ortho_fov(dist_raw) if fov_raw is None else fov_raw
                self._gl.opts['fov'] = fov
                self._gl.setCameraPosition(distance=self._auto_dist,
                                           elevation=p_el,
                                           azimuth=p_az)
        else:
            self._gl.hide()
            self._no_data.show()

    def load_alt_player(self, frames_3d_alt: dict):
        """
        Load an alternate 3D source (e.g. TRAM) for the current player.

        Parameters
        ----------
        frames_3d_alt : dict mapping frame_idx → np.ndarray (17, 3), or None/{}
                        to clear the alt skeleton.
        """
        self._frames_3d_alt = frames_3d_alt or {}
        self._current_frame = -1   # force re-render

        has_alt = bool(self._frames_3d_alt)
        self._alt_btn.setVisible(has_alt)
        self._legend_lbl.setVisible(has_alt)
        if _GL_OK:
            show = has_alt and self._alt_visible
            self._alt_joint_item.setVisible(show)
            self._alt_bone_item.setVisible(show)

    def set_alt_visible(self, visible: bool):
        """Programmatically show/hide the alt skeleton."""
        self._alt_visible = visible
        self._alt_btn.setChecked(visible)
        self._alt_btn.setText("Alt: ON" if visible else "Alt: OFF")
        self._alt_btn.setStyleSheet(
            ("background:#0e639c;" if visible else "background:#3c3c3c;") +
            " color:#fff; border:none; border-radius:2px; padding:0 6px; font-size:11px;"
        )
        if _GL_OK and self._frames_3d_alt:
            self._alt_joint_item.setVisible(visible)
            self._alt_bone_item.setVisible(visible)

    def _on_alt_toggle(self):
        self.set_alt_visible(self._alt_btn.isChecked())

    def update_frame(self, frame_idx: int):
        """
        Update the 3D display to the given video frame index.
        Safe to call every frame; skips work if already showing this frame.
        """
        if not _GL_OK or not self._frames_3d:
            return
        if frame_idx == self._current_frame:
            return
        self._current_frame = frame_idx
        kps = self._frames_3d.get(frame_idx)
        if kps is not None:
            self._render(kps)
        if self._frames_3d_alt and self._alt_visible:
            kps_alt = self._frames_3d_alt.get(frame_idx)
            if kps_alt is not None:
                self._render_alt(kps_alt)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self, kps: np.ndarray):
        """
        Push new keypoint positions to GL items without re-allocating.

        Coordinate pipeline (see module docstring for full derivation):
          1. Centre on mid-hip (COCO 11=L-hip, 12=R-hip).
          2. Negate Y  → Y-up world space (head +Y, feet -Y).
          3. Sagittal only: swap X↔Z so depth axis reads as horizontal.
          4. Swap Y↔Z  → pyqtgraph Z-up display space.
        """
        pts = kps.astype(np.float32).copy()
        root = (pts[11] + pts[12]) / 2.0
        pts -= root
        pts[:, 0] *= self._x_scale_corr                                 # restore W/H aspect ratio
        pts[:, 1] = -pts[:, 1]                                          # Y-down → Y-up
        if self._current_view == "Sagittal":
            pts[:, 0], pts[:, 2] = pts[:, 2].copy(), pts[:, 0].copy()  # X↔Z
        pts[:, 1], pts[:, 2] = pts[:, 2].copy(), pts[:, 1].copy()      # Y↔Z → pg Z-up

        # Update joint positions
        self._joint_item.setData(pos=pts, color=_JOINT_COLORS, size=8.0, pxMode=True)

        # Update bone positions — write pairs into the pre-allocated buffer
        for i, (a, b) in enumerate(COCO_SKELETON):
            self._bone_pts[i * 2]     = pts[a]
            self._bone_pts[i * 2 + 1] = pts[b]
        self._bone_item.setData(pos=self._bone_pts,
                                color=_BONE_COLORS,
                                width=2.0,
                                antialias=True,
                                mode='lines')

        # Update index labels — offset slightly so they don't overlap the dot
        for i, lbl in enumerate(self._label_items):
            lbl.setData(pos=pts[i] + np.array([0.01, 0.01, 0.0], dtype=np.float32))

    def _render_alt(self, kps: np.ndarray):
        """Push alt-source keypoints to the alt GL items (same coordinate pipeline)."""
        pts = kps.astype(np.float32).copy()
        root = (pts[11] + pts[12]) / 2.0
        pts -= root
        pts[:, 0] *= self._x_scale_corr
        pts[:, 1] = -pts[:, 1]
        if self._current_view == "Sagittal":
            pts[:, 0], pts[:, 2] = pts[:, 2].copy(), pts[:, 0].copy()
        pts[:, 1], pts[:, 2] = pts[:, 2].copy(), pts[:, 1].copy()

        self._alt_joint_item.setData(pos=pts, color=_ALT_JOINT_COLORS, size=6.0, pxMode=True)
        for i, (a, b) in enumerate(COCO_SKELETON):
            self._bone_pts_alt[i * 2]     = pts[a]
            self._bone_pts_alt[i * 2 + 1] = pts[b]
        self._alt_bone_item.setData(pos=self._bone_pts_alt,
                                    color=_ALT_BONE_COLORS,
                                    width=1.5,
                                    antialias=True,
                                    mode='lines')

    # ── Camera preset handling ────────────────────────────────────────────────

    def _on_preset_btn(self, name: str):
        self._anim_timer.stop()
        self._set_active_button(name)
        if name == "Free":
            self._current_view = "Free"
            return
        self._animate_to_preset(name)

    def _compute_ortho_fov(self, dist: float) -> float:
        """FOV that makes a skeleton at `dist` appear the same size as at auto_dist with 60°."""
        ref_half_tan = math.tan(math.radians(30.0))  # tan(60°/2)
        half_tan = ref_half_tan * self._auto_dist / dist
        return math.degrees(2.0 * math.atan(half_tan))

    def _animate_to_preset(self, name: str):
        """Smoothly interpolate camera from current position to preset."""
        target = self._PRESETS.get(name)
        if target is None or not _GL_OK or not hasattr(self, '_gl'):
            return
        cur_dist = float(self._gl.opts.get('distance', self._auto_dist))
        cur_el   = float(self._gl.opts.get('elevation', 0.0))
        cur_az   = float(self._gl.opts.get('azimuth', -90.0))
        cur_fov  = float(self._gl.opts.get('fov', 60.0))
        tgt_dist_raw, tgt_el, tgt_az, tgt_fov_raw = target
        tgt_dist = self._auto_dist   # use player-scaled distance, not hardcoded preset
        tgt_fov  = self._compute_ortho_fov(tgt_dist_raw) if tgt_fov_raw is None else tgt_fov_raw

        # Shortest angular path for azimuth
        daz = ((tgt_az - cur_az + 180.0) % 360.0) - 180.0

        self._anim_start  = (cur_dist, cur_el, cur_az, cur_fov)
        self._anim_target = (tgt_dist, tgt_el, cur_az + daz, tgt_fov)
        self._anim_t = 0.0
        self._anim_pending_view = name

        # Apply data-space transform (e.g. Sagittal X↔Z swap) immediately
        self._current_view = name
        if self._current_frame >= 0:
            kps = self._frames_3d.get(self._current_frame)
            if kps is not None:
                self._render(kps)
            if self._frames_3d_alt and self._alt_visible:
                kps_alt = self._frames_3d_alt.get(self._current_frame)
                if kps_alt is not None:
                    self._render_alt(kps_alt)

        self._anim_timer.stop()
        self._anim_timer.start(16)   # ~60 fps

    def _anim_step(self):
        """Advance one animation tick (called every 16 ms)."""
        self._anim_t = min(1.0, self._anim_t + 16.0 / 400.0)
        # Ease-in-out (cosine)
        t = 0.5 * (1.0 - math.cos(math.pi * self._anim_t))

        s_dist, s_el, s_az, s_fov = self._anim_start
        g_dist, g_el, g_az, g_fov = self._anim_target
        dist = s_dist + (g_dist - s_dist) * t
        el   = s_el   + (g_el   - s_el)   * t
        az   = s_az   + (g_az   - s_az)   * t
        fov  = s_fov  + (g_fov  - s_fov)  * t

        self._gl.opts['fov'] = fov
        self._gl.setCameraPosition(distance=dist, elevation=el, azimuth=az)

        if self._anim_t >= 1.0:
            self._anim_timer.stop()

    def _poll_camera(self):
        if not _GL_OK or not hasattr(self, '_gl'):
            return
        az = float(self._gl.opts.get('azimuth', -90.0))
        el = float(self._gl.opts.get('elevation', 0.0))

        # Feature A: update readout (az relative to Frontal preset az = -90)
        az_rel = ((az - (-90.0) + 180.0) % 360.0) - 180.0
        self._angle_label.setText(f"Az {az_rel:+.0f}°  El {el:+.0f}°")

        # Feature C: update gizmo
        self._gizmo.update_camera(az, el)

        # Auto-detect manual drag → switch to Free
        if self._current_view != "Free" and not self._anim_timer.isActive():
            preset = self._PRESETS.get(self._current_view)
            if preset:
                _, p_el, p_az, _ = preset
                if abs(az - p_az) > 2.5 or abs(el - p_el) > 2.5:
                    self._current_view = "Free"
                    self._set_active_button("Free")

    def _apply_preset(self, name: str):
        preset = self._PRESETS.get(name)
        if preset and hasattr(self, '_gl'):
            dist_raw, el, az, fov_raw = preset
            dist = getattr(self, '_auto_dist', dist_raw)
            fov  = self._compute_ortho_fov(dist_raw) if fov_raw is None else fov_raw
            self._gl.opts['fov'] = fov
            self._gl.setCameraPosition(distance=dist, elevation=el, azimuth=az)

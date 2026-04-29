"""
editor_window.py — main application window and entry-point logic.

Classes:
    KeypointEditor  — QMainWindow orchestrating all panels and playback

Functions:
    _dark_palette   — apply dark Fusion theme
    main            — parse CLI args and launch the application
"""

import sys
import os
import re
import csv as _csv
import json
import argparse
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import numpy as np

from PyQt5.QtCore import Qt, QTimer, QSettings, QSize
from PyQt5.QtGui import QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QSizePolicy, QStatusBar, QShortcut,
    QFrame, QStyle, QStyleOptionSlider, QSplitter, QDialog,
    QMessageBox, QFileDialog, QToolBar, QAction, QComboBox, QDockWidget,
)

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

from angles import compute_frame_angles
from constants import _parse_player_id, _fmt_angle
from models import PlayerState
from pose_scene import PoseScene
from video_view import VideoView
from anomaly_bar import AnomalyMarkBar, EVENT_TICK_COLORS
from player_panel import PlayerListPanel
from feature_panel import FeaturePanel, FeatureDescPanel
from setup_dialog import SetupDialog
from skeleton_3d import Skeleton3DPanel
from annotation_panel import AnnotationPanel, ANNOTATION_EVENTS

import cv2

# ── Event label metadata ───────────────────────────────────────────────────────
# Maps event_key (matching event_detection.py output fields) to
# (short_label, hex_color) shown in the frame controls when the current frame
# is that event.
_EVENT_LABELS: dict[str, tuple[str, str]] = {
    "athlete_move":   ("Move",       "#FF9500"),   # amber
    "ball_lift":      ("Ball lift",  "#FFD700"),
    "towel1_contact": ("T1 pickup",  "#78FF50"),
    "towel1_release": ("T1 drop",    "#00C853"),
    "towel2_contact": ("T2 pickup",  "#50C8FF"),
    "towel2_release": ("T2 drop",    "#0090FF"),
    "finish_cross":   ("Finish",     "#DC50FF"),
}


# ── Dark palette ───────────────────────────────────────────────────────────────

def _dark_palette(app: QApplication):
    p = app.palette()
    D = QColor
    p.setColor(QPalette.Window,          D(42, 42, 42))
    p.setColor(QPalette.WindowText,      D(220, 220, 220))
    p.setColor(QPalette.Base,            D(30, 30, 30))
    p.setColor(QPalette.AlternateBase,   D(42, 42, 42))
    p.setColor(QPalette.ToolTipBase,     D(55, 55, 55))
    p.setColor(QPalette.ToolTipText,     D(220, 220, 220))
    p.setColor(QPalette.Text,            D(220, 220, 220))
    p.setColor(QPalette.Button,          D(60, 60, 60))
    p.setColor(QPalette.ButtonText,      D(220, 220, 220))
    p.setColor(QPalette.BrightText,      Qt.red)
    p.setColor(QPalette.Highlight,       D(42, 130, 218))
    p.setColor(QPalette.HighlightedText, Qt.black)
    p.setColor(QPalette.Disabled, QPalette.Text,       D(80, 80, 80))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, D(80, 80, 80))
    p.setColor(QPalette.Disabled, QPalette.WindowText, D(80, 80, 80))
    app.setPalette(p)


# ── KeypointEditor ─────────────────────────────────────────────────────────────

class KeypointEditor(QMainWindow):

    def __init__(self, video_folder: str = "", poses_parent: str = "",
                 features_csv: str = "", anomaly_csv: str = "",
                 poses_3d_dir: str = "", poses_3d_alt_dir: str = "",
                 events_dir: str = "",
                 start_in_edit: bool = False):
        super().__init__()
        self.setWindowTitle("Keypoint Editor")
        self.resize(1560, 900)

        # ── Session data ──────────────────────────────────────────────────────
        self._video_folder    = video_folder
        self._poses_parent    = poses_parent
        self._features_csv    = features_csv
        self._anomaly_csv     = anomaly_csv
        self._poses_3d_dir    = poses_3d_dir
        self._poses_3d_alt_dir = poses_3d_alt_dir
        self._events_dir      = events_dir     # optional drill-event sidecar JSONs

        self._current_model: str = ""
        self._players:       dict[str, dict] = {}
        self._ordered_pids:  list[str] = []
        self._player_states: dict[str, PlayerState] = {}

        self._current_pid:  str | None = None
        self._pose_data:    dict | None = None
        self._frame_map:    dict[int, dict] = {}
        self._frame_list:   list[int] = []
        self._orig_kps:     dict[int, list] = {}
        self._cap:          cv2.VideoCapture | None = None
        self._total_frames: int = 0
        self._fps:          float = 30.0
        self._list_idx:     int = 0
        self._last_read_vf: int = -2
        self._anomaly_frames: set[int] = set()
        self._current_events: dict = {}          # loaded from <events_dir>/<pid>_events.json
        self._current_event_ticks: dict[str, list[int]] = {}  # vframe indices for mark bar

        self._annotations:      dict[str, dict[str, int]] = {}  # pid → {ev_key → vframe}
        self._annotations_csv:  str = ""

        self._frames_3d:     dict[int, np.ndarray] = {}
        self._frames_3d_alt: dict[int, np.ndarray] = {}

        self._feat_df:  object = None
        self._anom_df:  object = None

        self._edit_mode   = start_in_edit
        self._kps_visible = True
        self._playing     = False

        # ── Build scene / view ────────────────────────────────────────────────
        self.scene = PoseScene()
        self.view  = VideoView(self.scene)
        self.scene.pose_edited.connect(self._on_pose_edited)
        self.scene.live_changed.connect(self._on_live_changed)
        self.scene.setup_overlays()

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()
        self._build_shortcuts()

        # ── Playback timer ────────────────────────────────────────────────────
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_playback)

        # ── Load data ─────────────────────────────────────────────────────────
        if video_folder and poses_parent:
            self._init_session()
        else:
            self._run_setup_dialog()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_toolbar()

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setChildrenCollapsible(False)

        # Left: player list
        self._player_panel = PlayerListPanel()
        self._player_panel.player_selected.connect(self._on_player_selected)
        self._player_panel.request_features_folder.connect(self._add_features_folder)
        self._player_panel.request_anomaly_folder.connect(self._add_anomaly_folder)
        splitter.addWidget(self._player_panel)

        # Centre: video + controls
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self.view, 1)
        cv.addWidget(self._build_controls())
        splitter.addWidget(center)

        # Right: feature panel
        self._feat_panel = FeaturePanel()
        splitter.addWidget(self._feat_panel)

        # Far right: feature descriptions (hidden by default)
        self._feat_desc_panel = FeatureDescPanel()
        self._feat_desc_panel.hide()
        splitter.addWidget(self._feat_desc_panel)

        self._splitter = splitter
        splitter.setSizes([240, 1060, 272, 0])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setStretchFactor(3, 0)

        self.setCentralWidget(splitter)

        # ── 3D Skeleton dock (NEW) ─────────────────────────────────────────
        self._skeleton_3d = Skeleton3DPanel()
        self._dock_3d = QDockWidget("3D Skeleton", self)
        self._dock_3d.setWidget(self._skeleton_3d)
        self._dock_3d.setMinimumWidth(440)
        self._dock_3d.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable)
        self._dock_3d.hide()
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock_3d)
        # Keep toolbar checkbox in sync when dock is closed via X
        self._dock_3d.visibilityChanged.connect(
            lambda v: self._act_3d.setChecked(v))

        # ── Annotation dock ───────────────────────────────────────────────────
        self._annotation_panel = AnnotationPanel()
        self._annotation_panel.mark_requested.connect(self._on_mark_requested)
        self._annotation_panel.clear_requested.connect(self._on_clear_requested)
        self._annotation_panel.save_requested.connect(self._save_annotations)

        self._dock_ann = QDockWidget("Towel Annotations", self)
        self._dock_ann.setWidget(self._annotation_panel)
        self._dock_ann.setMinimumWidth(260)
        self._dock_ann.setMaximumWidth(320)
        self._dock_ann.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable)
        self._dock_ann.hide()
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock_ann)
        self._dock_ann.visibilityChanged.connect(
            lambda v: self._act_annotate.setChecked(v))

        self._status = QStatusBar()
        self._status.showMessage("Open the setup dialog to load a session  (File → New Session…)")
        self.setStatusBar(self._status)

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setStyleSheet("QToolButton { font-size: 12px; }")
        self.addToolBar(tb)

        # Mode toggle
        self._act_mode = QAction("View Mode", self)
        self._act_mode.setCheckable(True)
        self._act_mode.setChecked(self._edit_mode)
        self._act_mode.setToolTip(
            "Toggle between View mode (keypoints locked) and Editor mode\n"
            "(keypoints can be dragged to correct pose estimation errors)")
        self._act_mode.triggered.connect(self._on_mode_toggled)
        tb.addAction(self._act_mode)
        self._update_mode_label()

        tb.addSeparator()

        # Keypoint visibility
        self._act_kps = QAction("Keypoints  ✓", self)
        self._act_kps.setCheckable(True)
        self._act_kps.setChecked(True)
        self._act_kps.setShortcut(QKeySequence("K"))
        self._act_kps.setToolTip(
            "Show / hide the keypoint and skeleton overlay (K)\n"
            "The video frame is always visible.")
        self._act_kps.triggered.connect(self._on_kps_toggled)
        tb.addAction(self._act_kps)

        self._act_kp_idx = QAction("Indices", self)
        self._act_kp_idx.setCheckable(True)
        self._act_kp_idx.setChecked(False)
        self._act_kp_idx.setShortcut(QKeySequence("I"))
        self._act_kp_idx.setToolTip("Show / hide keypoint index numbers on 2D and 3D views (I)")
        self._act_kp_idx.triggered.connect(self._on_kp_idx_toggled)
        tb.addAction(self._act_kp_idx)

        tb.addSeparator()

        # Angle overlays
        _lbl_overlays = QLabel("  Overlays: ")
        _lbl_overlays.setStyleSheet("font-size: 12px; color: #888; font-style: italic;")
        tb.addWidget(_lbl_overlays)

        overlay_defs = [
            ("hip_hinge",  "Hip Hinge",   "Cyan arc at mid-hip showing torso–thigh bend angle"),
            ("torso_lean", "Torso Lean",  "Amber arc showing torso deviation from vertical"),
            ("knee_l",     "Knee L",      "Green arc at left knee showing knee flexion angle"),
            ("knee_r",     "Knee R",      "Red arc at right knee showing knee flexion angle"),
            ("shin_l",     "Shin L",      "Teal line + arc for left shin lean from vertical"),
            ("shin_r",     "Shin R",      "Salmon line + arc for right shin lean from vertical"),
        ]
        self._overlay_actions: dict[str, QAction] = {}
        for key, label, tip in overlay_defs:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(False)
            act.setToolTip(tip + "\n(click to toggle)")
            act.triggered.connect(lambda checked, k=key: self._on_overlay_toggled(k, checked))
            tb.addAction(act)
            self._overlay_actions[key] = act

        tb.addSeparator()
        act_all = QAction("All Off", self)
        act_all.setShortcut(QKeySequence("O"))
        act_all.setToolTip("Toggle all angle overlays on / off  (O)")
        act_all.triggered.connect(self._toggle_all_overlays)
        tb.addAction(act_all)
        self._act_all_overlays = act_all

        tb.addSeparator()

        # Model selector
        _lbl_model = QLabel("  Model: ")
        _lbl_model.setStyleSheet("font-size: 12px; color: #888; font-style: italic;")
        tb.addWidget(_lbl_model)
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(160)
        self._model_combo.setToolTip(
            "Select which pose model's keypoints to display for this player.\n"
            "Available models are the subfolders inside the Keypoints Parent Folder.")
        self._model_combo.setEnabled(False)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        tb.addWidget(self._model_combo)

        act_refresh = QAction("⟳ Refresh", self)
        act_refresh.setToolTip(
            "Re-scan the keypoints parent folder for new model subfolders or updated JSONs.")
        act_refresh.triggered.connect(self._refresh_keypoints)
        tb.addAction(act_refresh)

        tb.addSeparator()

        # Fit + feature desc + 3D panel + Setup
        act_fit = QAction("Fit  (F)", self)
        act_fit.setToolTip("Fit the current frame in the view  (F)")
        act_fit.triggered.connect(self.view.fit)
        tb.addAction(act_fit)

        act_desc = QAction("ⓘ Features", self)
        act_desc.setCheckable(True)
        act_desc.setChecked(False)
        act_desc.setToolTip("Show / hide the feature descriptions panel")
        act_desc.triggered.connect(self._on_feat_desc_toggled)
        tb.addAction(act_desc)
        self._act_feat_desc = act_desc

        # 3D panel toggle (NEW)
        self._act_3d = QAction("3D  (Ctrl+3)", self)
        self._act_3d.setCheckable(True)
        self._act_3d.setChecked(False)
        self._act_3d.setShortcut(QKeySequence("Ctrl+3"))
        self._act_3d.setToolTip(
            "Show / hide the 3D skeleton panel  (Ctrl+3)\n"
            "Requires a poses-3d folder to be loaded.")
        self._act_3d.triggered.connect(self._on_3d_toggled)
        tb.addAction(self._act_3d)

        self._act_annotate = QAction("Annotate  (A)", self)
        self._act_annotate.setCheckable(True)
        self._act_annotate.setChecked(False)
        self._act_annotate.setShortcut(QKeySequence("A"))
        self._act_annotate.setToolTip(
            "Show / hide the towel annotation panel  (A)\n"
            "Keys 1–4 stamp T1 pickup / drop / T2 pickup / drop at the current frame.")
        self._act_annotate.triggered.connect(self._on_annotate_toggled)
        tb.addAction(self._act_annotate)

        act_setup = QAction("New Session…", self)
        act_setup.setToolTip("Open the setup dialog to change folders or start a new session")
        act_setup.triggered.connect(self._run_setup_dialog)
        tb.addAction(act_setup)

    def _build_controls(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setFixedHeight(96)
        v = QVBoxLayout(panel)
        v.setContentsMargins(10, 5, 10, 4)
        v.setSpacing(3)

        h = QHBoxLayout()
        h.setSpacing(6)

        self._btn_prev = QPushButton("◀  Prev")
        self._btn_prev.setFixedWidth(82)
        self._btn_prev.setToolTip("Previous tracked frame  (←)")
        self._btn_prev.clicked.connect(self._prev_frame)
        h.addWidget(self._btn_prev)

        self._btn_play = QPushButton("▶  Play")
        self._btn_play.setFixedWidth(82)
        self._btn_play.setToolTip("Play / pause frame sequence  (Space)")
        self._btn_play.clicked.connect(self._toggle_play)
        h.addWidget(self._btn_play)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setFixedWidth(82)
        self._btn_stop.setToolTip("Stop playback")
        self._btn_stop.clicked.connect(self._stop)
        h.addWidget(self._btn_stop)

        self._btn_next = QPushButton("Next  ▶")
        self._btn_next.setFixedWidth(82)
        self._btn_next.setToolTip("Next tracked frame  (→)")
        self._btn_next.clicked.connect(self._next_frame)
        h.addWidget(self._btn_next)

        h.addSpacing(12)

        self._lbl_frame = QLabel("Frame: —")
        self._lbl_frame.setFixedWidth(200)
        self._lbl_frame.setToolTip("Current video frame / total frames  [position in tracked list]")
        h.addWidget(self._lbl_frame)

        self._lbl_time = QLabel("—")
        self._lbl_time.setFixedWidth(75)
        self._lbl_time.setToolTip("Timestamp within the video clip")
        h.addWidget(self._lbl_time)

        self._lbl_edited = QLabel("")
        self._lbl_edited.setStyleSheet("color:#fa0; font-style:italic;")
        self._lbl_edited.setFixedWidth(70)
        self._lbl_edited.setToolTip("This frame has unsaved keypoint edits")
        h.addWidget(self._lbl_edited)

        self._lbl_anomaly = QLabel("")
        self._lbl_anomaly.setFixedWidth(100)
        self._lbl_anomaly.setToolTip("This frame is flagged as anomalous by STG-NF scoring")
        h.addWidget(self._lbl_anomaly)

        self._lbl_event = QLabel("")
        self._lbl_event.setFixedWidth(88)
        self._lbl_event.setToolTip(
            "This frame is a key drill event\n"
            "(ball lift / towel pickup or drop / finish)")
        h.addWidget(self._lbl_event)

        h.addStretch()

        self._btn_save = QPushButton("Save  Ctrl+S")
        self._btn_save.setFixedWidth(110)
        self._btn_save.setToolTip("Save all keypoint edits to the JSON file  (Ctrl+S)\n"
                                  "A timestamped backup is created automatically.")
        self._btn_save.setStyleSheet(
            "QPushButton { background:#2a7; color:white; font-weight:bold; }"
            "QPushButton:hover { background:#3b8; }")
        self._btn_save.clicked.connect(self._save)
        h.addWidget(self._btn_save)

        v.addLayout(h)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setTracking(True)
        self._slider.setToolTip("Timeline — drag to seek.  Orange ticks are anomalous frames.")
        self._slider.valueChanged.connect(self._on_slider)
        v.addWidget(self._slider)

        self._mark_bar = AnomalyMarkBar(self._slider)
        self._mark_bar.seek_to.connect(self._on_event_tick_clicked)
        v.addWidget(self._mark_bar)

        return panel

    def _build_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left),  self, self._prev_frame)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._next_frame)
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence("Ctrl+S"),     self, self._save)
        QShortcut(QKeySequence("Ctrl+Z"),     self, self._undo)
        QShortcut(QKeySequence("R"),          self, self._reset_frame)
        QShortcut(QKeySequence("F"),          self, self.view.fit)
        QShortcut(QKeySequence("F11"),        self, self._toggle_fullscreen)

        # Annotation shortcuts — only fire when the annotation dock is visible
        for i, (ev_key, _label, _color) in enumerate(ANNOTATION_EVENTS, 1):
            QShortcut(QKeySequence(str(i)), self,
                      lambda k=ev_key: self._dock_ann.isVisible() and self._on_mark_requested(k))
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, self._save_annotations)

    # ─────────────────────────────────────────────────────────────────────────
    # Session initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _run_setup_dialog(self):
        dlg = SetupDialog(
            self,
            self._video_folder, self._poses_parent, self._poses_3d_dir,
            self._poses_3d_alt_dir,
            self._features_csv, self._anomaly_csv, self._events_dir,
            self._annotations_csv,
        )
        if dlg.exec_() != QDialog.Accepted:
            if not self._video_folder:
                QApplication.quit()
            return
        self._video_folder      = dlg.video_folder
        self._poses_parent      = dlg.poses_folder
        self._poses_3d_dir      = dlg.poses_3d_folder
        self._poses_3d_alt_dir  = dlg.poses_3d_alt_folder
        self._features_csv      = dlg.features_csv
        self._anomaly_csv       = dlg.anomaly_csv
        self._events_dir        = dlg.events_folder
        self._annotations_csv   = dlg.annotations_csv
        self._annotations.clear()
        self._current_model = ""
        self._init_session()

    def _init_session(self):
        self._load_csvs()
        if self._annotations_csv:
            self._load_annotations_csv(self._annotations_csv)
        self._scan_players()
        self._player_panel.populate(list(self._players.values()))

    def _load_csvs(self):
        self._feat_df = None
        self._anom_df = None
        if not _PANDAS:
            return
        if self._features_csv and Path(self._features_csv).is_file():
            try:
                self._feat_df = pd.read_csv(self._features_csv)
                self._status.showMessage(
                    f"Features loaded: {Path(self._features_csv).name}  "
                    f"({len(self._feat_df)} rows)")
            except Exception as e:
                self._status.showMessage(f"Could not load features CSV: {e}")
        if self._anomaly_csv and Path(self._anomaly_csv).is_file():
            try:
                self._anom_df = pd.read_csv(self._anomaly_csv)
            except Exception as e:
                self._status.showMessage(f"Could not load anomaly CSV: {e}")

    @staticmethod
    def _build_video_map(video_folder: str) -> dict:
        if not video_folder:
            return {}
        folder = Path(video_folder)
        if not folder.is_dir():
            return {}
        _NIC_RE = re.compile(
            r"^(\d{4})\s+NIC\s+(.+?)\s+(DL\s*\d+)(?:\s+(.+))?$", re.IGNORECASE)
        vid_map: dict = {}
        for mp4 in folder.glob("*.mp4"):
            m = _NIC_RE.match(mp4.stem.strip())
            if m:
                base = f"{m.group(1)}_{m.group(2).replace(' ', '_')}_{m.group(3).replace(' ', '')}"
                drill = m.group(4).strip().replace(" ", "_").upper() if m.group(4) else ""
                pid = f"{base}_{drill}" if drill else base
            else:
                pid = mp4.stem.replace(" ", "_")
            vid_map[pid] = mp4
            # Also store a sanitized variant (apostrophes / special chars → _)
            # so JSON stems produced by extraction scripts on macOS match too.
            sanitized = re.sub(r"[^A-Za-z0-9_]", "_", pid)
            if sanitized != pid:
                vid_map[sanitized] = mp4
        return vid_map

    @staticmethod
    def _scan_model_dirs(poses_parent: str) -> dict[str, Path]:
        parent = Path(poses_parent)
        if not parent.is_dir():
            return {}
        return {sub.name: sub for sub in sorted(parent.iterdir())
                if sub.is_dir() and any(sub.glob("*.json"))}

    def _scan_players(self):
        self._players.clear()
        model_dirs = self._scan_model_dirs(self._poses_parent)
        if not model_dirs:
            self._status.showMessage(
                "No model subfolders found in the keypoints parent folder.")
            return

        feat_ids = set(self._feat_df["player_id"].astype(str)) if self._feat_df is not None else set()
        anom_ids = set(self._anom_df["player_id"].astype(str)) if self._anom_df is not None else set()
        vid_map  = self._build_video_map(self._video_folder)

        pid_models: dict[str, dict[str, str]] = {}
        for model_name, folder in model_dirs.items():
            for jf in sorted(folder.glob("*.json")):
                pid = jf.stem
                pid_models.setdefault(pid, {})[model_name] = str(jf)

        _PREFERRED = ["poses_yolo11x", "poses_yolo26x", "poses_rtmpose_body17"]

        for pid, models in pid_models.items():
            vid = vid_map.get(pid)
            default = next((m for m in _PREFERRED if m in models), next(iter(models)))
            n_frames = 0
            try:
                with open(models[default]) as f:
                    raw = json.load(f)
                n_frames = len(raw.get("athlete_frames", []))
            except Exception:
                pass
            self._players[pid] = {
                "player_id":    pid,
                "has_video":    vid is not None,
                "has_features": pid in feat_ids,
                "has_anomaly":  pid in anom_ids,
                "n_frames":     n_frames,
                "video_path":   str(vid) if vid is not None else None,
                "poses_path":   models[default],
                "models":       models,
                "default_model": default,
            }

        self._ordered_pids = list(self._players.keys())
        n     = len(self._ordered_pids)
        n_vid = sum(1 for p in self._players.values() if p["has_video"])
        self._status.showMessage(
            f"Loaded {n} players ({n_vid} with video)  —  "
            f"{len(model_dirs)} model(s): {', '.join(model_dirs)}  —  "
            f"{'No features' if self._feat_df is None else f'{len(feat_ids)} with features'}  "
            f"{'No anomalies' if self._anom_df is None else 'anomaly data loaded'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Player selection
    # ─────────────────────────────────────────────────────────────────────────

    def _on_player_selected(self, pid: str):
        if pid == self._current_pid:
            return
        if self._edit_mode and self._current_pid:
            state = self._player_states.get(self._current_pid)
            if state and state.edits:
                result = self._prompt_save(self._current_pid, state)
                if result == QMessageBox.Cancel:
                    self._player_panel.select_player(self._current_pid)
                    return
        self._save_player_state()
        self._current_pid = pid
        self._load_player(pid)

    def _save_player_state(self):
        if self._current_pid is None:
            return
        state = self._player_states.setdefault(self._current_pid, PlayerState())
        state.frame_idx    = self._list_idx
        state.kps_visible  = self._kps_visible
        state.overlay_states = {k: act.isChecked() for k, act in self._overlay_actions.items()}
        state.edits        = dict(self._player_states.get(self._current_pid,
                                                          PlayerState()).edits)

    def _prompt_save(self, pid: str, state: PlayerState) -> int:
        _, name, _ = _parse_player_id(pid)
        n = len(state.edits)
        msg = QMessageBox(self)
        msg.setWindowTitle("Unsaved Edits")
        msg.setIcon(QMessageBox.Warning)
        msg.setText(f"<b>{name}</b> has <b>{n} edited frame(s)</b> that haven't been saved.")
        msg.setInformativeText("Save changes before switching players?")
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Save)
        result = msg.exec_()
        if result == QMessageBox.Save:
            self._save()
        elif result == QMessageBox.Discard:
            state.edits.clear()
        return result

    def _on_model_changed(self, model_name: str):
        if not model_name or not self._current_pid:
            return
        if model_name == self._current_model:
            return
        info = self._players.get(self._current_pid)
        if info is None or model_name not in info.get("models", {}):
            return
        self._current_model = model_name
        info["poses_path"] = info["models"][model_name]
        try:
            with open(info["poses_path"]) as f:
                self._pose_data = json.load(f)
        except Exception as e:
            self._status.showMessage(f"Cannot load JSON for model {model_name}: {e}")
            return
        self._frame_map.clear()
        self._frame_list.clear()
        self._orig_kps.clear()
        for entry in self._pose_data.get("athlete_frames", []):
            vf = entry["frame"]
            self._frame_map[vf] = entry
            self._orig_kps[vf] = deepcopy(entry["keypoints"])
        if self._total_frames > 0:
            self._frame_list = list(range(self._total_frames))
        else:
            self._frame_list = sorted(self._frame_map.keys())
        n = len(self._frame_list)
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._list_idx = 0
        self._frames_3d = self._load_3d_poses(self._current_pid)
        self._skeleton_3d.load_player(self._frames_3d)
        self._frames_3d_alt = self._load_3d_poses(self._current_pid, alt=True)
        self._skeleton_3d.load_alt_player(self._frames_3d_alt)
        self._show(0)
        self.view.fit()
        self._status.showMessage(
            f"{self._current_pid}  |  model: {model_name}  |  {n} frames  |  {len(self._frame_map)} poses")

    def _refresh_keypoints(self):
        if not self._poses_parent:
            return
        model_dirs = self._scan_model_dirs(self._poses_parent)
        for pid, info in self._players.items():
            new_models: dict[str, str] = {}
            for model_name, folder in model_dirs.items():
                jf = folder / f"{pid}.json"
                if jf.is_file():
                    new_models[model_name] = str(jf)
            info["models"] = new_models
            if new_models and info["poses_path"] not in new_models.values():
                info["poses_path"] = next(iter(new_models.values()))
        if self._current_pid:
            info = self._players.get(self._current_pid)
            if info:
                self._populate_model_combo(info)
        self._status.showMessage(
            f"Refreshed — {len(model_dirs)} model(s): {', '.join(model_dirs)}")

    def _populate_model_combo(self, info: dict):
        models = info.get("models", {})
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in sorted(models):
            self._model_combo.addItem(m)
        target = self._current_model if self._current_model in models else info.get("default_model", "")
        if target:
            idx = self._model_combo.findText(target)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._model_combo.setEnabled(len(models) > 0)
        self._model_combo.blockSignals(False)
        if target and target != self._current_model:
            self._current_model = self._model_combo.currentText()

    def _on_feat_desc_toggled(self, checked: bool):
        if checked:
            self._feat_desc_panel.show()
            sizes = self._splitter.sizes()
            if sizes[3] == 0:
                sizes[3] = 320
                sizes[1] = max(200, sizes[1] - 320)
                self._splitter.setSizes(sizes)
        else:
            self._feat_desc_panel.hide()

    def _on_3d_toggled(self, checked: bool):
        """Show/hide the 3D skeleton dock panel."""
        self._dock_3d.setVisible(checked)

    def _on_annotate_toggled(self, checked: bool):
        """Show/hide the towel annotation dock panel."""
        self._dock_ann.setVisible(checked)

    # ─────────────────────────────────────────────────────────────────────────
    # Player loading (3D data integrated)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_3d_poses(self, pid: str, alt: bool = False) -> dict[int, np.ndarray]:
        """
        Load 3D keypoints for a player.

        Priority (primary, alt=False):
          1. Explicit 3D poses folder (configured via File → New Session).
          2. keypoints_3d fields in the already-loaded 2D pose JSON.

        For alt=True only the explicit alt 3D dir is checked (no JSON fallback).

        Returns dict mapping video frame index → (17, 3) float32 array,
        or empty dict if no 3D data is available.
        """
        def _extract(data: dict) -> dict[int, np.ndarray]:
            out: dict[int, np.ndarray] = {}
            for entry in data.get("athlete_frames", []):
                kps3d = entry.get("keypoints_3d")
                if kps3d is not None:
                    out[entry["frame"]] = np.array(kps3d, dtype=np.float32)
            return out

        if alt:
            # Alt source: only from explicit alt dir
            if self._poses_3d_alt_dir:
                p = Path(self._poses_3d_alt_dir) / f"{pid}.json"
                if p.is_file():
                    try:
                        with open(p) as f:
                            return _extract(json.load(f))
                    except Exception:
                        pass
            return {}

        # 1 — explicit primary 3D directory
        if self._poses_3d_dir:
            p = Path(self._poses_3d_dir) / f"{pid}.json"
            if p.is_file():
                try:
                    with open(p) as f:
                        return _extract(json.load(f))
                except Exception:
                    pass

        # 2 — fall back to keypoints_3d in the current pose JSON
        if self._pose_data:
            result = _extract(self._pose_data)
            if result:
                return result

        return {}

    def _load_player(self, pid: str):
        info = self._players.get(pid)
        if info is None:
            return
        self._stop()

        # Open video
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if info.get("video_path"):
            self._cap = cv2.VideoCapture(info["video_path"])
            if self._cap.isOpened():
                self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                self._last_read_vf = -2
            else:
                self._cap = None

        # Model combo
        self._populate_model_combo(info)
        self._current_model = self._model_combo.currentText()
        if self._current_model and self._current_model in info.get("models", {}):
            info["poses_path"] = info["models"][self._current_model]

        # Load 2D JSON
        try:
            with open(info["poses_path"]) as f:
                self._pose_data = json.load(f)
        except Exception as e:
            self._status.showMessage(f"Cannot load JSON: {e}")
            return

        self._frame_map.clear()
        self._frame_list.clear()
        self._orig_kps.clear()

        for entry in self._pose_data.get("athlete_frames", []):
            vf = entry["frame"]
            self._frame_map[vf] = entry
            self._orig_kps[vf] = deepcopy(entry["keypoints"])

        # Show all video frames; overlay only where poses exist
        if self._total_frames > 0:
            self._frame_list = list(range(self._total_frames))
        else:
            self._frame_list = sorted(self._frame_map.keys())

        n = len(self._frame_list)
        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)

        # Load primary 3D poses
        self._frames_3d = self._load_3d_poses(pid)
        self._skeleton_3d.load_player(self._frames_3d)

        # Load alt 3D poses (e.g. TRAM) if configured
        self._frames_3d_alt = self._load_3d_poses(pid, alt=True)
        self._skeleton_3d.load_alt_player(self._frames_3d_alt)

        # Restore player state
        state = self._player_states.get(pid)
        if state is None:
            state = PlayerState()
            self._player_states[pid] = state

        for vf, edited_kps in state.edits.items():
            if vf in self._frame_map:
                self._frame_map[vf]["keypoints"] = edited_kps

        for key, act in self._overlay_actions.items():
            v = state.overlay_states.get(key, False)
            act.setChecked(v)
            self.scene.set_overlay_visible(key, v)

        self._kps_visible = state.kps_visible
        self._act_kps.setChecked(self._kps_visible)

        # Anomaly data
        self._anomaly_frames.clear()
        self._mark_bar.clear()
        if self._anom_df is not None:
            rows = self._anom_df[
                (self._anom_df["player_id"].astype(str) == pid) &
                (self._anom_df["is_low_prob"].astype(str).str.lower() == "true")
            ]
            self._anomaly_frames = set(rows["frame"].astype(int))
            flagged_idxs = [vf for vf in self._anomaly_frames if vf < n]
            self._mark_bar.set_marks(max(0, n - 1), flagged_idxs)

        # Drill event markers
        self._current_events.clear()
        if self._events_dir:
            ev_path = Path(self._events_dir) / f"{pid}_events.json"
            if ev_path.is_file():
                try:
                    with open(ev_path) as _ef:
                        self._current_events = json.load(_ef)
                except Exception:
                    pass
        self._update_event_marks(n)

        # Annotation panel
        self._annotation_panel.update_display(self._annotations.get(pid))

        # Feature panel
        feature_row = None
        if self._feat_df is not None:
            rows = self._feat_df[self._feat_df["player_id"].astype(str) == pid]
            if not rows.empty:
                feature_row = rows.iloc[0].to_dict()
        self._feat_panel.load_player_agg(feature_row, pid)
        self._feat_panel.load_player_events(self._current_events)

        _, name, pos = _parse_player_id(pid)
        self.setWindowTitle(f"Keypoint Editor — {name}  [{pos}]")

        start_idx = min(state.frame_idx, max(0, n - 1))
        self._list_idx = 0
        self._show(start_idx)
        if start_idx == 0:
            self.view.fit()

        has_3d = bool(self._frames_3d)
        has_3d_alt = bool(self._frames_3d_alt)
        n_poses = len(self._frame_map)
        self._status.showMessage(
            f"{pid}  |  {n} frames  |  {n_poses} poses  |  {self._fps:.1f} fps"
            + (f"  |  3D: {len(self._frames_3d)} frames" if has_3d else "  |  3D: none")
            + (f"  |  Alt: {len(self._frames_3d_alt)} frames" if has_3d_alt else "  |  Alt: none")
            + (f"  |  {len(self._anomaly_frames)} anomalous" if self._anomaly_frames else "")
        )

    def _update_event_marks(self, maximum: int):
        """
        Convert event video-frame numbers to timeline list-indices, store in
        self._current_event_ticks, then call _update_annotation_ticks() which
        merges with manual annotations and pushes to the mark bar.
        Since _frame_list now covers all video frames, list_idx == vframe directly.
        """
        self._current_event_ticks = {}
        for ev_key in _EVENT_LABELS:
            vf = self._current_events.get(f"{ev_key}_frame")
            if vf is None:
                continue
            idx = max(0, min(vf, maximum))
            self._current_event_ticks[ev_key] = [idx]
        self._update_annotation_ticks()

    def _update_annotation_ticks(self):
        """Merge auto-detected event ticks with manual annotations and update the mark bar."""
        ann = self._annotations.get(self._current_pid, {})
        n = len(self._frame_list)
        events = dict(self._current_event_ticks)
        for ev_key, vf in ann.items():
            idx = max(0, min(vf, max(0, n - 1)))
            events[ev_key] = [idx]   # manual annotation overrides auto for same key
        self._mark_bar.set_events(events)

    def _on_event_tick_clicked(self, list_idx: int):
        """Called when the user clicks an event tick in the mark bar."""
        self._stop()
        self._show(list_idx)

    # ─────────────────────────────────────────────────────────────────────────
    # Frame display
    # ─────────────────────────────────────────────────────────────────────────

    def _show(self, list_idx: int, fast: bool = False):
        if not self._frame_list:
            return
        list_idx = max(0, min(list_idx, len(self._frame_list) - 1))
        self._list_idx = list_idx
        vf    = self._frame_list[list_idx]
        entry = self._frame_map.get(vf)

        if self._cap is not None:
            img = self._read_frame(vf)
            if img is not None:
                self.scene.set_frame_bgr(img)

        if entry is not None:
            kps = entry["keypoints"]
            if fast and self.scene._kps:
                self.scene.update_keypoints_inplace(kps, self._edit_mode)
            else:
                self.scene.load_keypoints(kps, self._edit_mode)
            self.scene.set_kps_visible(self._kps_visible)
        else:
            self.scene.clear_keypoints()

        # Update 3D skeleton panel (NEW)
        self._skeleton_3d.update_frame(vf)

        # Update annotation panel current-frame hint
        self._annotation_panel.set_current_frame(vf)

        self._slider.blockSignals(True)
        self._slider.setValue(list_idx)
        self._slider.blockSignals(False)

        ts = entry.get("timestamp", vf / self._fps) if entry else vf / self._fps
        pose_tag = f"  [{len(self._frame_map)} poses]" if entry is None else ""
        self._lbl_frame.setText(
            f"Frame {vf}/{max(0, self._total_frames-1)}{pose_tag}")
        self._lbl_time.setText(f"{ts:.3f} s")
        state = self._player_states.get(self._current_pid)
        has_edits = bool(state and vf in state.edits)
        self._lbl_edited.setText("● edited" if has_edits else "")
        if vf in self._anomaly_frames:
            self._lbl_anomaly.setText("⚠ anomalous")
            self._lbl_anomaly.setStyleSheet("color:#f84; font-weight:bold;")
        else:
            self._lbl_anomaly.setText("")
            self._lbl_anomaly.setStyleSheet("")

        # Event label
        ev_label, ev_color = "", ""
        for ev_key, (label, color) in _EVENT_LABELS.items():
            if self._current_events.get(f"{ev_key}_frame") == vf:
                ev_label, ev_color = label, color
                break
        if ev_label:
            self._lbl_event.setText(ev_label)
            self._lbl_event.setStyleSheet(f"color:{ev_color}; font-weight:bold;")
        else:
            self._lbl_event.setText("")
            self._lbl_event.setStyleSheet("")

    def _read_frame(self, vframe: int):
        if self._cap is None:
            return None
        if vframe != self._last_read_vf + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, vframe)
        ret, frame = self._cap.read()
        if ret:
            self._last_read_vf = vframe
            return frame
        self._last_read_vf = -2
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Toolbar actions
    # ─────────────────────────────────────────────────────────────────────────

    def _on_mode_toggled(self, checked: bool):
        self._edit_mode = checked
        self.scene.set_editable(checked)
        self._update_mode_label()

    def _update_mode_label(self):
        if self._edit_mode:
            self._act_mode.setText("Editor Mode")
            self._act_mode.setToolTip(
                "Currently in EDITOR mode — keypoints can be dragged.\n"
                "Click to switch to View mode (keypoints locked).")
            self.statusBar().setStyleSheet("QStatusBar { border-top: 2px solid #e65100; }")
        else:
            self._act_mode.setText("View Mode")
            self._act_mode.setToolTip(
                "Currently in VIEW mode — keypoints are locked.\n"
                "Click to switch to Editor mode.")
            self.statusBar().setStyleSheet("")

    def _on_kps_toggled(self, checked: bool):
        self._kps_visible = checked
        self.scene.set_kps_visible(checked)
        self._act_kps.setText(f"Keypoints  {'✓' if checked else '✗'}")

    def _on_kp_idx_toggled(self, checked: bool):
        self.scene.set_kp_indices_visible(checked)
        self._skeleton_3d.set_labels_visible(checked)

    def _on_overlay_toggled(self, key: str, checked: bool):
        self.scene.set_overlay_visible(key, checked)
        any_on = any(act.isChecked() for act in self._overlay_actions.values())
        self._act_all_overlays.setText("All Off" if any_on else "All On")

    def _toggle_all_overlays(self):
        any_on = any(act.isChecked() for act in self._overlay_actions.values())
        target = not any_on
        for key, act in self._overlay_actions.items():
            act.setChecked(target)
            self.scene.set_overlay_visible(key, target)
        self._act_all_overlays.setText("All Off" if target else "All On")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ─────────────────────────────────────────────────────────────────────────
    # Playback
    # ─────────────────────────────────────────────────────────────────────────

    def _next_frame(self):
        self._stop()
        self._show(self._list_idx + 1)

    def _prev_frame(self):
        self._stop()
        self._show(self._list_idx - 1)

    def _toggle_play(self):
        if self._playing:
            self._stop()
        else:
            self._play()

    def _play(self):
        if not self._frame_list:
            return
        self._playing = True
        self._btn_play.setText("⏸  Pause")
        self._play_timer.start(max(1, int(1000 / self._fps)))

    def _stop(self):
        self._playing = False
        self._play_timer.stop()
        self._btn_play.setText("▶  Play")

    def _advance_playback(self):
        nxt = self._list_idx + 1
        if nxt >= len(self._frame_list):
            self._stop()
            return
        self._show(nxt, fast=True)

    def _on_slider(self, value: int):
        if not self._playing:
            self._show(value)

    # ─────────────────────────────────────────────────────────────────────────
    # Live angle update
    # ─────────────────────────────────────────────────────────────────────────

    def _on_live_changed(self):
        kps = self.scene.get_kps_array()
        if kps is not None:
            angles = compute_frame_angles(kps)
            self._feat_panel.update_live(angles)
        else:
            self._feat_panel.update_live(None)

    # ─────────────────────────────────────────────────────────────────────────
    # Edit actions
    # ─────────────────────────────────────────────────────────────────────────

    def _on_pose_edited(self):
        if not self._frame_list or self._current_pid is None:
            return
        vf = self._frame_list[self._list_idx]
        state = self._player_states.setdefault(self._current_pid, PlayerState())
        state.edits[vf] = self.scene.get_keypoints()
        if vf in self._frame_map:
            self._frame_map[vf]["keypoints"] = state.edits[vf]
        self._lbl_edited.setText("● edited")

    def _undo(self):
        self.scene.undo_last()

    def _reset_frame(self):
        if not self._frame_list or self._current_pid is None:
            return
        vf = self._frame_list[self._list_idx]
        orig = self._orig_kps.get(vf)
        if orig is None:
            return
        self.scene.reset_to(orig)
        state = self._player_states.get(self._current_pid)
        if state:
            state.edits.pop(vf, None)
        if vf in self._frame_map:
            self._frame_map[vf]["keypoints"] = orig
        self._lbl_edited.setText("")
        self._status.showMessage(f"Frame {vf}: reset to original keypoints.")

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self):
        if self._current_pid is None or self._pose_data is None:
            return
        state = self._player_states.get(self._current_pid)
        if not state or not state.edits:
            self._status.showMessage("No edits to save for current player.")
            return
        if not self._edit_mode:
            self._status.showMessage("Switch to Editor Mode to save edits.")
            return
        n = 0
        for entry in self._pose_data.get("athlete_frames", []):
            vf = entry["frame"]
            if vf in state.edits:
                entry["keypoints"] = state.edits[vf]
                n += 1
        poses_path = self._players[self._current_pid]["poses_path"]
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = poses_path + f".bak_{ts_str}"
        shutil.copy2(poses_path, bak)
        with open(poses_path, "w") as f:
            json.dump(self._pose_data, f, indent=2)
        state.edits.clear()
        self._lbl_edited.setText("")
        self._status.showMessage(
            f"Saved {n} edited frame(s) → {Path(poses_path).name}  "
            f"(backup: {Path(bak).name})")

    # ─────────────────────────────────────────────────────────────────────────
    # Annotation: mark / clear / load / save
    # ─────────────────────────────────────────────────────────────────────────

    def _on_mark_requested(self, event_key: str):
        if not self._frame_list or self._current_pid is None:
            return
        vf = self._frame_list[self._list_idx]
        self._annotations.setdefault(self._current_pid, {})[event_key] = vf
        self._annotation_panel.update_display(self._annotations[self._current_pid])
        self._update_annotation_ticks()
        self._save_annotations(silent=True)
        self._status.showMessage(
            f"Annotated {event_key} → frame {vf}  ({self._current_pid})")

    def _on_clear_requested(self, event_key: str):
        if self._current_pid is None:
            return
        self._annotations.get(self._current_pid, {}).pop(event_key, None)
        self._annotation_panel.update_display(self._annotations.get(self._current_pid))
        self._update_annotation_ticks()
        self._save_annotations(silent=True)

    def _load_annotations_csv(self, path: str):
        """Load annotations from a CSV, supporting both column naming conventions."""
        p = Path(path)
        if not p.is_file():
            return
        COL_MAP = {
            "t1_contact_frame":      "towel1_contact",
            "t1_release_frame":      "towel1_release",
            "t2_contact_frame":      "towel2_contact",
            "t2_release_frame":      "towel2_release",
            "towel1_contact_frame":  "towel1_contact",
            "towel1_release_frame":  "towel1_release",
            "towel2_contact_frame":  "towel2_contact",
            "towel2_release_frame":  "towel2_release",
        }
        _BAD = {"not detected", "not found", ""}
        loaded = 0
        try:
            with open(p, newline="") as fh:
                for row in _csv.DictReader(fh):
                    pid = row.get("player_id", "").strip()
                    if not pid:
                        continue
                    entry: dict[str, int] = {}
                    for col, ev_key in COL_MAP.items():
                        val = row.get(col, "").strip()
                        if val and val.lower() not in _BAD:
                            try:
                                entry[ev_key] = int(float(val))
                            except ValueError:
                                pass
                    if entry:
                        self._annotations[pid] = entry
                        loaded += 1
        except Exception as e:
            self._status.showMessage(f"Could not load annotations CSV: {e}")
            return
        self._status.showMessage(
            f"Annotations loaded: {Path(path).name}  ({loaded} players)")

    def _save_annotations(self, silent: bool = False):
        """Write all annotations to the configured CSV, preserving existing rows."""
        if not self._annotations_csv:
            return
        path = Path(self._annotations_csv)
        cols = ["player_id"] + [k + "_frame" for k, _lbl, _col in ANNOTATION_EVENTS]
        # Preserve any rows on disk that are not yet in memory
        existing: dict[str, dict] = {}
        if path.is_file():
            try:
                with open(path, newline="") as fh:
                    for row in _csv.DictReader(fh):
                        pid = row.get("player_id", "")
                        if pid:
                            existing[pid] = dict(row)
            except Exception:
                pass
        for pid, ann in self._annotations.items():
            row = {"player_id": pid}
            for ev_key, _lbl, _col in ANNOTATION_EVENTS:
                row[ev_key + "_frame"] = ann.get(ev_key, "")
            existing[pid] = row
        try:
            with open(path, "w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for row in sorted(existing.values(), key=lambda r: r["player_id"]):
                    w.writerow(row)
        except Exception as e:
            self._status.showMessage(f"Could not save annotations: {e}")
            return
        msg = (f"{'Auto-saved' if silent else 'Saved'} "
               f"{len(self._annotations)} player(s) → {path.name}")
        self._annotation_panel.set_status(msg)
        if not silent:
            self._status.showMessage(msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Add optional data folders after session start
    # ─────────────────────────────────────────────────────────────────────────

    def _add_features_folder(self):
        start = str(Path(self._features_csv).parent) if self._features_csv else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select Features Folder", start)
        if not folder:
            return
        matches = []
        for csv_path in sorted(Path(folder).glob("*.csv")):
            try:
                with open(csv_path, newline="") as fh:
                    header = next(_csv.reader(fh), [])
                if "player_id" in header:
                    matches.append(csv_path)
            except Exception:
                pass
        if not matches:
            QMessageBox.warning(self, "No Features CSV",
                                "No CSV with a 'player_id' column was found in that folder.")
            return
        if len(matches) == 1:
            chosen = str(matches[0])
        else:
            from PyQt5.QtWidgets import QInputDialog
            names = [m.name for m in matches]
            name, ok = QInputDialog.getItem(self, "Select Features CSV",
                                            "Multiple CSVs found:", names, 0, False)
            if not ok:
                return
            chosen = str(Path(folder) / name)
        self._features_csv = chosen
        self._load_csvs()
        feat_ids = set(self._feat_df["player_id"].astype(str)) if self._feat_df is not None else set()
        for pid, info in self._players.items():
            info["has_features"] = pid in feat_ids
        self._player_panel.refresh_status(list(self._players.values()))
        if self._current_pid and self._feat_df is not None:
            rows = self._feat_df[self._feat_df["player_id"].astype(str) == self._current_pid]
            fr = rows.iloc[0].to_dict() if not rows.empty else None
            self._feat_panel.load_player_agg(fr, self._current_pid)
        self._status.showMessage(f"Features loaded: {Path(chosen).name}")

    def _add_anomaly_folder(self):
        start = str(Path(self._anomaly_csv).parent) if self._anomaly_csv else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select Anomaly Folder", start)
        if not folder:
            return
        matches = []
        for csv_path in sorted(Path(folder).glob("*.csv")):
            try:
                with open(csv_path, newline="") as fh:
                    header = next(_csv.reader(fh), [])
                if all(c in header for c in ["player_id", "frame", "is_low_prob"]):
                    matches.append(csv_path)
            except Exception:
                pass
        if not matches:
            QMessageBox.warning(self, "No Anomaly CSV",
                                "No CSV with 'player_id', 'frame', 'is_low_prob' columns found.")
            return
        chosen = str(matches[0])
        self._anomaly_csv = chosen
        self._load_csvs()
        anom_ids = set(self._anom_df["player_id"].astype(str)) if self._anom_df is not None else set()
        for pid, info in self._players.items():
            info["has_anomaly"] = pid in anom_ids
        self._player_panel.refresh_status(list(self._players.values()))
        if self._current_pid and self._anom_df is not None:
            self._load_player(self._current_pid)
        self._status.showMessage(f"Anomaly data loaded: {Path(chosen).name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        unsaved = [(pid, state) for pid, state in self._player_states.items()
                   if state.edits]
        if unsaved:
            n_players = len(unsaved)
            n_frames  = sum(len(s.edits) for _, s in unsaved)
            reply = QMessageBox.question(
                self, "Unsaved Edits",
                f"You have unsaved edits across <b>{n_players} player(s)</b> "
                f"({n_frames} frames total).\n\nSave the current player before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self._save()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        self._stop()
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Keypoint Editor — multi-player pose viewer and annotation tool.")
    parser.add_argument("--videos",    default="", help="Folder containing player .mp4 files")
    parser.add_argument("--poses",     default="", help="Folder containing player pose .json files")
    parser.add_argument("--features",  default="", help="Folder containing feature CSV(s)")
    parser.add_argument("--anomalies", default="", help="Folder containing anomaly CSV(s)")
    parser.add_argument("--events",    default="", help="Folder containing drill-event sidecar JSONs "
                                                        "(outputs/events/)")
    parser.add_argument("--edit",      action="store_true",
                        help="Start in Editor mode instead of View mode (default)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("KeypointEditor")
    app.setOrganizationName("NFL-Combine")
    _dark_palette(app)

    def _auto_csv(folder, required_cols):
        p = Path(folder)
        if not p.is_dir():
            return ""
        for csv_p in sorted(p.glob("*.csv")):
            try:
                with open(csv_p, newline="") as fh:
                    header = next(_csv.reader(fh), [])
                if all(c in header for c in required_cols):
                    return str(csv_p)
            except Exception:
                pass
        return ""

    settings = QSettings("NFL-Combine", "KeypointEditor")

    video_folder  = args.videos     or settings.value("last_video_folder",    "")
    poses_parent  = args.poses      or settings.value("last_poses_parent",
                                       settings.value("last_poses_folder", ""))
    feat_folder   = args.features   or settings.value("last_features_folder",  "")
    anom_folder   = args.anomalies  or settings.value("last_anomalies_folder",  "")
    events_dir    = args.events     or settings.value("last_events_dir", "")
    poses_3d_dir  = settings.value("last_poses_3d_dir", "")

    features_csv  = _auto_csv(feat_folder, ["player_id"]) if feat_folder else ""
    anomaly_csv   = _auto_csv(anom_folder, ["player_id", "frame", "is_low_prob"]) if anom_folder else ""

    window = KeypointEditor(
        video_folder=video_folder,
        poses_parent=poses_parent,
        features_csv=features_csv,
        anomaly_csv=anomaly_csv,
        poses_3d_dir=poses_3d_dir,
        events_dir=events_dir,
        start_in_edit=args.edit,
    )
    window.show()
    sys.exit(app.exec_())

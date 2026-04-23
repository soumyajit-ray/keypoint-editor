"""
setup_dialog.py — first-run session setup dialog.
"""

import sys
import os
import csv as _csv
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QDialogButtonBox, QGroupBox, QWidget, QComboBox, QFileDialog,
)


class SetupDialog(QDialog):
    """
    First-run session setup dialog.  Remembers last used paths via QSettings.
    """

    def __init__(self, parent=None,
                 init_videos="", init_poses="", init_poses_3d="",
                 init_features="", init_anomalies="", init_events="",
                 init_annotations=""):
        super().__init__(parent)
        self.setWindowTitle("Keypoint Editor — Session Setup")
        self.setMinimumWidth(560)
        self.setModal(True)

        self._settings = QSettings("NFL-Combine", "KeypointEditor")

        self.video_folder    = ""
        self.poses_folder    = ""
        self.poses_3d_folder = ""
        self.features_csv    = ""
        self.anomaly_csv     = ""
        self.events_folder   = ""
        self.annotations_csv = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 16, 20, 16)

        layout.addWidget(self._hdr("Session Setup", size=14, bold=True))
        layout.addWidget(self._hdr("Select folders containing pose data.  "
                                   "Fields marked * are required.", color="#a8c0d8", size=12))
        layout.addSpacing(4)

        # Video folder
        self._vid_edit, vid_grp = self._folder_row(
            "Video Folder  *",
            "Folder containing one .mp4 file per player  "
            "(named <player_id>.mp4, e.g. 2022_HUTCHINSON_AIDAN_DL31.mp4)",
            init_videos or self._settings.value("last_video_folder", ""),
        )
        layout.addWidget(vid_grp)

        # Poses parent folder
        self._pose_edit, pose_grp = self._folder_row(
            "Keypoints Parent Folder  *",
            "Parent folder containing per-model subfolders of pose JSONs  "
            "(e.g. outputs/ — subfolders like poses_yolo11x/, poses_rtmpose_body17/ "
            "each containing one .json per player)",
            init_poses or self._settings.value("last_poses_parent",
                          self._settings.value("last_poses_folder", "")),
        )
        layout.addWidget(pose_grp)

        # 3D Poses folder
        self._pose3d_edit, pose3d_grp = self._folder_row(
            "3D Poses Folder  (optional)",
            "Folder containing 3D pose JSONs from a lifting model  "
            "(e.g. outputs/poses_3d_motionagformer/ or outputs/poses_3d_4dhumans/).  "
            "Use convert_4dhumans.py to convert 4D-Humans pkl outputs first.  "
            "Leave empty to skip.",
            init_poses_3d or self._settings.value("last_poses_3d_dir", ""),
            required=False,
        )
        layout.addWidget(pose3d_grp)

        # Features folder
        self._feat_edit, feat_grp = self._folder_row(
            "Features Folder  (optional)",
            "Folder containing a feature CSV with a 'player_id' column  "
            "(e.g. outputs/features/).  Leave empty to skip.",
            init_features or self._settings.value("last_features_folder", ""),
            required=False,
        )
        layout.addWidget(feat_grp)

        self._feat_csv_row = QWidget()
        feat_csv_h = QHBoxLayout(self._feat_csv_row)
        feat_csv_h.setContentsMargins(0, 0, 0, 0)
        feat_csv_h.addWidget(QLabel("Select CSV:"))
        self._feat_csv_combo = QComboBox()
        self._feat_csv_combo.setToolTip("Choose which feature CSV to use")
        feat_csv_h.addWidget(self._feat_csv_combo, 1)
        self._feat_csv_row.hide()
        layout.addWidget(self._feat_csv_row)

        self._feat_status = QLabel("")
        self._feat_status.setStyleSheet("color:#a8c0d8; font-size:12px; font-style:italic;")
        layout.addWidget(self._feat_status)

        # Anomaly folder
        self._anom_edit, anom_grp = self._folder_row(
            "Anomaly Folder  (optional)",
            "Folder containing an anomaly CSV with columns: player_id, frame, is_low_prob  "
            "(e.g. outputs/features/).  Leave empty to skip.",
            init_anomalies or self._settings.value("last_anomalies_folder", ""),
            required=False,
        )
        layout.addWidget(anom_grp)

        self._anom_csv_row = QWidget()
        anom_csv_h = QHBoxLayout(self._anom_csv_row)
        anom_csv_h.setContentsMargins(0, 0, 0, 0)
        anom_csv_h.addWidget(QLabel("Select CSV:"))
        self._anom_csv_combo = QComboBox()
        self._anom_csv_combo.setToolTip("Choose which anomaly CSV to use")
        anom_csv_h.addWidget(self._anom_csv_combo, 1)
        self._anom_csv_row.hide()
        layout.addWidget(self._anom_csv_row)

        self._anom_status = QLabel("")
        self._anom_status.setStyleSheet("color:#a8c0d8; font-size:12px; font-style:italic;")
        layout.addWidget(self._anom_status)

        # Events folder
        self._events_edit, events_grp = self._folder_row(
            "Events Folder  (optional)",
            "Folder containing per-player drill-event sidecar JSONs "
            "(<player_id>_events.json, generated by event_detection.py).  "
            "Shows ball lift, towel pickup/drop and finish markers on the timeline.  "
            "Leave empty to skip.",
            init_events or self._settings.value("last_events_dir", ""),
            required=False,
        )
        layout.addWidget(events_grp)

        self._events_status = QLabel("")
        self._events_status.setStyleSheet("color:#a8c0d8; font-size:12px; font-style:italic;")
        layout.addWidget(self._events_status)

        self._events_edit.textChanged.connect(self._on_events_folder_changed)
        if self._events_edit.text():
            self._on_events_folder_changed(self._events_edit.text())

        # Annotations CSV
        self._ann_edit, ann_grp = self._file_row(
            "Annotations CSV  (optional)",
            "CSV to read/write manual towel frame annotations.  "
            "Accepts docs/towel_frames.csv column style (t1_contact_frame …) "
            "and native style (towel1_contact_frame …).  Leave empty to skip.",
            init_annotations or self._settings.value("last_annotations_csv", ""),
        )
        layout.addWidget(ann_grp)

        self._ann_status = QLabel("")
        self._ann_status.setStyleSheet("color:#a8c0d8; font-size:12px; font-style:italic;")
        layout.addWidget(self._ann_status)

        self._ann_edit.textChanged.connect(self._on_ann_csv_changed)
        if self._ann_edit.text():
            self._on_ann_csv_changed(self._ann_edit.text())

        # Buttons
        layout.addSpacing(6)
        btns = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        btns.button(QDialogButtonBox.Ok).setText("Open Session")
        btns.button(QDialogButtonBox.Ok).setToolTip(
            "Load the selected folders and open the player browser")
        btns.button(QDialogButtonBox.Cancel).setToolTip("Quit the application")
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        self._ok_btn = btns.button(QDialogButtonBox.Ok)
        layout.addWidget(btns)

        self._vid_edit.textChanged.connect(self._validate)
        self._pose_edit.textChanged.connect(self._validate)
        self._feat_edit.textChanged.connect(self._on_feat_folder_changed)
        self._anom_edit.textChanged.connect(self._on_anom_folder_changed)

        if self._feat_edit.text():
            self._on_feat_folder_changed(self._feat_edit.text())
        if self._anom_edit.text():
            self._on_anom_folder_changed(self._anom_edit.text())

        self._validate()

    # ── UI helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _hdr(text, size=12, bold=False, color="#ccc") -> QLabel:
        lbl = QLabel(text)
        style = f"color:{color}; font-size:{size}px;"
        if bold:
            style += " font-weight:bold;"
        lbl.setStyleSheet(style)
        return lbl

    def _folder_row(self, title: str, hint: str, default: str = "",
                    required: bool = True):
        grp = QGroupBox(title)
        grp.setStyleSheet(
            "QGroupBox { color:#d4d4d4; font-size:12px; border:1px solid #444; "
            "border-radius:4px; margin-top:8px; padding-top:8px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
        )
        v = QVBoxLayout(grp)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        h = QHBoxLayout()
        edit = QLineEdit(default)
        edit.setPlaceholderText("Click Browse… to select a folder")
        edit.setToolTip(hint)
        browse = QPushButton("Browse…")
        browse.setFixedWidth(80)
        browse.setToolTip(f"Open folder browser — {hint}")
        browse.clicked.connect(lambda: self._browse(edit))
        h.addWidget(edit)
        h.addWidget(browse)
        v.addLayout(h)
        hint_lbl = QLabel(hint)
        hint_lbl.setStyleSheet("color:#a8c0d8; font-size:12px;")
        hint_lbl.setWordWrap(True)
        v.addWidget(hint_lbl)
        return edit, grp

    def _browse(self, edit: QLineEdit):
        start = edit.text() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if d:
            edit.setText(d)

    def _browse_file(self, edit: QLineEdit):
        start = str(Path(edit.text()).parent) if edit.text() else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", start, "CSV files (*.csv)")
        if path:
            edit.setText(path)

    def _file_row(self, title: str, hint: str, default: str = ""):
        """Like _folder_row but the Browse button opens a file picker."""
        grp = QGroupBox(title)
        grp.setStyleSheet(
            "QGroupBox { color:#d4d4d4; font-size:12px; border:1px solid #444; "
            "border-radius:4px; margin-top:8px; padding-top:8px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
        )
        v = QVBoxLayout(grp)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        h = QHBoxLayout()
        edit = QLineEdit(default)
        edit.setPlaceholderText("Click Browse… to select a CSV file")
        edit.setToolTip(hint)
        browse = QPushButton("Browse…")
        browse.setFixedWidth(80)
        browse.setToolTip(f"Open file browser — {hint}")
        browse.clicked.connect(lambda: self._browse_file(edit))
        h.addWidget(edit)
        h.addWidget(browse)
        v.addLayout(h)
        hint_lbl = QLabel(hint)
        hint_lbl.setStyleSheet("color:#a8c0d8; font-size:12px;")
        hint_lbl.setWordWrap(True)
        v.addWidget(hint_lbl)
        return edit, grp

    # ── CSV detection ─────────────────────────────────────────────────────────

    def _find_csvs(self, folder: str, required_cols: list) -> list:
        p = Path(folder)
        if not p.is_dir():
            return []
        matches = []
        for csv_path in sorted(p.glob("*.csv")):
            try:
                with open(csv_path, newline="") as f:
                    header = next(_csv.reader(f), [])
                if all(col in header for col in required_cols):
                    matches.append(csv_path)
            except Exception:
                pass
        return matches

    def _on_feat_folder_changed(self, text: str):
        matches = self._find_csvs(text, ["player_id"])
        self._feat_csv_combo.clear()
        if not text.strip():
            self._feat_csv_row.hide()
            self._feat_status.setText("")
        elif not matches:
            self._feat_csv_row.hide()
            self._feat_status.setText("No CSVs with 'player_id' column found in this folder")
            self._feat_status.setStyleSheet("color:#f88; font-size:12px; font-style:italic;")
        elif len(matches) == 1:
            self._feat_csv_row.hide()
            self._feat_status.setText(f"Found: {matches[0].name}")
            self._feat_status.setStyleSheet("color:#69f0ae; font-size:12px; font-style:italic;")
            self._feat_csv_combo.addItem(str(matches[0]))
        else:
            for m in matches:
                self._feat_csv_combo.addItem(str(m), str(m))
            self._feat_csv_row.show()
            self._feat_status.setText(f"Found {len(matches)} CSVs — select one:")
            self._feat_status.setStyleSheet("color:#ffcc02; font-size:12px; font-style:italic;")

    def _on_anom_folder_changed(self, text: str):
        matches = self._find_csvs(text, ["player_id", "frame", "is_low_prob"])
        self._anom_csv_combo.clear()
        if not text.strip():
            self._anom_csv_row.hide()
            self._anom_status.setText("")
        elif not matches:
            self._anom_csv_row.hide()
            self._anom_status.setText(
                "No CSVs with 'player_id', 'frame', 'is_low_prob' columns found")
            self._anom_status.setStyleSheet("color:#f88; font-size:12px; font-style:italic;")
        elif len(matches) == 1:
            self._anom_csv_row.hide()
            self._anom_status.setText(f"Found: {matches[0].name}")
            self._anom_status.setStyleSheet("color:#69f0ae; font-size:12px; font-style:italic;")
            self._anom_csv_combo.addItem(str(matches[0]))
        else:
            for m in matches:
                self._anom_csv_combo.addItem(str(m), str(m))
            self._anom_csv_row.show()
            self._anom_status.setText(f"Found {len(matches)} CSVs — select one:")
            self._anom_status.setStyleSheet("color:#ffcc02; font-size:12px; font-style:italic;")

    def _on_events_folder_changed(self, text: str):
        p = Path(text.strip())
        if not text.strip():
            self._events_status.setText("")
        elif not p.is_dir():
            self._events_status.setText("Folder not found")
            self._events_status.setStyleSheet("color:#f88; font-size:12px; font-style:italic;")
        else:
            n = len(list(p.glob("*_events.json")))
            if n == 0:
                self._events_status.setText("No *_events.json files found in this folder")
                self._events_status.setStyleSheet("color:#f88; font-size:12px; font-style:italic;")
            else:
                self._events_status.setText(f"Found {n} event file(s)")
                self._events_status.setStyleSheet(
                    "color:#69f0ae; font-size:12px; font-style:italic;")

    def _on_ann_csv_changed(self, text: str):
        p = Path(text.strip())
        if not text.strip():
            self._ann_status.setText("")
            return
        if not p.is_file():
            self._ann_status.setText("File not found")
            self._ann_status.setStyleSheet("color:#f88; font-size:12px; font-style:italic;")
            return
        try:
            with open(p, newline="") as f:
                reader = _csv.reader(f)
                header = next(reader, [])
                n = sum(1 for _ in reader)
            known_cols = {
                "t1_contact_frame", "t1_release_frame",
                "t2_contact_frame", "t2_release_frame",
                "towel1_contact_frame", "towel1_release_frame",
                "towel2_contact_frame", "towel2_release_frame",
            }
            if "player_id" not in header or not known_cols.intersection(header):
                self._ann_status.setText(
                    "No recognised annotation columns found (expected player_id + t1_*/towel1_* …)")
                self._ann_status.setStyleSheet("color:#f88; font-size:12px; font-style:italic;")
            else:
                self._ann_status.setText(f"Found {n} player row(s)")
                self._ann_status.setStyleSheet(
                    "color:#69f0ae; font-size:12px; font-style:italic;")
        except Exception as e:
            self._ann_status.setText(f"Could not read file: {e}")
            self._ann_status.setStyleSheet("color:#f88; font-size:12px; font-style:italic;")

    def _validate(self):
        ok = (Path(self._vid_edit.text()).is_dir() and
              Path(self._pose_edit.text()).is_dir())
        self._ok_btn.setEnabled(ok)

    def _accept(self):
        self.video_folder    = self._vid_edit.text().strip()
        self.poses_folder    = self._pose_edit.text().strip()
        self.poses_3d_folder = self._pose3d_edit.text().strip()
        self.events_folder   = self._events_edit.text().strip()
        if self._feat_csv_combo.count() == 1:
            self.features_csv = self._feat_csv_combo.itemText(0)
        elif self._feat_csv_combo.count() > 1:
            self.features_csv = self._feat_csv_combo.currentText()
        else:
            self.features_csv = ""
        if self._anom_csv_combo.count() == 1:
            self.anomaly_csv = self._anom_csv_combo.itemText(0)
        elif self._anom_csv_combo.count() > 1:
            self.anomaly_csv = self._anom_csv_combo.currentText()
        else:
            self.anomaly_csv = ""
        self._settings.setValue("last_video_folder",    self.video_folder)
        self._settings.setValue("last_poses_parent",    self.poses_folder)
        self._settings.setValue("last_poses_3d_dir",    self.poses_3d_folder)
        self._settings.setValue("last_features_folder",
                                Path(self.features_csv).parent.as_posix()
                                if self.features_csv else "")
        self._settings.setValue("last_anomalies_folder",
                                Path(self.anomaly_csv).parent.as_posix()
                                if self.anomaly_csv else "")
        self._settings.setValue("last_events_dir", self.events_folder)
        self.annotations_csv = self._ann_edit.text().strip()
        self._settings.setValue("last_annotations_csv", self.annotations_csv)
        self.accept()

"""
annotation_panel.py — manual towel-event annotation dock panel.

Constants:
    ANNOTATION_EVENTS  — ordered list of (event_key, label, hex_color)

Classes:
    AnnotationPanel    — QWidget dock panel with Mark / Clear buttons per event
"""

import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)


# Event key names match EVENT_TICK_COLORS in anomaly_bar.py so stamped frames
# use the same timeline-bar colors as auto-detected events.
ANNOTATION_EVENTS: list[tuple[str, str, str]] = [
    ("towel1_contact",  "T1 Pickup",  "#78FF50"),
    ("towel1_release",  "T1 Drop",    "#00C853"),
    ("towel2_contact",  "T2 Pickup",  "#50C8FF"),
    ("towel2_release",  "T2 Drop",    "#0090FF"),
]


class AnnotationPanel(QWidget):
    """
    Dock panel for manually stamping the four towel events.

    Signals
    -------
    mark_requested(event_key)
        Emitted when the user clicks Mark or presses the key shortcut.
    clear_requested(event_key)
        Emitted when the user clicks the Clear (✕) button.
    save_requested()
        Emitted when the user clicks the explicit Save CSV button.
    """

    mark_requested  = pyqtSignal(str)
    clear_requested = pyqtSignal(str)
    save_requested  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # Title
        title = QLabel("Towel Annotations")
        title.setStyleSheet("font-size:13px; font-weight:bold; color:#ddd;")
        root.addWidget(title)

        hint = QLabel("Keys 1–4 mark events when this panel is open")
        hint.setStyleSheet("font-size:11px; color:#888; font-style:italic;")
        root.addWidget(hint)

        # Current frame hint
        self._lbl_frame = QLabel("Current frame: —")
        self._lbl_frame.setStyleSheet("font-size:12px; color:#aaa;")
        root.addWidget(self._lbl_frame)

        # Separator
        root.addWidget(self._make_separator())

        # One row per event
        self._frame_labels:  dict[str, QLabel]      = {}
        self._clear_buttons: dict[str, QPushButton]  = {}

        for ev_key, label, color in ANNOTATION_EVENTS:
            row = QHBoxLayout()
            row.setSpacing(4)

            name_lbl = QLabel(label)
            name_lbl.setFixedWidth(72)
            name_lbl.setStyleSheet(f"font-size:12px; font-weight:bold; color:{color};")
            row.addWidget(name_lbl)

            frame_lbl = QLabel("—")
            frame_lbl.setFixedWidth(54)
            frame_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            frame_lbl.setStyleSheet("font-size:12px; color:#ccc; font-family:monospace;")
            row.addWidget(frame_lbl)
            self._frame_labels[ev_key] = frame_lbl

            mark_btn = QPushButton("Mark")
            mark_btn.setFixedWidth(46)
            mark_btn.setStyleSheet(
                "QPushButton { font-size:11px; padding:2px 4px; }"
            )
            mark_btn.clicked.connect(lambda _checked, k=ev_key: self.mark_requested.emit(k))
            row.addWidget(mark_btn)

            clear_btn = QPushButton("✕")
            clear_btn.setFixedWidth(26)
            clear_btn.setEnabled(False)
            clear_btn.setStyleSheet(
                "QPushButton { font-size:11px; padding:2px 2px; color:#f88; }"
                "QPushButton:disabled { color:#555; }"
            )
            clear_btn.clicked.connect(lambda _checked, k=ev_key: self.clear_requested.emit(k))
            row.addWidget(clear_btn)
            self._clear_buttons[ev_key] = clear_btn

            root.addLayout(row)

        root.addWidget(self._make_separator())

        # Save button
        save_btn = QPushButton("Save CSV   Ctrl+Shift+S")
        save_btn.setStyleSheet(
            "QPushButton { background:#2a7; color:white; font-weight:bold; font-size:12px; "
            "padding:4px; }"
            "QPushButton:hover { background:#3b8; }"
        )
        save_btn.clicked.connect(self.save_requested.emit)
        root.addWidget(save_btn)

        # Status label
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet(
            "font-size:11px; color:#69f0ae; font-style:italic;"
        )
        self._lbl_status.setWordWrap(True)
        root.addWidget(self._lbl_status)

        root.addStretch()

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_display(self, annotations: dict | None):
        """Refresh frame-number labels and enable/disable Clear buttons."""
        for ev_key, lbl in self._frame_labels.items():
            vf = annotations.get(ev_key) if annotations else None
            if vf is not None:
                lbl.setText(str(vf))
                self._clear_buttons[ev_key].setEnabled(True)
            else:
                lbl.setText("—")
                self._clear_buttons[ev_key].setEnabled(False)

    def set_current_frame(self, vframe: int):
        """Update the 'Current frame: N' hint label."""
        self._lbl_frame.setText(f"Current frame: {vframe}")

    def set_status(self, msg: str):
        """Update the status label at the bottom of the panel."""
        self._lbl_status.setText(msg)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color:#444;")
        return line

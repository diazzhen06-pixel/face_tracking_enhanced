"""
analytics.py — CSV analytics reports and occupancy helpers
"""

import csv
import os
from datetime import datetime

DATA_DIR = "data"

# ------------------------------------------------------------------ #
# Occupancy classification (project spec)
# ------------------------------------------------------------------ #
OCCUPANCY_LEVELS = [
    (5,  "LOW",    (0, 200, 80)),
    (15, "MEDIUM", (0, 165, 255)),
    (999,"HIGH",   (0, 0, 220)),
]

def occupancy_status(count):
    for threshold, label, _ in OCCUPANCY_LEVELS:
        if count <= threshold:
            return label
    return "HIGH"

def occupancy_color(count):
    for threshold, _, color in OCCUPANCY_LEVELS:
        if count <= threshold:
            return color
    return (0, 0, 220)


# ------------------------------------------------------------------ #
# Per-session stats
# ------------------------------------------------------------------ #
class SessionStats:
    def __init__(self):
        self.start_time       = datetime.now()
        self.total_detections = 0
        self.peak_count       = 0
        self.alert_count      = 0
        self.frame_count      = 0

    def update(self, current_count):
        self.frame_count      += 1
        self.total_detections += current_count
        if current_count > self.peak_count:
            self.peak_count = current_count

    def avg_count(self):
        if self.frame_count == 0:
            return 0.0
        return self.total_detections / self.frame_count

    def elapsed_seconds(self):
        return (datetime.now() - self.start_time).total_seconds()


# ------------------------------------------------------------------ #
# CSV analytics report
# ------------------------------------------------------------------ #
REPORT_PATH = os.path.join(DATA_DIR, "analytics_report.csv")

def _ensure_report_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "frame", "face_count",
                "occupancy_status", "event_type", "object_id", "notes"
            ])

def log_analytics(frame_num, face_count, event_type="", object_id="", notes=""):
    _ensure_report_header()
    with open(REPORT_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_num,
            face_count,
            occupancy_status(face_count),
            event_type,
            object_id,
            notes,
        ])

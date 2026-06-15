"""
logger.py — Event logger for restricted-area alerts
"""

import csv
import os
from datetime import datetime

DATA_DIR = "data"
EVENT_LOG = os.path.join(DATA_DIR, "event_log.csv")

def _ensure_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(EVENT_LOG):
        with open(EVENT_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "object_id", "event", "notes"])

def log_event(face_id, event, notes=""):
    _ensure_header()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(EVENT_LOG, "a", newline="") as f:
        csv.writer(f).writerow([ts, face_id, event, notes])
    print(f"[EVENT] {ts}  ID={face_id}  {event}  {notes}")
    return ts

"""
main.py — Class Surveillance & Analytics System
Enhanced Face Tracking Edition

Features:
  • YuNet face detection (GPU via OpenCL when available)
  • MOG2 background subtraction for motion gating
  • Stable centroid tracker — persistent IDs, disappear/reappear lifecycle
  • Face trajectory / history trail
  • Restricted-area ENTER and EXIT detection
  • Screenshot capture on alerts
  • CSV analytics + event logs
  • FPS monitoring
  • Full HUD dashboard overlay

Controls:
  ESC / Q — quit
  S       — manual screenshot
  R       — reset session stats
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

from .tracker        import CentroidTracker
from .analytics      import SessionStats, log_analytics
from .logger         import log_event
from .dashboard      import draw_dashboard, draw_desktop_command_view
from .face_detector  import GpuFaceDetector
from .camera         import ThreadedCapture


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Video source: 0 = built-in webcam, 1 = external USB camera, URL, or video file path.
def _video_source():
    value = os.environ.get("VIDEO_SOURCE", "0").strip()
    if value.isdigit():
        return int(value)
    return value


SOURCE = _video_source()

# Restricted area  (x1, y1, x2, y2) in pixels — adjust to your frame size
#RESTRICTED_RECT = (100, 50, 700, 500)

# ── Performance ──────────────────────────────────────────────────────────────
MAX_FPS          = False   # Master switch: enables all speed optimizations below

USE_GPU          = None   # None = auto-pick fastest, True = force GPU, False = force CPU
FACE_SCORE       = 0.7
DETECT_WIDTH     = 320 if MAX_FPS else 0   # Inference width (0 = full frame)
DETECT_INTERVAL  = 2 if MAX_FPS else 1     # Run detector every N frames
THREADED_CAPTURE = MAX_FPS
LIGHT_HUD        = MAX_FPS
WINDOW_WIDTH     = 0 if MAX_FPS else 1366  # 0 = native camera size (no upscale)
WINDOW_HEIGHT    = 0 if MAX_FPS else 768

# Camera capture — MJPG codec unlocks higher USB camera FPS on Windows
CAMERA_WIDTH     = 640
CAMERA_HEIGHT    = 480
CAMERA_FOURCC    = "MJPG" if MAX_FPS else ""

# MOG2 — motion gating: only pass face detections that overlap motion regions
MOG2_HISTORY     = 200
MOG2_VAR_THRESH  = 40
MOTION_GATE      = False   # Set True to gate detections through MOG2 mask

# Tracker
MAX_DISAPPEARED = 40   # frames before ID is retired
MAX_DISTANCE    = 90   # pixels to match centroid to existing ID
HISTORY_LEN     = 50   # trajectory length

# Alert banner duration in seconds
ALERT_DURATION  = 4.0

# Screenshot output directory
SCREENSHOT_DIR = os.path.join("data", "screenshots")

# CSV analytics — log every N frames (0 = every frame)
ANALYTICS_INTERVAL = 5

# ═══════════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════════

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

face_detector = GpuFaceDetector(
    use_gpu=USE_GPU,
    score_threshold=FACE_SCORE,
    detect_width=DETECT_WIDTH,
    warmup_size=(DETECT_WIDTH or CAMERA_WIDTH, DETECT_WIDTH and int(DETECT_WIDTH * 0.75) or CAMERA_HEIGHT),
)

tracker = CentroidTracker(
    max_disappeared=MAX_DISAPPEARED,
    max_distance=MAX_DISTANCE,
    history_len=HISTORY_LEN,
)

bg_subtractor = (
    cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESH,
        detectShadows=False,
    )
    if MOTION_GATE
    else None
)

stats        = SessionStats()
_cap_backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

if isinstance(SOURCE, int) and THREADED_CAPTURE:
    cap = ThreadedCapture(SOURCE, _cap_backend, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOURCC)
    ret = False
    for _ in range(50):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.05)
    if not ret:
        raise RuntimeError(f"Cannot open video source: {SOURCE}")
else:
    cap = cv2.VideoCapture(SOURCE, _cap_backend) if isinstance(SOURCE, int) else cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {SOURCE}")
    if isinstance(SOURCE, int):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if CAMERA_FOURCC:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
        if CAMERA_WIDTH > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        if CAMERA_HEIGHT > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Track zone state: entered / exited
in_zone_ids       = set()   # IDs currently inside the restricted area
ever_entered      = set()   # IDs that entered at some point (for exit detection)
zone_entry_count  = 0     # Total times anyone entered the restricted area

alert_msg   = ""
alert_until = 0.0

frame_num   = 0
fps         = 0.0
fps_timer   = time.time()
fps_counter = 0


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def take_screenshot(frame, reason="manual"):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    fn  = os.path.join(SCREENSHOT_DIR, f"alert_{ts}_{reason}.jpg")
    cv2.imwrite(fn, frame)
    print(f"[SCREENSHOT] Saved -> {fn}")
    return fn


def point_in_rect(cx, cy, rect):
    x1, y1, x2, y2 = rect
    return x1 < cx < x2 and y1 < cy < y2


def fire_alert(frame, oid, event_label, notes=""):
    global alert_msg, alert_until
    ts = log_event(oid, event_label, notes)
    stats.alert_count += 1
    log_analytics(frame_num, len(tracker.objects), event_label, oid, notes)
    alert_msg   = f"Person ID {oid}-{event_label}"
    alert_until = time.time() + ALERT_DURATION
    take_screenshot(frame, reason=event_label.replace(" ", "_"))


# ═══════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════

print("Class Surveillance started.")
if MAX_FPS:
    print(f"  MAX FPS mode - detect every {DETECT_INTERVAL} frames at {DETECT_WIDTH}px wide")
print("  ESC / Q - quit   |   S - screenshot   |   R - reset stats")

cv2.namedWindow("Smart Campus Command View", cv2.WINDOW_NORMAL)
if WINDOW_WIDTH > 0 and WINDOW_HEIGHT > 0:
    cv2.resizeWindow("Smart Campus Command View", WINDOW_WIDTH, WINDOW_HEIGHT)

last_rects = []
restricted_rect = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of stream or camera error.")
        break

    h, w = frame.shape[:2]

    if restricted_rect is None:
        zone_w = int(w * 0.55)
        zone_h = int(h * 0.75)
        x1 = (w - zone_w) // 2
        y1 = (h - zone_h) // 2
        restricted_rect = (x1, y1, x1 + zone_w, y1 + zone_h)

    RESTRICTED_RECT = restricted_rect

    frame_num += 1

    # ── FPS calculation ────────────────────────────────────────────────
    fps_counter += 1
    now = time.time()
    if now - fps_timer >= 1.0:
        fps         = fps_counter / (now - fps_timer)
        fps_counter = 0
        fps_timer   = now

    # ── MOG2 motion mask (for optional gating) ─────────────────────────
    if MOTION_GATE:
        fg_mask = bg_subtractor.apply(frame)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # ── Face detection (every DETECT_INTERVAL frames) ──────────────────
    if DETECT_INTERVAL <= 1 or frame_num % DETECT_INTERVAL == 0:
        last_rects = face_detector.detect(frame)

    rects = last_rects
    if MOTION_GATE:
        rects = []
        for (x, y, fw, fh) in last_rects:
            roi = fg_mask[y:y+fh, x:x+fw]
            motion_ratio = np.count_nonzero(roi) / float(roi.size + 1e-6)
            if motion_ratio >= 0.05:
                rects.append((x, y, fw, fh))

    if not LIGHT_HUD:
        for (x, y, fw, fh) in rects:
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 220, 80), 2)

    # ── Update tracker ─────────────────────────────────────────────────
    objects = tracker.update(rects)

    # If a person vanishes while marked inside, close the zone event cleanly.
    for oid in tracker.last_deregistered_ids:
        if oid in in_zone_ids:
            fire_alert(frame, oid, "Exited Restricted Area", "Track expired after missed detections")
            in_zone_ids.remove(oid)

    # ── Zone detection: ENTER and EXIT ─────────────────────────────────
    current_in_zone = set()
    for oid, (cx, cy) in objects.items():
        if point_in_rect(cx, cy, RESTRICTED_RECT):
            current_in_zone.add(oid)
            if oid not in in_zone_ids:
                # New entry
                ever_entered.add(oid)
                zone_entry_count += 1
                fire_alert(frame, oid, "Entered Restricted Area")
        else:
            if oid in in_zone_ids:
                # Exit event
                fire_alert(frame, oid, "Exited Restricted Area")

    in_zone_ids = current_in_zone

    # ── Session stats update ────────────────────────────────────────────
    stats.update(len(objects))

    # ── Periodic CSV analytics ─────────────────────────────────────────
    if ANALYTICS_INTERVAL > 0 and frame_num % ANALYTICS_INTERVAL == 0:
        log_analytics(frame_num, len(objects))

    # ── Alert timer ────────────────────────────────────────────────────
    remaining_alert = max(0.0, alert_until - time.time())

    # ── Draw dashboard ─────────────────────────────────────────────────
    if LIGHT_HUD:
        display_frame = draw_dashboard(
            frame, tracker, stats, fps,
            RESTRICTED_RECT, in_zone_ids,
            alert_msg, remaining_alert,
            lite=True,
        )
    else:
        display_frame = draw_desktop_command_view(
            frame, tracker, stats, fps,
            RESTRICTED_RECT, in_zone_ids,
            alert_msg, remaining_alert,
            zone_entry_count=zone_entry_count,
            status="RUNNING",
        )

    # ── Show ────────────────────────────────────────────────────────────
    cv2.imshow("Smart Campus Command View", display_frame)

    # ── Keyboard ────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q"), ord("Q")):
        break
    elif key in (ord("s"), ord("S")):
        take_screenshot(display_frame, "manual")
    elif key in (ord("r"), ord("R")):
        stats = SessionStats()
        tracker = CentroidTracker(MAX_DISAPPEARED, MAX_DISTANCE, HISTORY_LEN)
        in_zone_ids.clear()
        ever_entered.clear()
        zone_entry_count = 0
        print("[RESET] Session stats and tracker cleared.")


# ── Cleanup ────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

# Final session summary to CSV
log_analytics(frame_num, 0, "SESSION_END", "",
              f"peak={stats.peak_count} alerts={stats.alert_count} "
              f"avg={stats.avg_count():.2f} elapsed={stats.elapsed_seconds():.1f}s")

print("\n-- Session Summary ------------------------------")
print(f"  Total frames   : {frame_num}")
print(f"  Peak face count: {stats.peak_count}")
print(f"  Avg face count : {stats.avg_count():.2f}")
print(f"  Total alerts   : {stats.alert_count}")
print(f"  Zone entries   : {zone_entry_count}")
print(f"  Session length : {stats.elapsed_seconds():.1f}s")
print(f"  Analytics CSV  : data/analytics_report.csv")
print(f"  Event log CSV  : data/event_log.csv")
print(f"  Screenshots    : {SCREENSHOT_DIR}/")

"""Flask web dashboard for the face tracking surveillance system."""

import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from .analytics import SessionStats, log_analytics
from .camera import ThreadedCapture
from .dashboard import draw_dashboard
from .face_detector import GpuFaceDetector
from .logger import log_event
from .tracker import CentroidTracker


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _video_source():
    value = os.environ.get("VIDEO_SOURCE", "0").strip()
    if value.isdigit():
        return int(value)
    return value


SOURCE = _video_source()
USE_GPU = None
FACE_SCORE = float(os.environ.get("FACE_SCORE", "0.7"))
DETECT_WIDTH = _env_int("DETECT_WIDTH", 0)
DETECT_INTERVAL = _env_int("DETECT_INTERVAL", 1)
THREADED_CAPTURE = _env_bool("THREADED_CAPTURE", False)
LIGHT_HUD = _env_bool("LIGHT_HUD", False)
CAMERA_WIDTH = _env_int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT = _env_int("CAMERA_HEIGHT", 480)
CAMERA_FOURCC = ""
MOG2_HISTORY = 200
MOG2_VAR_THRESH = 40
MOTION_GATE = _env_bool("MOTION_GATE", False)
MAX_DISAPPEARED = 40
MAX_DISTANCE = 90
HISTORY_LEN = 50
ALERT_DURATION = 4.0
SCREENSHOT_DIR = os.path.join("data", "screenshots")
ANALYTICS_INTERVAL = 5


class WebSurveillanceEngine:
    def __init__(self):
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.latest_jpeg = None

        self.face_detector = GpuFaceDetector(
            use_gpu=USE_GPU,
            score_threshold=FACE_SCORE,
            detect_width=DETECT_WIDTH,
            warmup_size=(DETECT_WIDTH or CAMERA_WIDTH, DETECT_WIDTH and int(DETECT_WIDTH * 0.75) or CAMERA_HEIGHT),
        )
        self.tracker = CentroidTracker(MAX_DISAPPEARED, MAX_DISTANCE, HISTORY_LEN)
        self.bg_subtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=MOG2_HISTORY,
                varThreshold=MOG2_VAR_THRESH,
                detectShadows=False,
            )
            if MOTION_GATE
            else None
        )
        self.stats = SessionStats()
        self.in_zone_ids = set()
        self.ever_entered = set()
        self.zone_entry_count = 0
        self.alert_msg = ""
        self.alert_until = 0.0
        self.frame_num = 0
        self.fps = 0.0
        self.fps_timer = time.time()
        self.fps_counter = 0
        self.last_rects = []
        self.restricted_rect = None
        self.status = "starting"

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def snapshot(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
        return self._take_screenshot(frame, "manual")

    def metrics(self):
        with self.lock:
            return {
                "status": self.status,
                "frame": self.frame_num,
                "fps": round(self.fps, 1),
                "faces": len(self.tracker.objects),
                "insideZone": len(self.in_zone_ids),
                "occupancy": self._occupancy(),
                "peak": self.stats.peak_count,
                "alerts": self.stats.alert_count,
                "zoneEntries": self.zone_entry_count,
                "alertMessage": self.alert_msg if time.time() < self.alert_until else "",
                "elapsedSeconds": round(self.stats.elapsed_seconds(), 1),
            }

    def jpeg_frames(self):
        while True:
            with self.lock:
                jpeg = self.latest_jpeg
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(0.01)

    def _occupancy(self):
        face_count = len(self.tracker.objects)
        if face_count <= 5:
            return "LOW"
        if face_count <= 15:
            return "MEDIUM"
        return "HIGH"

    def _take_screenshot(self, frame, reason="manual"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        fn = os.path.join(SCREENSHOT_DIR, f"alert_{ts}_{reason}.jpg")
        cv2.imwrite(fn, frame)
        return fn

    def _point_in_rect(self, cx, cy, rect):
        x1, y1, x2, y2 = rect
        return x1 < cx < x2 and y1 < cy < y2

    def _fire_alert(self, frame, oid, event_label, notes=""):
        log_event(oid, event_label, notes)
        self.stats.alert_count += 1
        log_analytics(self.frame_num, len(self.tracker.objects), event_label, oid, notes)
        self.alert_msg = f"Person ID {oid}-{event_label}"
        self.alert_until = time.time() + ALERT_DURATION
        self._take_screenshot(frame, reason=event_label.replace(" ", "_"))

    def _open_capture(self):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        if isinstance(SOURCE, int) and THREADED_CAPTURE:
            return ThreadedCapture(SOURCE, backend, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOURCC)

        cap = cv2.VideoCapture(SOURCE, backend) if isinstance(SOURCE, int) else cv2.VideoCapture(SOURCE)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {SOURCE}")
        if isinstance(SOURCE, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if CAMERA_FOURCC:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        return cap

    def _run(self):
        cap = None
        try:
            cap = self._open_capture()
            self.status = "running"
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.status = "camera_error"
                    time.sleep(0.1)
                    continue
                self._process_frame(frame)
        except Exception as exc:
            self.status = f"error: {exc}"
        finally:
            if cap is not None:
                cap.release()

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        if self.restricted_rect is None:
            zone_w = int(w * 0.55)
            zone_h = int(h * 0.75)
            x1 = (w - zone_w) // 2
            y1 = (h - zone_h) // 2
            self.restricted_rect = (x1, y1, x1 + zone_w, y1 + zone_h)

        self.frame_num += 1
        self.fps_counter += 1
        now = time.time()
        if now - self.fps_timer >= 1.0:
            self.fps = self.fps_counter / (now - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = now

        if MOTION_GATE:
            fg_mask = self.bg_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        if DETECT_INTERVAL <= 1 or self.frame_num % DETECT_INTERVAL == 0:
            self.last_rects = self.face_detector.detect(frame)

        rects = self.last_rects
        if MOTION_GATE:
            rects = []
            for (x, y, fw, fh) in self.last_rects:
                roi = fg_mask[y:y + fh, x:x + fw]
                motion_ratio = np.count_nonzero(roi) / float(roi.size + 1e-6)
                if motion_ratio >= 0.05:
                    rects.append((x, y, fw, fh))

        if not LIGHT_HUD:
            for (x, y, fw, fh) in rects:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 220, 80), 2)

        objects = self.tracker.update(rects)
        for oid in self.tracker.last_deregistered_ids:
            if oid in self.in_zone_ids:
                self._fire_alert(frame, oid, "Exited Restricted Area", "Track expired after missed detections")
                self.in_zone_ids.remove(oid)

        current_in_zone = set()
        for oid, (cx, cy) in objects.items():
            if self._point_in_rect(cx, cy, self.restricted_rect):
                current_in_zone.add(oid)
                if oid not in self.in_zone_ids:
                    self.ever_entered.add(oid)
                    self.zone_entry_count += 1
                    self._fire_alert(frame, oid, "Entered Restricted Area")
            elif oid in self.in_zone_ids:
                self._fire_alert(frame, oid, "Exited Restricted Area")
        self.in_zone_ids = current_in_zone

        self.stats.update(len(objects))
        if ANALYTICS_INTERVAL > 0 and self.frame_num % ANALYTICS_INTERVAL == 0:
            log_analytics(self.frame_num, len(objects))

        remaining_alert = max(0.0, self.alert_until - time.time())
        draw_dashboard(
            frame, self.tracker, self.stats, self.fps,
            self.restricted_rect, self.in_zone_ids,
            self.alert_msg, remaining_alert,
            lite=LIGHT_HUD,
        )
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return
        with self.lock:
            self.latest_frame = frame.copy()
            self.latest_jpeg = encoded.tobytes()


engine = WebSurveillanceEngine()


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    engine.start()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(
            engine.jpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/status")
    def api_status():
        return jsonify(engine.metrics())

    @app.route("/api/snapshot", methods=["POST"])
    def api_snapshot():
        path = engine.snapshot()
        return jsonify({"ok": path is not None, "path": path})

    return app

"""Threaded camera capture — reads frames in the background to reduce blocking."""

import threading
from collections import deque

import cv2


class ThreadedCapture:
    """Always returns the latest frame; drops older buffered frames."""

    def __init__(self, source, backend=cv2.CAP_ANY, width=0, height=0, fourcc=""):
        self._cap = cv2.VideoCapture(source, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame

    def get_resolution(self):
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()

"""
face_detector.py — YuNet face detection with auto CPU/GPU backend selection
"""

import os
import time
import urllib.request

import cv2
import numpy as np

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "assets", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")


def _ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("[Detector] Downloading YuNet model (one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _make_detector(backend, target):
    return cv2.FaceDetectorYN.create(
        MODEL_PATH, "", (320, 320), 0.7, 0.3, 5000, backend, target
    )


def _bench_detector(target, size, passes=8):
    backend = cv2.dnn.DNN_BACKEND_OPENCV
    det = _make_detector(backend, target)
    w, h = size
    dummy = np.zeros((h, w, 3), np.uint8)
    det.setInputSize((w, h))
    t0 = time.time()
    for _ in range(passes):
        det.detect(dummy)
    return passes / max(time.time() - t0, 1e-6)


def _pick_backend(use_gpu, detect_size):
    cpu_label = "CPU"
    if use_gpu is False:
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, cpu_label

    if use_gpu is True and cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL, "OpenCL (GPU)"

    # Auto — pick whichever backend is faster at the actual inference size
    cpu_fps = _bench_detector(cv2.dnn.DNN_TARGET_CPU, detect_size)
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        gpu_fps = _bench_detector(cv2.dnn.DNN_TARGET_OPENCL, detect_size)
        if gpu_fps > cpu_fps:
            return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL, f"OpenCL (GPU, {gpu_fps:.0f} infer/s)"
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, f"CPU ({cpu_fps:.0f} infer/s)"

    return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, f"CPU ({cpu_fps:.0f} infer/s)"


class GpuFaceDetector:
    """YuNet detector; auto-selects fastest backend and optional downscaled inference."""

    def __init__(
        self,
        use_gpu=None,
        score_threshold=0.7,
        nms_threshold=0.3,
        detect_width=0,
        warmup_size=None,
    ):
        _ensure_model()
        self.detect_width = detect_width
        self._input_size = None
        self._score = score_threshold
        self._nms = nms_threshold

        infer_size = warmup_size or (detect_width or 640, int((detect_width or 640) * 0.75))
        if detect_width:
            infer_size = (detect_width, int(detect_width * 0.75))

        backend, target, label = _pick_backend(use_gpu, infer_size)
        self._detector = cv2.FaceDetectorYN.create(
            MODEL_PATH, "", (320, 320),
            score_threshold, nms_threshold, 5000,
            backend, target,
        )
        h, w = infer_size[1], infer_size[0]
        dummy = np.zeros((h, w, 3), np.uint8)
        self._detector.setInputSize((w, h))
        for _ in range(5):
            self._detector.detect(dummy)
        self._input_size = (w, h)
        print(f"[Detector] YuNet on {label}")

    def detect(self, frame):
        h, w = frame.shape[:2]

        if self.detect_width and w > self.detect_width:
            dw = self.detect_width
            dh = int(h * dw / w)
            detect_frame = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
            input_size = (dw, dh)
            sx, sy = w / dw, h / dh
        else:
            detect_frame = frame
            input_size = (w, h)
            sx = sy = 1.0

        if self._input_size != input_size:
            self._detector.setInputSize(input_size)
            self._input_size = input_size

        _, faces = self._detector.detect(detect_frame)
        if faces is None:
            return []

        return [
            (int(x * sx), int(y * sy), int(fw * sx), int(fh * sy))
            for x, y, fw, fh, *_ in faces
        ]

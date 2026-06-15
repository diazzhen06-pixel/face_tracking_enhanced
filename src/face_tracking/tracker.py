"""
tracker.py — Stable Centroid Tracker with persistent IDs
Handles: disappear/reappear lifecycle, deregistration timeout, trajectory history
"""

from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80, history_len=40):
        """
        max_disappeared  : frames before a face is deregistered
        max_distance     : max pixel distance to match a centroid to an existing ID
        history_len      : number of past centroids kept per face (for trajectory)
        """
        self.next_id         = 0
        self.objects         = OrderedDict()   # id -> (cx, cy)
        self.disappeared     = OrderedDict()   # id -> frame count
        self.history         = OrderedDict()   # id -> [(cx,cy), ...]
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance
        self.history_len     = history_len
        self.last_registered_ids   = []
        self.last_deregistered_ids = []
        self.last_matched_ids      = []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _point(self, centroid):
        return (int(centroid[0]), int(centroid[1]))

    def _register(self, centroid):
        centroid = self._point(centroid)
        oid = self.next_id
        self.objects[oid]    = centroid
        self.disappeared[oid] = 0
        self.history[oid]    = [centroid]
        self.last_registered_ids.append(oid)
        self.next_id += 1

    def _deregister(self, oid):
        self.last_deregistered_ids.append(oid)
        del self.objects[oid]
        del self.disappeared[oid]
        del self.history[oid]

    def _append_history(self, oid, centroid):
        self.history[oid].append(centroid)
        if len(self.history[oid]) > self.history_len:
            self.history[oid].pop(0)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(self, rects):
        """
        rects : list of (x, y, w, h) face bounding boxes
        returns: dict { object_id: (cx, cy) }
        """
        self.last_registered_ids   = []
        self.last_deregistered_ids = []
        self.last_matched_ids      = []

        # --- No detections this frame ---
        if len(rects) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return dict(self.objects)

        # Compute input centroids
        input_centroids = np.array(
            [(x + w // 2, y + h // 2) for x, y, w, h in rects],
            dtype="int"
        )

        # --- No existing objects — register all ---
        if len(self.objects) == 0:
            for c in input_centroids:
                self._register(tuple(c))
            return dict(self.objects)

        # --- Match existing objects to new detections ---
        object_ids       = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()), dtype="int")

        # Distance matrix  (existing x input)
        D = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :],
            axis=2
        )

        # Greedy assignment: smallest distances first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            centroid = self._point(input_centroids[col])
            self.objects[oid]    = centroid
            self.disappeared[oid] = 0
            self.last_matched_ids.append(oid)
            self._append_history(oid, centroid)
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing objects — mark disappeared
        for row in set(range(D.shape[0])) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        # Unmatched input centroids — register as new
        for col in set(range(D.shape[1])) - used_cols:
            self._register(tuple(input_centroids[col]))

        return dict(self.objects)

    def get_history(self, oid):
        return self.history.get(oid, [])

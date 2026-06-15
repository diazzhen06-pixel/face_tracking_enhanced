"""
dashboard.py - Draws the HUD / dashboard overlay on the frame.

The layout keeps the original concept: status panel, FPS meter, timestamp,
restricted zone, person IDs, alert banner, and occupancy badge.
"""

from datetime import datetime

import cv2

from .analytics import occupancy_color, occupancy_status


# ------------------------------------------------------------------ #
# Color palette (BGR for OpenCV)
# ------------------------------------------------------------------ #
C_WHITE   = (244, 246, 248)
C_BLACK   = (0, 0, 0)
C_PANEL   = (18, 22, 28)
C_BORDER  = (78, 88, 102)
C_MUTED   = (175, 184, 194)
C_CYAN    = (255, 230, 0)
C_GREEN   = (0, 220, 80)
C_RED     = (0, 50, 220)
C_ORANGE  = (0, 165, 255)
C_ALERT   = (0, 0, 255)
C_ZONE    = (0, 0, 200)

# ID colour cycling - 8 distinct colours
ID_COLORS = [
    (0, 220, 80),   (255, 180, 0),  (0, 120, 255),  (200, 0, 200),
    (0, 200, 200),  (255, 80, 80),  (80, 255, 80),  (255, 140, 200),
]


def id_color(oid):
    return ID_COLORS[oid % len(ID_COLORS)]


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _clamp(value, low, high):
    if high < low:
        return low
    return max(low, min(high, value))


def _semi_rect(frame, x1, y1, x2, y2, alpha=0.45, color=C_PANEL, border=None):
    """Draw a clipped semi-transparent filled rectangle."""
    h, w = frame.shape[:2]
    x1 = _clamp(int(x1), 0, w - 1)
    x2 = _clamp(int(x2), 0, w - 1)
    y1 = _clamp(int(y1), 0, h - 1)
    y2 = _clamp(int(y2), 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    if border is not None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)


def _fit_scale(txt, max_width, scale, thickness):
    min_scale = 0.30
    while scale > min_scale:
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw <= max_width:
            return scale
        scale -= 0.03
    return min_scale


def _text(frame, txt, x, y, scale=0.55, color=C_WHITE, thickness=1,
          font=cv2.FONT_HERSHEY_SIMPLEX, shadow=True, max_width=None):
    if max_width is not None:
        scale = _fit_scale(txt, max_width, scale, thickness)
    if shadow:
        cv2.putText(frame, txt, (x + 1, y + 1), font, scale, C_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _text_size(txt, scale=0.5, thickness=1):
    return cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]


def _draw_panel_row(frame, label, value, x, y, value_color=C_WHITE):
    _text(frame, label, x, y, 0.34, C_MUTED, 1)
    _text(frame, str(value), x + 92, y, 0.34, value_color, 1)


# ------------------------------------------------------------------ #
# Main draw call
# ------------------------------------------------------------------ #
def draw_dashboard(frame, tracker, stats, fps, restricted_rect,
                   in_zone_ids, alert_msg, alert_timer, lite=False):
    h, w = frame.shape[:2]
    face_count = len(tracker.objects)
    inside_count = len(in_zone_ids)
    occ = occupancy_status(face_count)
    occ_col = occupancy_color(face_count)

    if lite:
        _semi_rect(frame, 8, 6, min(w - 8, 360), 34, 0.70, C_PANEL, C_BORDER)
        _text(
            frame,
            f"FPS {fps:.0f}  |  Faces {face_count}  |  Inside Zone {inside_count}",
            16,
            25,
            0.50,
            C_CYAN,
            1,
            max_width=max(80, min(w - 32, 330)),
        )
        rx1, ry1, rx2, ry2 = restricted_rect
        zone_col = C_ALERT if inside_count else C_ZONE
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), zone_col, 2, cv2.LINE_AA)
        _text(frame, f"INSIDE: {inside_count}", rx1 + 4, ry1 + 18, 0.5, C_ALERT if inside_count else C_MUTED)
        for oid, (cx, cy) in tracker.objects.items():
            col = id_color(oid)
            label = f"ID {oid}"
            lx = _clamp(cx - 20, 8, w - 54)
            ly = _clamp(cy - 10, 48, h - 10)
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if alert_msg and alert_timer > 0:
            _text(frame, alert_msg, 12, h - 14, 0.45, C_ALERT, 1, shadow=True, max_width=w - 24)
        return frame

    # Top-left info panel
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 210, 160
    panel_w = min(panel_w, max(160, w - 20))
    _semi_rect(frame, panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, 0.76, C_PANEL, C_BORDER)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 4, panel_y + panel_h), C_CYAN, -1)

    elapsed = stats.elapsed_seconds()
    mm, ss = int(elapsed) // 60, int(elapsed) % 60

    _text(frame, "CLASS SURVEILLANCE", panel_x + 14, panel_y + 22, 0.42, C_CYAN, 1, max_width=panel_w - 24)
    _draw_panel_row(frame, "Faces", face_count, panel_x + 14, panel_y + 46)
    _draw_panel_row(frame, "Occupancy", occ, panel_x + 14, panel_y + 64, occ_col)
    _draw_panel_row(frame, "FPS", f"{fps:.1f}", panel_x + 14, panel_y + 82)
    _draw_panel_row(frame, "Peak", stats.peak_count, panel_x + 14, panel_y + 100)
    _draw_panel_row(frame, "Alerts", stats.alert_count, panel_x + 14, panel_y + 118, C_ALERT if stats.alert_count else C_WHITE)
    _draw_panel_row(frame, "Inside Zone", inside_count, panel_x + 14, panel_y + 136, C_ALERT if inside_count else C_WHITE)
    _draw_panel_row(frame, "Session", f"{mm:02d}:{ss:02d}", panel_x + 14, panel_y + 154)

    # FPS bar (top-right)
    bar_w, bar_h = 132, 22
    bar_x = max(panel_x + panel_w + 12, w - bar_w - 14)
    bar_y = 12
    if bar_x + bar_w + 14 <= w:
        _semi_rect(frame, bar_x - 6, bar_y - 6, bar_x + bar_w + 6, bar_y + bar_h + 8, 0.72, C_PANEL, C_BORDER)
        fps_ratio = min(max(fps / 30.0, 0.0), 1.0)
        fps_color = C_GREEN if fps >= 20 else (C_ORANGE if fps >= 12 else C_RED)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (36, 42, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * fps_ratio), bar_y + bar_h), fps_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), C_BORDER, 1, cv2.LINE_AA)
        _text(frame, f"FPS {fps:.0f}/30", bar_x + 8, bar_y + 16, 0.42, C_WHITE, 1)

    # Timestamp (top-right below FPS bar)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    ts_w, _ = _text_size(ts, 0.43, 1)
    if w >= 520:
        ts_x = _clamp(w - ts_w - 14, panel_x + panel_w + 10, w - ts_w - 8)
        ts_y = 59
    else:
        ts_x = 14
        ts_y = panel_y + panel_h + 28
    _semi_rect(frame, ts_x - 8, ts_y - 17, ts_x + min(ts_w, w - 28) + 8, ts_y + 7, 0.55, C_PANEL, C_BORDER)
    _text(frame, ts, ts_x, ts_y, 0.43, C_MUTED, 1, max_width=w - ts_x - 18)

    # Restricted area
    rx1, ry1, rx2, ry2 = restricted_rect
    zone_col = C_ALERT if inside_count else C_ZONE
    zone_alpha = 0.20 if inside_count else 0.07
    _semi_rect(frame, rx1, ry1, rx2, ry2, zone_alpha, zone_col)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), zone_col, 2, cv2.LINE_AA)

    zone_label = f"RESTRICTED ZONE  |  INSIDE: {inside_count}"
    zw, zh = _text_size(zone_label, 0.48, 1)
    zl_x = _clamp(rx1 + 8, 8, w - zw - 22)
    zl_y = _clamp(ry1 + 26, 72, h - 10)
    _semi_rect(frame, zl_x - 6, zl_y - zh - 7, zl_x + zw + 8, zl_y + 5, 0.72, C_PANEL, C_BORDER)
    _text(frame, zone_label, zl_x, zl_y, 0.48, C_ALERT if inside_count else C_MUTED, 1, max_width=w - zl_x - 12)

    # Per-face: IDs and optional in-zone marker
    for oid, (cx, cy) in tracker.objects.items():
        col = id_color(oid)
        in_zone = oid in in_zone_ids
        label = f"ID {oid}" + (" [IN ZONE]" if in_zone else "")
        label_color = C_ALERT if in_zone else col

        lw, lh = _text_size(label, 0.50, 1)
        lx = _clamp(cx - lw // 2, 8, w - lw - 10)
        ly = _clamp(cy - 34, 72, h - 10)
        _semi_rect(frame, lx - 6, ly - lh - 6, lx + lw + 6, ly + 6, 0.72, C_PANEL, C_BORDER)
        _text(frame, label, lx, ly, 0.50, label_color, 1)
        cv2.circle(frame, (cx, cy), 4, label_color, -1, cv2.LINE_AA)

        if in_zone:
            cv2.circle(frame, (cx, cy), 12, C_ALERT, 2, cv2.LINE_AA)

    # Alert notification (bottom-left)
    if alert_msg and alert_timer > 0:
        box_w = min(max(320, w // 2), w - 20)
        box_h = 42
        x1 = 10
        y1 = h - box_h - 10
        x2 = x1 + box_w
        y2 = h - 10

        _semi_rect(frame, x1, y1, x2, y2, 0.88, C_PANEL, C_ALERT)
        cv2.rectangle(frame, (x1, y1), (x1 + 4, y2), C_ALERT, -1)
        _text(frame, alert_msg, x1 + 14, y1 + 27, 0.45, C_WHITE, 1, max_width=box_w - 24)

    # Occupancy badge (bottom-right)
    badge_w, badge_h = 126, 70
    badge_x = w - badge_w - 12
    badge_y = h - badge_h - 12
    _semi_rect(frame, badge_x, badge_y, badge_x + badge_w, badge_y + badge_h, 0.72, C_PANEL, C_BORDER)
    _text(frame, "OCCUPANCY", badge_x + 12, badge_y + 22, 0.43, C_MUTED, 1, max_width=badge_w - 24)
    _text(frame, occ, badge_x + 12, badge_y + 54, 0.82, occ_col, 2, max_width=badge_w - 24)

    return frame

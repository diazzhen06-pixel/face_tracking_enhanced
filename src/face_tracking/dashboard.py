"""
dashboard.py - Draws the HUD / dashboard overlay on the frame.

The layout keeps the original concept: status panel, FPS meter, timestamp,
restricted zone, person IDs, alert banner, and occupancy badge.
"""

from datetime import datetime

import cv2
import numpy as np

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

CMD_BG = (17, 13, 9)
CMD_PANEL = (31, 24, 18)
CMD_PANEL_2 = (39, 30, 21)
CMD_PANEL_3 = (24, 18, 13)
CMD_LINE = (63, 50, 38)
CMD_LINE_SOFT = (45, 36, 27)
CMD_TEXT = (247, 243, 237)
CMD_MUTED = (173, 160, 146)
CMD_GREEN = (127, 209, 49)
CMD_AMBER = (72, 179, 231)
CMD_RED = (87, 91, 255)
CMD_BLUE = (255, 167, 100)
CMD_DANGER_BG = (43, 33, 46)
CMD_PRIMARY_BG = (39, 30, 21)


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


def _solid_rect(frame, x1, y1, x2, y2, color, border=None):
    h, w = frame.shape[:2]
    x1 = _clamp(x1, 0, w - 1)
    x2 = _clamp(x2, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    y2 = _clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    if border is not None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)


def _draw_grid(frame, x1, y1, x2, y2, step=32):
    _solid_rect(frame, x1, y1, x2, y2, (5, 7, 7), CMD_LINE_SOFT)
    for x in range(x1, x2, step):
        cv2.line(frame, (x, y1), (x, y2), (13, 24, 19), 1)
    for y in range(y1, y2, step):
        cv2.line(frame, (x1, y), (x2, y), (13, 24, 19), 1)


def _draw_cmd_metric(canvas, x, y, w, h, label, value, accent=None, primary=False, wide=False):
    bg = CMD_DANGER_BG if accent == "danger" else CMD_PRIMARY_BG
    _solid_rect(canvas, x, y, x + w, y + h, bg, CMD_LINE_SOFT)
    _text(canvas, label.upper(), x + 12, y + 23, 0.34, CMD_MUTED, 1, max_width=w - 24)
    value_color = (213, 214, 255) if accent == "danger" else CMD_TEXT
    value_scale = 1.08 if primary else 0.58
    value_thick = 3 if primary else 2
    value_text = str(value)
    value_max_w = w - 24
    if wide:
        scale = _fit_scale(value_text, max(40, value_max_w // 2), value_scale, value_thick)
        (tw, _), _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, scale, value_thick)
        _text(canvas, value_text, x + w - tw - 12, y + h - 16, scale, value_color, value_thick)
    else:
        _text(canvas, value_text, x + 12, y + h - 14, value_scale, value_color, value_thick, max_width=value_max_w)


def _draw_cmd_feed_overlay(feed, tracker, restricted_rect, in_zone_ids, alert_active):
    fh, fw = feed.shape[:2]
    rx1, ry1, rx2, ry2 = restricted_rect
    rx1 = int(rx1 * fw)
    ry1 = int(ry1 * fh)
    rx2 = int(rx2 * fw)
    ry2 = int(ry2 * fh)
    zone_col = CMD_RED if alert_active or in_zone_ids else (60, 85, 210)
    overlay = feed.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), zone_col, -1)
    cv2.addWeighted(overlay, 0.10 if in_zone_ids else 0.05, feed, 0.90 if in_zone_ids else 0.95, 0, feed)
    cv2.rectangle(feed, (rx1, ry1), (rx2, ry2), zone_col, 2, cv2.LINE_AA)
    _text(feed, f"RESTRICTED ZONE  INSIDE: {len(in_zone_ids)}", rx1 + 8, ry1 + 22, 0.46, zone_col, 1)

    for oid, (cx, cy) in tracker.objects.items():
        px = int(cx * fw)
        py = int(cy * fh)
        col = CMD_RED if oid in in_zone_ids else id_color(oid)
        cv2.circle(feed, (px, py), 5, col, -1, cv2.LINE_AA)
        cv2.circle(feed, (px, py), 14, col, 2, cv2.LINE_AA)
        label = f"ID {oid}" + ("  IN ZONE" if oid in in_zone_ids else "")
        _text(feed, label, _clamp(px - 28, 8, fw - 120), _clamp(py - 22, 28, fh - 8), 0.44, col, 1)


def _fit_image_into(src, width, height):
    sh, sw = src.shape[:2]
    scale = min(width / max(sw, 1), height / max(sh, 1))
    new_w = max(1, int(sw * scale))
    new_h = max(1, int(sh * scale))
    return cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_desktop_command_view(frame, tracker, stats, fps, restricted_rect,
                              in_zone_ids, alert_msg, alert_timer,
                              zone_entry_count=0, status="RUNNING"):
    """Build a full desktop command-center canvas for the OpenCV window."""
    canvas_w, canvas_h = 1366, 768
    canvas = np.full((canvas_h, canvas_w, 3), CMD_BG, dtype=np.uint8)

    margin = 14
    top_h = 70
    side_w = 290
    gap = 14
    feed_x = margin
    feed_y = top_h + 6
    feed_w = canvas_w - side_w - gap - margin * 2
    feed_h = canvas_h - feed_y - margin
    side_x = feed_x + feed_w + gap
    side_y = feed_y

    _text(canvas, "SMART CAMPUS SECURITY", margin, 24, 0.34, CMD_MUTED, 1)
    _text(canvas, "Command View", margin, 60, 1.06, CMD_TEXT, 2, max_width=420)

    clock = datetime.now().strftime("%I:%M:%S %p")
    _solid_rect(canvas, canvas_w - 236, 25, canvas_w - 142, 56, CMD_PANEL, CMD_LINE)
    _text(canvas, clock, canvas_w - 225, 47, 0.45, CMD_TEXT, 1)
    status_color = CMD_RED if "ERROR" in status.upper() else CMD_GREEN
    _solid_rect(canvas, canvas_w - 132, 25, canvas_w - margin, 56, CMD_PANEL, CMD_LINE)
    cv2.circle(canvas, (canvas_w - 112, 40), 4, status_color, -1, cv2.LINE_AA)
    _text(canvas, status.upper(), canvas_w - 100, 45, 0.35, status_color, 1, max_width=80)

    _solid_rect(canvas, feed_x, feed_y, feed_x + feed_w, feed_y + 50, CMD_PANEL, CMD_LINE)
    _text(canvas, "CAMERA 01", feed_x + 12, feed_y + 21, 0.34, CMD_MUTED, 1)
    _text(canvas, "Main Entrance Monitoring", feed_x + 12, feed_y + 39, 0.44, CMD_TEXT, 1)
    _solid_rect(canvas, feed_x + feed_w - 66, feed_y + 14, feed_x + feed_w - 12, feed_y + 39, (38, 49, 30), CMD_GREEN)
    cv2.circle(canvas, (feed_x + feed_w - 52, feed_y + 26), 4, CMD_GREEN, -1, cv2.LINE_AA)
    _text(canvas, "LIVE", feed_x + feed_w - 42, feed_y + 31, 0.34, CMD_GREEN, 1)

    frame_area_y = feed_y + 50
    frame_area_h = feed_h - 50
    _draw_grid(canvas, feed_x, frame_area_y, feed_x + feed_w, feed_y + feed_h, 32)

    fh, fw = frame.shape[:2]
    rx1, ry1, rx2, ry2 = restricted_rect
    norm_rect = (rx1 / fw, ry1 / fh, rx2 / fw, ry2 / fh)
    feed = frame.copy()
    _draw_cmd_feed_overlay(feed, tracker, norm_rect, in_zone_ids, alert_msg and alert_timer > 0)
    fitted = _fit_image_into(feed, feed_w, frame_area_h)
    ih, iw = fitted.shape[:2]
    paste_x = feed_x + (feed_w - iw) // 2
    paste_y = frame_area_y + (frame_area_h - ih) // 2
    canvas[paste_y:paste_y + ih, paste_x:paste_x + iw] = fitted

    _solid_rect(canvas, side_x, side_y, side_x + side_w, side_y + feed_h, CMD_PANEL, CMD_LINE)
    pad = 12
    metric_w = (side_w - pad * 3) // 2
    y = side_y + pad
    face_count = len(tracker.objects)
    inside_count = len(in_zone_ids)
    occ = occupancy_status(face_count)
    elapsed = int(stats.elapsed_seconds())
    elapsed_text = f"{elapsed // 60}m {elapsed % 60:02d}s" if elapsed >= 60 else f"{elapsed}s"

    _draw_cmd_metric(canvas, side_x + pad, y, metric_w, 88, "Faces", face_count, primary=True)
    _draw_cmd_metric(canvas, side_x + pad * 2 + metric_w, y, metric_w, 88, "Inside Zone", inside_count, "danger", True)
    y += 98
    _draw_cmd_metric(canvas, side_x + pad, y, metric_w, 62, "Occupancy", occ)
    _draw_cmd_metric(canvas, side_x + pad * 2 + metric_w, y, metric_w, 62, "FPS", f"{fps:.0f}")
    y += 72
    _draw_cmd_metric(canvas, side_x + pad, y, metric_w, 62, "Peak", stats.peak_count)
    _draw_cmd_metric(canvas, side_x + pad * 2 + metric_w, y, metric_w, 62, "Alerts", stats.alert_count)
    y += 72
    _draw_cmd_metric(canvas, side_x + pad, y, side_w - pad * 2, 58, "Zone Entries", zone_entry_count, wide=True)
    y += 68
    _draw_cmd_metric(canvas, side_x + pad, y, side_w - pad * 2, 58, "Session Time", elapsed_text, wide=True)
    y += 72

    _solid_rect(canvas, side_x + pad, y, side_x + side_w - pad, y + 86, CMD_PANEL_3, CMD_LINE_SOFT)
    _text(canvas, "LATEST EVENT", side_x + pad + 10, y + 22, 0.34, CMD_MUTED, 1)
    event = alert_msg if alert_msg and alert_timer > 0 else "No active alert"
    cv2.rectangle(canvas, (side_x + pad + 10, y + 38), (side_x + pad + 14, y + 74), CMD_AMBER, -1)
    _solid_rect(canvas, side_x + pad + 15, y + 38, side_x + side_w - pad - 10, y + 74, (26, 26, 22))
    _text(canvas, event, side_x + pad + 28, y + 62, 0.42, CMD_TEXT, 1, max_width=side_w - 58)

    controls_y = side_y + feed_h - 38
    _solid_rect(canvas, side_x + pad, controls_y, side_x + side_w - pad, controls_y + 26, (55, 42, 31), CMD_LINE)
    _text(canvas, "S Save Screenshot    R Reset    Q Quit", side_x + pad + 14, controls_y + 18, 0.35, CMD_TEXT, 1)
    return canvas


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

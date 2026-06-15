# Smart Campus Surveillance - Enhanced Face Tracking

Course: MITI 266 - Computer Vision using OpenCV and NumPy

## Run

Desktop OpenCV window:

```bash
pip install -r requirements.txt
python main.py
```

Browser dashboard:

```bash
pip install -r requirements.txt
python web.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Project Structure

```text
face_tracking_enhanced/
├── main.py                     # Root entry point
├── web.py                      # Browser dashboard entry point
├── requirements.txt
├── README.md
├── src/
│   └── face_tracking/
│       ├── app.py              # Main OpenCV loop
│       ├── web_app.py          # Flask MJPEG streaming app
│       ├── tracker.py          # Persistent centroid tracker
│       ├── face_detector.py    # YuNet detector and backend selection
│       ├── dashboard.py        # HUD / overlay rendering
│       ├── analytics.py        # Occupancy stats and CSV report
│       ├── logger.py           # Restricted-zone event logger
│       └── camera.py           # Optional threaded capture
│       ├── templates/          # Browser dashboard HTML
│       └── static/             # Browser dashboard CSS
├── assets/
│   └── models/                 # YuNet ONNX model
└── data/
    ├── analytics_report.csv    # Generated per-frame/session stats
    ├── event_log.csv           # Generated zone events
    └── screenshots/            # Generated alert/manual screenshots
```

## Runtime Controls

| Key | Action |
| --- | --- |
| ESC / Q | Quit |
| S | Manual screenshot |
| R | Reset session stats and tracker |

## Configuration

Edit the configuration block near the top of `src/face_tracking/app.py`.

| Variable | Purpose |
| --- | --- |
| `SOURCE` | `0` for webcam, `1` for external camera, or a video file path |
| `MOTION_GATE` | Only accept face detections overlapping MOG2 motion regions |
| `MAX_DISAPPEARED` | Frames before an ID is retired |
| `MAX_DISTANCE` | Max pixel jump to keep the same ID |
| `HISTORY_LEN` | Trajectory trail length |
| `ANALYTICS_INTERVAL` | Log analytics every N frames |
| `ALERT_DURATION` | Alert banner duration in seconds |

## Notes

- The root `main.py` stays as the launcher, while application code lives in `src/face_tracking/`.
- `web.py` runs the same OpenCV detection/tracking pipeline and streams processed frames to the browser.
- Runtime output is isolated in `data/` so source files stay clean.
- The original surveillance HUD concept is preserved; only imports and output paths changed for the cleaner folder layout.

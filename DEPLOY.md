# Deploy Online

This Flask dashboard can run on a cloud host, but the cloud server needs a video source it can access.

## Important

`VIDEO_SOURCE=0` means "use the webcam attached to the machine running the app." On a cloud host, that is not your laptop camera.

For a real online deployment, use one of these:

- An IP camera or RTSP URL, for example `rtsp://user:password@camera-ip:554/stream`
- A video file included with the deployed project
- A cloud-accessible HTTP video stream

## Environment Variables

Set these on your hosting provider:

```text
VIDEO_SOURCE=rtsp://user:password@camera-ip:554/stream
PORT=8000
```

Optional:

```text
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
FACE_SCORE=0.7
DETECT_WIDTH=0
DETECT_INTERVAL=1
LIGHT_HUD=false
MOTION_GATE=false
```

## Render

1. Push this project to GitHub.
2. Open Render Dashboard.
3. Click New, then Blueprint.
4. Connect the GitHub repository.
5. Render will read `render.yaml`.
6. When Render asks for `VIDEO_SOURCE`, enter your RTSP/IP camera URL or another cloud-accessible video source.
7. Deploy the service.

If you create a Web Service manually instead of using the Blueprint, use these settings:

```text
Runtime: Python 3
Build Command: pip install -r requirements-web.txt
Start Command: gunicorn web:app --bind 0.0.0.0:$PORT --threads 4 --timeout 120
```

Set this environment variable:

```text
VIDEO_SOURCE=rtsp://user:password@camera-ip:554/stream
```

Render will provide your public URL after the deploy finishes.

## Cloud Install Command

```bash
pip install -r requirements-web.txt
```

## Cloud Start Command

```bash
gunicorn web:app --bind 0.0.0.0:$PORT --threads 4 --timeout 120
```

Platforms that support `Procfile` can use the included `Procfile` automatically.

## Local Preview

```bash
.\.venv\Scripts\python.exe web.py
```

Then open:

```text
http://127.0.0.1:8000
```

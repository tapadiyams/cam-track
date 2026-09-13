# sample_data/

Drop sample video files here (e.g. `warehouse_loading_dock.mp4`) to use as
a camera `source` in `configs/cameras.yaml` for a demo that does not
require a real RTSP camera. Any file OpenCV's `VideoCapture` can open
works: `RtspFrameReader` (src/ingestion/rtsp_reader.py) treats a file path
exactly like a stream URL.

For a meaningful demo (one that actually produces detections and tracks),
use a short clip -- 30 seconds to a couple of minutes, 720p or lower is
plenty -- that has people or vehicles in it: a phone recording, a screen
recording, or a free stock clip (e.g. Pexels or Pixabay video, no
attribution required). `.gitignore` already excludes `*.mp4`, `*.avi`,
`*.mov`, `*.mkv`, and `*.webm` under this folder, so anything you drop
here for local testing stays local and never gets pushed to the (public)
repo.

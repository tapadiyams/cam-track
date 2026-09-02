# sample_data/

Drop sample video files here (e.g. `warehouse_loading_dock.mp4`) to use as
a camera `source` in `configs/cameras.yaml` for a demo that does not
require a real RTSP camera. Any file OpenCV's `VideoCapture` can open
works: `RtspFrameReader` (src/ingestion/rtsp_reader.py) treats a file path
exactly like a stream URL.

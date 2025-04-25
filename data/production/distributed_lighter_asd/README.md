# Lighter-ASD

ASD (Active Speaker Detection) determines if and when each visible person in the video is speaking.

Lighter-ASD is an optimized implementation of https://github.com/Junhua-Liao/Light-ASD. The original repository is very slow during video/face relevant processing, resulting in a total cost of ~5min for a 5min video. 

We optimized the following:
- Video Loading: ffmpeg -> decord
- Face Detection: MTCNN -> SCRFD
- Face Tracking: N x for-loop IoU-based -> torchvision batchify IoU-based
- Face Tube Cropping: ffmpeg -> torch on-the-fly cropping
- ASD: N x for-loop -> torch batchify
- Introduce window clip sampling for a video: window=150 frames, interval=750frames
- Distribute on multiple GPUs, multiple nodes

With these optimization, we only need 1-1.5s for processing a 5min video, achieving 200-300x speed.

Please refer to (TODO) for more details.

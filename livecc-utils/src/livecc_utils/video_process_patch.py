# NOTE: Some parts were borrowed from qwen_vl_utils. We modified them for better use in LiveCC.
# Feel free to contact joyachen@u.nus.edu for any problems. Thank you!

import os, torch
import numpy as np
import decord # NOTE: import decord should be after torch, otherwise seg fault
from transformers import logging
from torchvision import transforms

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord+'
os.environ['VIDEO_MAX_PIXELS'] = str(int(os.environ.get('VIDEO_MAX_PIXELS', 24576 * 28 * 28))) # increase this for streaming. 24576 * 28 * 28 = 19267584
import qwen_vl_utils.vision_process
qwen_vl_utils.vision_process.VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 100 * 28 * 28)) # follow qwen2vl paper
qwen_vl_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 480)) # decrease this for efficiency 
from qwen_vl_utils.vision_process import (
    FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
    smart_nframes, smart_resize
)

logger = logging.get_logger(__name__)

logger.warning(f'{__name__}: {FORCE_QWENVL_VIDEO_READER=}, {FPS_MAX_FRAMES=}, {VIDEO_MIN_PIXELS=}, {VIDEO_TOTAL_PIXELS=}')

def _read_video_decord_plus(ele: dict, strict_fps: bool = False, drop_last: bool = True, return_pts: bool = False):
    """read video using decord.VideoReader. can handle more cases compared to _read_video_decord.

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
        sample_fps
        clip_pts if return_pts=True
    """
    video_path = ele["video"]
    if os.path.exists(video_path):
        vr = decord.VideoReader(video_path, num_threads=2)
    elif ele['remote_loader'] is not None:
        vr = decord.VideoReader(ele['remote_loader'](video_path), num_threads=2)
    else:
        raise ValueError(f'video_path {video_path} not found')
    video_start = ele.get('video_start', None)
    video_end = ele.get('video_end', None)
    video_fps = vr.get_avg_fps()
    clip_idxs, clip_pts = None, None
    if video_start is not None or video_end is not None:
        vr.get_frame_timestamp(0)
        video_pts = vr._frame_pts[:,1]
        video_start = video_pts[0] if not video_start else video_start
        video_end = video_pts[-1] if not video_end else video_end
        clip_idxs = ((video_start <= video_pts) & (video_pts <= video_end)).nonzero()[0]
        clip_pts = video_pts[clip_idxs]
        total_frames = len(clip_idxs)
    else:
        total_frames = len(vr)
    if not strict_fps:
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
        clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]
    else:
        if clip_pts is None: # no video_start/video_end
            vr.get_frame_timestamp(0)
            clip_pts = vr._frame_pts[:,1]
            clip_idxs = np.arange(len(clip_pts))
        expected_timestamps = np.arange(clip_pts[0], clip_pts[-1] + 1e-6, 1 / FPS)
        if len(expected_timestamps) > FPS_MAX_FRAMES:
            if drop_last:
                expected_timestamps = expected_timestamps[:FPS_MAX_FRAMES]
            else:
                expected_timestamps = expected_timestamps[np.linspace(0, len(expected_timestamps) - 1, FPS_MAX_FRAMES).round().astype(int)]
        expected_idxs_for_clip_pts = (expected_timestamps[:, None] <= clip_pts).argmax(axis=1)
        clip_pts, clip_idxs = clip_pts[expected_idxs_for_clip_pts].tolist(), clip_idxs[expected_idxs_for_clip_pts].tolist()
        while len(clip_idxs) % FRAME_FACTOR != 0:
            clip_idxs.append(clip_idxs[-1])
            clip_pts.append(clip_pts[-1])
    clip = torch.from_numpy(vr.get_batch(clip_idxs).asnumpy()).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = len(clip_idxs) / max(total_frames, 1e-6) * video_fps
    if return_pts:
        return clip, sample_fps, clip_pts
    return clip, sample_fps

from qwen_vl_utils.vision_process import VIDEO_READER_BACKENDS
_video_reader_backend = VIDEO_READER_BACKENDS['decord+'] = _read_video_decord_plus

def _spatial_resize_video(video: torch.Tensor, nframes: int = None):
    if not nframes:
        nframes, _, height, width = video.shape
    else:
        height, width = video.shape[2:]
    max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=max_pixels,
    )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    ).float() # need float?
    return video

def get_smart_resized_video_reader(video_path: str, max_pixels: int = None):
    video_reader = decord.VideoReader(video_path)
    nframes = min(len(video_reader), FPS_MAX_FRAMES)
    height, width, _ = video_reader.next().shape

    if max_pixels is None:
        max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=max_pixels,
    )
    video_reader = decord.VideoReader(video_path, num_threads=2)
    return video_reader, resized_height, resized_width

def get_smart_resized_clip(
    video_reader: decord.VideoReader, 
    resized_height: int,
    resized_width: int,
    timestamps: torch.Tensor, 
    video_pts: np.ndarray, 
    video_pts_index_from: int = 0, 
):
    while len(timestamps) % FRAME_FACTOR != 0:
        timestamps = torch.cat([timestamps, timestamps[-1:] + 1 / FPS])
    clip_idxs = []
    for timestamp in timestamps:
        while video_pts_index_from < len(video_pts) and video_pts[video_pts_index_from] < timestamp:
            video_pts_index_from += 1
        if video_pts_index_from >= len(video_pts):
            break
        clip_idxs.append(video_pts_index_from)
    while len(clip_idxs) % FRAME_FACTOR != 0:
        clip_idxs = clip_idxs[:-1]
        timestamps = timestamps[:-1]
    clip = torch.from_numpy(video_reader.get_batch(clip_idxs).asnumpy()).permute(0, 3, 1 ,2)
    clip = transforms.functional.resize(
        clip,
        [resized_height, resized_width],
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    return clip, timestamps, clip_idxs
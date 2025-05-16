import logging, torch, torchaudio, cv2, io, decord
import numpy as np
from torchvision.transforms.functional import resized_crop
from torchvision.io import write_video

logger = logging.getLogger(__name__)

def resized_crop_faces(gray_frame_windows: torch.tensor, track_windows: list, size: tuple=(112, 112)):
    return [
        [
            torch.cat([resized_crop(gray_frame_windows[window_idx, idx][None], top=box[1], left=box[0], width=box[2]-box[0], height=box[3]-box[1], size=size) for idx, box in track]) for track in tracks
        ] for window_idx, tracks in enumerate(track_windows)
    ]

def visualize_track_windows(frame_windows: torch.tensor, track_windows, score_windows, output_video_path: str, fps: int = 25):
    frame_windows = np.ascontiguousarray(frame_windows.permute(0,1,3,4,2).cpu().numpy()) # w, t, c, h, w
    for frames, tracks, scores in zip(frame_windows, track_windows, score_windows):
        for track, score in zip(tracks, scores):
            for (frame_idx, box), s in zip(track, score):
                x1, y1, x2, y2 = box.tolist()
                color = (255, 0, 0) if s > 0 else (0, 255, 0)
                frames[frame_idx] = cv2.rectangle(frames[frame_idx], (x1, y1), (x2, y2), color, 2)
                if s > 0:
                    prob = s.sigmoid().item() * 100
                    frames[frame_idx] = cv2.putText(frames[frame_idx], f'ASD: {prob:.1f}%', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    window_frames = torch.from_numpy(frame_windows).flatten(0, 1).to(torch.uint8)
    write_video(output_video_path, window_frames, fps)

def visualize_boxes(frames: list[np.ndarray], boxes: list[list[tuple]], output_video_path: str, fps: int = 25):
    if isinstance(frames, tuple):
        frames = list(frames)
    for frame_idx, box in enumerate(boxes):
        if box is not None:
            for x1, y1, x2, y2 in box.tolist():
                frames[frame_idx] = cv2.rectangle(frames[frame_idx], (x1, y1), (x2, y2), (255, 0, 0), 2)
    frames = torch.from_numpy(np.stack(frames)).to(torch.uint8)
    write_video(output_video_path, frames, fps)

def get_audio(video_path: str, start: float = None, end: float = None, sample_rate: int = 16000):
    audio = decord.AudioReader(video_path, sample_rate=sample_rate)._array.T
    if start is not None or end is not None:
        start_idx = int(start * sample_rate) if start is not None else 0
        end_idx = int(end * sample_rate) if end is not None else audio.shape[-1]
        audio = audio[start_idx:end_idx+1]
    audio = audio / np.max(np.abs(audio))  
    audio = (audio * 32767).astype(np.int16)
    return audio

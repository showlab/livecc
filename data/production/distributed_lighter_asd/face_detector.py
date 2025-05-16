import cv2, decord, io, insightface, torch, tqdm
from functools import partial
from torchvision.ops import nms
from torchvision.transforms.functional import normalize
import numpy as np
from utils.multiprocessor import local_mt

class FaceDetector:
    def __init__(self, width: int = 288, height: int = 160, device_id: int = 0, num_workers: int = 4):
        self.detector = insightface.app.FaceAnalysis(name='buffalo_sc', providers=['CUDAExecutionProvider'], provider_options=[{"device_id": device_id}], allowed_modules=['detection']).det_model
        self.detector.prepare(ctx_id=device_id, input_size=(width, height))
        self.detector.use_kps = False
        self.num_workers = num_workers

        self.fpn_flatten_anchor_centers = []
        for stride in self.detector._feat_stride_fpn:
            anchor_centers = np.stack(np.mgrid[:height // stride, :width // stride][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
            anchor_centers = np.stack([anchor_centers]*self.detector._num_anchors, axis=1).reshape( (-1,2) )
            self.fpn_flatten_anchor_centers.append(torch.from_numpy(anchor_centers))
        self.fpn_flatten_anchor_centers = torch.cat(self.fpn_flatten_anchor_centers)
        self.width = width
        self.height = height
    
    def __call__(self, video_path: str, start: float = None, end: float = None, num_audio_to_visual_frames: int = None, window_video_clips: torch.tensor = None): 
        reader = decord.VideoReader(video_path, width=self.width, height=self.height, num_threads=2)
        if start or end:
            reader.get_frame_timestamp(0)
            frame_pts = reader._frame_pts
            start = frame_pts[0, 0] if not start else start
            end = frame_pts[-1, 1] if not end else end
            frame_idxs = ((start <= frame_pts[:,0]) & (frame_pts[:,1] <= end)).nonzero()[0]
        else:
            frame_idxs = np.arange(len(reader))
        num_video_frames = len(frame_idxs)
        if num_audio_to_visual_frames is not None:
            idxs = np.linspace(0, num_video_frames - 1, num_audio_to_visual_frames).round().clip(max=num_video_frames-1).astype(int)
            if window_video_clips is not None:
                idxs = idxs[window_video_clips].flatten()
            frame_idxs = frame_idxs[idxs]
        decord.bridge.set_bridge('torch')
        frames = reader.get_batch(frame_idxs).permute(0, 3, 1, 2)
        boxes = self.batch_detect(frames)
        if window_video_clips is not None:
            frames = frames.reshape(*window_video_clips.shape, *frames.shape[1:])
            boxes = [boxes[i:i+frames.shape[1]] for i in range(0, len(frame_idxs), frames.shape[1])]
        return boxes, frames
    
    def detect(self, idx, frames):
        net_outs = self.detector.session.run(self.detector.output_names, {self.detector.input_name: frames[idx][None]})
        scores, distances = zip(*[(net_outs[idx], net_outs[idx+self.detector.fmc] * stride) for idx, stride in enumerate(self.detector._feat_stride_fpn)])
        scores = torch.from_numpy(np.vstack(scores)).flatten()
        distances = torch.from_numpy(np.vstack(distances))
        boxes = torch.cat([self.fpn_flatten_anchor_centers - distances[:, :2], self.fpn_flatten_anchor_centers + distances[:, 2:]], dim=1)
        keep = scores >= self.detector.det_thresh
        scores, boxes = scores[keep], boxes[keep]
        boxes = boxes[nms(boxes, scores, iou_threshold=self.detector.nms_thresh)].int()
        if not boxes.numel():
            boxes = None
        return boxes
    
    def batch_detect(self, frames):
        # frames = self.normalize(frames)
        frames = normalize(frames.to(torch.float), mean=[127.5, 127.5, 127.5], std=[128, 128, 128]).numpy()
        idxs = range(len(frames))
        return local_mt(idxs, partial(self.detect, frames=frames), num_workers=self.num_workers)

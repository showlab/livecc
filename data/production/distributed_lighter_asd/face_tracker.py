import logging, tqdm, torch
from torchvision.ops import box_iou

logger = logging.getLogger(__name__)

class FaceTracker:
    def __init__(self, iou4track: float = 0.7, min_num_tracks: int = 25):
        self.iou4track = iou4track
        self.min_num_tracks = min_num_tracks
    
    def __call__(self, batch_boxes: list[torch.tensor]):
        tracks, tracking_idxs = [], []
        for frame_idx, boxes in enumerate(batch_boxes):
            if boxes is None:
                tracking_idxs = []
                continue
            if not tracking_idxs:
                for box in boxes:
                    tracks.append([[frame_idx, box]])
                    tracking_idxs.append(len(tracks) - 1)
            else:
                tracking_boxes = torch.stack([tracks[idx][-1][1] for idx in tracking_idxs]) 
                ious = box_iou(tracking_boxes, boxes)
                next_tracking_idx_to_boxes_idx = {}
                for boxes_idx, tracking_boxes_idx in enumerate(ious.argmax(axis=0)):
                    tracking_idx = tracking_idxs[tracking_boxes_idx]
                    if ious[tracking_boxes_idx, boxes_idx] > self.iou4track:
                        if tracking_idx in next_tracking_idx_to_boxes_idx: 
                            matched_boxes_idx = next_tracking_idx_to_boxes_idx[tracking_idx]
                            if ious[tracking_boxes_idx, boxes_idx] > ious[tracking_boxes_idx, matched_boxes_idx]:
                                tracks[tracking_idx][-1] = [frame_idx, boxes[boxes_idx]]
                                next_tracking_idx_to_boxes_idx[tracking_idx] = boxes_idx
                        else:
                            tracks[tracking_idx].append([frame_idx, boxes[boxes_idx]]) 
                            next_tracking_idx_to_boxes_idx[tracking_idx] = boxes_idx
                    else: 
                        tracks.append([[frame_idx, boxes[boxes_idx]]])
                        next_tracking_idx_to_boxes_idx[len(tracks) - 1] = boxes_idx
                tracking_idxs = list(next_tracking_idx_to_boxes_idx.keys())
        return [track for track in tracks if len(track) >= self.min_num_tracks]

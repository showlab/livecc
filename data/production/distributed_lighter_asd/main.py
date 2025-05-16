import json, argparse, logging, python_speech_features, torch, functools, os, itertools, tqdm
from torch.nn.utils.rnn import pad_sequence

from face_tracker import FaceTracker
from face_detector import FaceDetector
from lightasd import LightASD

from multiprocessor import local_mt
from asd_utils import resized_crop_faces, visualize_track_windows, get_audio

logger = logging.getLogger(__name__)

class DistributedASD:
    def __init__(self, device_ids: list[int] = list(range(8))):
        self.distributed_asds = {}
        for device_id in device_ids:
            face_detector = FaceDetector(device_id=device_id, num_workers=2)
            active_speaker_detector = LightASD()
            active_speaker_detector.load_state_dict(torch.load('data/production/distributed_asd/finetuning_TalkSet.model', weights_only=True), strict=False)
            active_speaker_detector.eval()
            active_speaker_detector.to(f'cuda:{device_id}')
            to_gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=f'cuda:{device_id}')
            self.distributed_asds[device_id] = [face_detector,  active_speaker_detector, to_gray_weights]
        self.face_tracker = FaceTracker(iou4track=0.7, min_num_tracks=5)

    def __call__(self, video_path: str, start: float, end: float, device_id: int):
        face_detector, active_speaker_detector, to_gray_weights = self.distributed_asds[device_id] 
        
        mfcc = python_speech_features.mfcc(get_audio(video_path, start=start, end=end))
        mfcc = torch.from_numpy(mfcc).to(device=f'cuda:{device_id}', dtype=torch.float)
        num_audio_to_visual_frames = len(mfcc) // 4
        mfcc = mfcc[:num_audio_to_visual_frames * 4]

        # sample 150 frames clips (6s for fps=25, 5s for fps=30), gap with 750 frames (30s for fps=25, 25s for fps=30)
        window, interval = 150, 750
        window_video_clips = (torch.arange(window - 1, num_audio_to_visual_frames, interval)[:, None] - torch.arange(window)).flip(dims=[1])
        window_audio_clips = (torch.arange(window * 4 - 1, num_audio_to_visual_frames * 4, interval * 4)[:, None] - torch.arange(window * 4)).flip(dims=[1])
        mfcc_windows = mfcc[window_audio_clips]
        mfcc_windows = mfcc_windows.to(device=f'cuda:{device_id}', dtype=torch.float32)

        # face detection
        box_windows, frame_windows = face_detector(video_path, start, end, num_audio_to_visual_frames=num_audio_to_visual_frames, window_video_clips=window_video_clips)
        frame_windows = frame_windows.to(device=to_gray_weights.device, dtype=to_gray_weights.dtype)
        gray_frame_windows = torch.tensordot(frame_windows, to_gray_weights, dims=([2], [0]))

        # face tracking
        track_windows = [self.face_tracker(boxes) for boxes in box_windows]
        
        # face crop
        gray_face_tube_windows = resized_crop_faces(gray_frame_windows, track_windows)

        # asd. batchify all
        gray_face_tube_batch = sum(gray_face_tube_windows, [])
        if not gray_face_tube_batch:
            return frame_windows, [], []
        padded_gray_face_tube_batch = pad_sequence(gray_face_tube_batch, batch_first=True, padding_value=0)
        mfcc_batch = [mfccs[track[0][0]*4:(track[-1][0]+1)*4] for mfccs, tracks in zip(mfcc_windows, track_windows) for track in tracks]
        padded_mfcc_batch = pad_sequence(mfcc_batch, batch_first=True, padding_value=0)
        padded_mfcc_batch = torch.nn.functional.pad(padded_mfcc_batch, pad=(0,0,0,padded_gray_face_tube_batch.shape[1]*4-padded_mfcc_batch.shape[1],0,0), mode='constant', value=0)
        padded_score_batch = active_speaker_detector(padded_gray_face_tube_batch, padded_mfcc_batch)
        score_batch = [score[:gray_face_tube.shape[0]] for score, gray_face_tube in zip(padded_score_batch, gray_face_tube_batch)]
        score_batch_iter = iter(score_batch)
        score_windows = [list(itertools.islice(score_batch_iter, len(gray_face_tubes))) for gray_face_tubes in gray_face_tube_windows]
        # for visualize
        # visualize_track_windows(frame_windows, track_windows, score_windows, f'{device_id}.mp4')
        return frame_windows, track_windows, score_windows

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=str, default='live_whisperx_7c_30-240s_lmloss1.5-5_1.54m.jsonl')
    parser.add_argument('--outputs', type=str, default='datasets/live_whisperx_7c_30-240s_lmloss1.5-5_1.54m_idx2asd/')
    parser.add_argument('--local_from', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=8)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    local = int(os.getenv('ARNOLD_ID')) + args.local_from
    lines = open(args.inputs).readlines()
    local_idxs = list(range(local, len(lines), args.num_nodes))
    
    distributed_asd = DistributedASD(device_ids=list(range(args.num_gpus)))

    def calculate_asd_ratio(device_id: int):
        local_gpu_idxs = local_idxs[device_id::args.num_gpus]
        chunk_size = 1000
        for chunk_idx in range(0, len(local_gpu_idxs), chunk_size):
            local_gpu_chunk_idxs = local_gpu_idxs[chunk_idx:chunk_idx+chunk_size]
            local_gpu_chunk_asd_ratios = []
            path = f'local{local}_gpu{device_id}_chunk{chunk_idx}-{chunk_idx+len(local_gpu_chunk_idxs)}.json'
            for idx in tqdm.tqdm(local_gpu_chunk_idxs, desc=path):
                line = lines[idx]
                datum = json.loads(line)
                video_path = datum['video']
                start, end = datum['content'][0][1], datum['content'][-1][1]
                try:
                    frame_windows, track_windows, score_windows = distributed_asd(video_path, start, end, device_id)
                    total_frames = sum(len(f) for f in frame_windows)
                    asd_frames = 0
                    for tracks, scores in zip(track_windows, score_windows):
                        asd_frame_idxs = set()
                        for track, score in zip(tracks, scores):
                            for (frame_idx, box), s in zip(track, score):
                                if s > 0:
                                    asd_frame_idxs.add(frame_idx)
                        asd_frames += len(asd_frame_idxs)
                    ratio = round(asd_frames / total_frames, 2)
                except Exception as e:
                    print(f'Error when processing {video_path=}, {start=}, {end=}: {e}. Set ratio=-1')
                    ratio = -1
                local_gpu_chunk_asd_ratios.append(ratio)
            local_gpu_chunk_idx_ratios = list(zip(local_gpu_chunk_idxs, local_gpu_chunk_asd_ratios))
            
            with open(path, 'w') as f:
                json.dump(local_gpu_chunk_idx_ratios, f)
            hput('./' + path, args.outputs)

    local_mt(range(args.num_gpus), calculate_asd_ratio, desc='calculate_asd_ratio', num_workers=args.num_gpus)
    

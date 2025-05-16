import os, argparse, tqdm, json, torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from decord import AudioReader

from data.tos import get_tos_client, get_tos_bytes
from utils.multiprocessor import local_mt

class WhisperX4Video:
    def __init__(self, batch_size: int = 16, device_id: int = 0):
        self.batch_size = batch_size
        model = WhisperModel("large-v3-turbo", device="cuda", device_index=device_id, compute_type="float16")
        self.sample_rate = model.feature_extractor.sampling_rate
        self.model = BatchedInferencePipeline(model=model)
        self.tos_client = get_tos_client()
    
    def load_audio(self, video: str):
        if not os.path.exists(video):
            video = get_tos_bytes(video, self.tos_client, md5_check=True) # replace with your method to get video bytes or path str
        audio = AudioReader(video, sample_rate=self.sample_rate)._array.T
        return audio.mean(axis=1)
        
    @torch.inference_mode
    def __call__(self, video: str):
        audio = self.load_audio(video)
        segments, _ = self.model.transcribe(audio, word_timestamps=True, batch_size=self.batch_size)
        words = [[word.start, word.end, word.word.strip()] for segment in segments for word in segment.words]
        return words

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--metadata_path', type=str, default='live_cc_5m_np_metadatas.json')
    return parser.parse_args()

def line2metadata(line):
    video = line[line.index('"video":')+10:line.index('"video_start":')-3]
    title = line[line.index('title')+9:line.index('assistant')-16]
    return (video, title)

if __name__ == '__main__':
    args = get_args()
    local = int(os.getenv('ARNOLD_ID')) # replace with your local node id
    models = [WhisperX4Video(batch_size=args.batch_size, device_id=worker % 8) for worker in range(args.num_workers)]

    video_metadatas = json.load(open(args.metadata_path))
    local_video_metadatas = video_metadatas[local::args.num_nodes]
    
    def distribute(worker: int):
        model = models[worker]
        local_worker_video_metadatas = local_video_metadatas[worker::args.num_workers]
        video_asrs = []
        chunk_size = 1000
        for i in range(0, len(local_worker_video_metadatas), chunk_size):
            chunked_local_worker_video_metadatas = local_worker_video_metadatas[i:i+chunk_size]
            path = f'node{local}_worker{worker}_chunk{i}-{i+len(chunked_local_worker_video_metadatas)}.jsonl'
            for video, title in tqdm.tqdm(chunked_local_worker_video_metadatas, desc=path):
                try:
                    video_asrs.append({'video': video, 'content': model(video), 'title': title})
                except Exception as e:
                    print(video, e)
            with open(path, 'w') as f:
                for video_asr in video_asrs:
                    f.write(json.dumps(video_asr) + '\n')

    local_mt(range(args.num_workers), distribute, desc='distributed_whisperx', num_workers=args.num_workers)

    

import json, os, argparse, cv2, torchvision, torch
import soundfile as sf
from pydub import AudioSegment
from PIL import Image
import numpy as np
from kokoro import KPipeline
from demo.render.bubble import ResponseBubble, QueryBubble
from moviepy import VideoFileClip, AudioFileClip, ImageSequenceClip

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render video with separate Query and Response bubbles and TTS"
    )
    parser.add_argument(
        '--result_json', type=str, required=True,
        help='JSON file with video_path, commentaries, and query'
    )
    parser.add_argument(
        '--query_start', type=float, required=True,
        help='Start time (sec) for query bubble and TTS'
    )
    parser.add_argument(
        '--response_start', type=float, required=True,
        help='Start time (sec) for response bubbles and TTS'
    )
    parser.add_argument(
        '--sentence_tts',
        action='store_true',
        help='Concatenate audio for the entire sentence. If not set, audio is processed word by word.'
    )
    parser.add_argument(
        '--voice', type=str, default='af_heart',
        help='Start time (sec) for response bubbles and TTS'
    )
    return parser.parse_args()

def generate_segment_audio(pipeline, text, voice, out_path):
    """
    Generate TTS audio for a text segment, save WAV, and return its duration.
    """
    audio_buffer = []
    for _, _, audio in pipeline(text, voice=voice):
        audio_buffer.append(audio)
    audio_array = np.concatenate(audio_buffer)
    sf.write(out_path, audio_array, 24000)
    clip = AudioFileClip(out_path)
    duration = clip.duration
    clip.close()
    return duration

def assemble_full_audio(segments, total_duration, base_name="tts_seg"):
    """
    Overlay TTS segments onto a silent track matching video duration.
    segments: list of (start_time_sec, text).
    """
    full = AudioSegment.silent(duration=int(total_duration * 1000))
    for idx, (start, _) in enumerate(segments):
        seg_path = f"demo/cache/{base_name}_{idx}.wav"
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Missing {seg_path}")
        seg = AudioSegment.from_file(seg_path)
        full = full.overlay(seg, position=int(start * 1000))
    out_path = "demo/cache/combined_audio.wav"
    full.export(out_path, format="wav")
    return out_path

def annotate_frame(frame, t, query_start, response_start,
                   query_text, response_commentary,
                   query_bubble, response_bubble):
    """
    Draw QueryBubble if t in [query_start, response_start], else
    draw Response ChatBubble if within any commentary window.
    """
    offset = 0
    # Query interval
    if query_start <= t <= response_start and query_text:
        pil = Image.fromarray(frame[:, :, ::-1])
        pil = query_bubble.draw_bubble(
            base_img=pil,
            position=(40, 40),
            text=query_text,
            metadata=f"User"
        )
        arr = np.array(pil)
    else:
        # Response intervals
        start, end, text = response_commentary
        if t >= start:
            pil = Image.fromarray(frame[:, :, ::-1])
            pil = response_bubble.draw_bubble(
                base_img=pil,
                position=(40, 40),
                text=text,
                metadata=f"Video Time: {t:.1f}s | Model: LiveCC-7B-Instruct"
            )
            arr = np.array(pil)
        if t > end:
            offset = 1
    return arr, offset

def main():
    args = parse_args()
    result_path = args.result_json
    query_start = args.query_start
    response_start = args.response_start
    sentence_tts = args.sentence_tts
    voice = args.voice

    with open(result_path, 'r') as f:
        result = json.load(f)
    video_path = result['video_path']
    commentaries = result['commentaries']  # list of [start, end, text]
    query_text = result.get('query', '').strip()

    # rebase timestamps
    commentaries = [(s-query_start, e-query_start, t) for s, e, t in commentaries]
    response_start = response_start - query_start
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(query_start * fps)
    # Annotate only from query_start to the end
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    query_start = 0

    # Initialize TTS pipeline and bubbles
    pipeline = KPipeline(lang_code='a')
    query_bubble = QueryBubble()
    response_bubble = ResponseBubble()

    # Prepare TTS segments: query first, then responses after response_start
    audio_commentaries = []
    sentence_start, sentence_text = None, ''
    for i, (s, e, t) in enumerate(commentaries):
        t = t.replace(' ...', '').strip()
        if t:
            if not sentence_tts:
                audio_commentaries.append([s, e, t])
                continue
            sentence_text += ' ' + t
        if sentence_start is None and t:
            sentence_start = s
        if t.endswith(',') or t.endswith('.') or t.endswith('!') or t.endswith('?') or i == len(commentaries) - 1:
            if sentence_text.strip():
                audio_commentaries.append([sentence_start, e, sentence_text.strip()])
            sentence_start = None
            sentence_text = ''
    audio_segments = []
    idx = 0
    for start, end, text in audio_commentaries:
        if end > response_start:
            audio_start = max(start, response_start)
            seg_path = f"demo/cache/tts_seg_{idx}.wav"
            os.makedirs(os.path.dirname(seg_path), exist_ok=True)
            generate_segment_audio(pipeline, text, voice, seg_path)
            # Re-base all audio segments so they align to the trimmed video
            audio_segments.append((audio_start, text))
            idx += 1
    print(f'{audio_segments=}')

    # Assemble full audio track
    combined_audio_path = assemble_full_audio(audio_segments, commentaries[-1][1])

    frames = []
    frame_idx = 0
    commentary_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # compute the relative time in original video
        t = frame_idx / fps
        print(f'render {frame_idx=}, {t=}')
        annotated, commentary_idx_offset = annotate_frame(
            frame, t, query_start, response_start,
            query_text, commentaries[commentary_idx],
            query_bubble, response_bubble
        )
        commentary_idx += commentary_idx_offset
        frames.append(annotated)
        frame_idx += 1
        if commentary_idx >= len(commentaries):
            break
    cap.release()

    # Write silent annotated video
    video_clip = ImageSequenceClip(frames, fps=fps)
    audio_clip = AudioFileClip(combined_audio_path)
    final = video_clip.with_audio(audio_clip)
    save_path = os.path.splitext(result_path)[0] + '_rendered.mp4'
    final.write_videofile(
        save_path, 
        codec='libx264', 
        audio_codec='aac', 
        fps=fps,
        ffmpeg_params=[
            '-crf', '18',           # again, bump up quality
            '-preset', 'slow',      # or 'veryslow' if you have time
            '-pix_fmt', 'yuv420p',  # ensure broad compatibility
        ]
    )

    # Cleanup
    for i in range(idx):
        os.remove(f"demo/cache/tts_seg_{i}.wav")
    os.remove(combined_audio_path)

    video_clip.close()
    audio_clip.close()

    print(f"Finished rendering: {save_path}")

if __name__ == '__main__':
    main()

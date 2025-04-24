import json
from demo.infer import LiveCCDemoInfer

if __name__ == '__main__':
    model_path = 'chenjoya/LiveCC-7B-Instruct'
    video_path = "demo/sources/howto_fix_laptop_mute_1080p.mp4"
    query = """Please describe the video."""
    
    infer = LiveCCDemoInfer(model_path=model_path)
    state = {'video_path': video_path}
    commentaries = []
    t = 0
    for t in range(31):
        state['video_timestamp'] = t
        for (start_t, stop_t), response, state in infer.live_cc(
            message=query, state=state, 
            max_pixels = 384 * 28 * 28, repetition_penalty=1.05, 
            streaming_eos_base_threshold=0.0, streaming_eos_threshold_step=0
        ):
            print(f'{start_t}s-{stop_t}s: {response}')
            commentaries.append([start_t, stop_t, response])
        if state.get('video_end', False):
            break
        t += 1
    result = {'video_path': video_path, 'query': query, 'commentaries': commentaries}
    result_path = video_path.replace('/assets/', '/results/').replace('.mp4', '.json')
    print(f"{video_path=}, {query=} => {model_path=} => {result_path=}")
    json.dump(result, open(result_path, 'w'))
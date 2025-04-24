import os, json, openai, tqdm, time, argparse
from utils.multiprocessor import local_mt

gpt = openai.AzureOpenAI(
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    api_version="gpt-4o-2024-08-06",
    api_key=os.environ.get('AZURE_OPENAI_API_KEY'), 
)


template = """This is the speech transcription from a video clip:

"{asr}"

Analyze this transcription and determine if ALL of the following conditions are met:
1. The speech is describing or commenting on real-time video content
2. The speaker is not sharing personal experiences or feelings
3. The transcription is not from multiple people having a conversation
4. the transcription text has no garbled characters

If ALL conditions are met, respond with "YES" and suggest a generic user query that would prompt an AI to generate commentary in a similar style (without including any specific content from the original transcription).

If ANY condition is NOT met, respond with "NO" and no further explanation.

Format your response as JSON:
{"result": "YES" or "NO", "query": "your suggested query here or empty string if result is NO"}
"""

def get_prompt(datum: dict):
    asr = ' '.join(w.strip() for s, e, w in datum['content'])
    query = template.replace('{asr}', asr)
    # According to the commentary style, give the prompt like "Please comment this video in real time...", with the commentary style requirements if necessary. Be concise and do not leak the ground-truth commentary.
    while True: 
        try:
            response = gpt.chat.completions.create(
                model='gpt-4o-2024-08-06', 
                messages=[{"role": "user", "content": query}],
            ).choices[0].message.content
            return response
        except Exception as e:
            print(e)

if __name__ == '__main__':
    path = 'live_whisperx_7c_30-240s_lmloss1.5-5_asd0-0.05_1.01m.jsonl'
    lines = open(path).readlines()
    datums = local_mt(lines, json.loads, desc='json.loads', num_workers=16)
    chunk_size = 1000
    for chunk_idx in range(0, len(datums), chunk_size):
        chunk_datums = datums[chunk_idx:chunk_idx+chunk_size]
        chunk_prompts = local_mt(chunk_datums, get_prompt, desc='make prompt', num_workers=16)
        chunk_path = f'live_whisperx_7c_30-240s_lmloss1.5-5_asd0-0.05_1.01m_prompted_chunk{chunk_idx}-{chunk_idx+len(chunk_datums)}.jsonl'
        with open(chunk_path, 'w') as f:
            for datum, prompt in tqdm.tqdm(zip(chunk_datums, chunk_prompts), desc=chunk_path):
                try:
                    json_string = prompt[prompt.index('json')+4:prompt.rindex('```')]
                    json_dict = json.loads(json_string)
                    if json_dict['result'] == 'YES':
                        datum['query'] = json_dict['query']
                    f.write(json.dumps(datum) + '\n')
                except Exception as e:
                    print(e)
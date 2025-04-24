import torch, json, tqdm, os
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.multiprocessor import local_mt, local_mp

# Custom Dataset to load data from JSONL using seek positions
class ConversationDataset(Dataset):
    def __init__(self, data_lines: list[str]):
        self.data_lines = data_lines

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        datum = json.loads(self.data_lines[idx])
        title = datum['title']
        subtitles = datum.get('content', [])
        start, end = subtitles[0][0], subtitles[-1][1]
        cc = ' '.join(t.strip() for s, e, t in subtitles)
        duration = sum(e - s for s, e, t in subtitles)
        if 'previous' not in datum:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Video Title: {title}\nPlease try to output the possible speech transcription of the video from start to end."},
                {"role": "assistant", "content": cc},
            ]
        else:
            previous = datum['previous']
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Video Title: {title}\nPrevious transcription: {previous}\nPlease try to output the possible speech transcription of the video following previous transcription."},
                {"role": "assistant", "content": cc},
            ]
        return conversation, datum['video'], duration, start, end

def BatchForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    loss = nn.functional.cross_entropy(logits, shift_labels.view(-1).to(logits.device), ignore_index=ignore_index, reduction='none')
    loss = loss.view_as(shift_labels).sum(dim=-1) / (shift_labels > 0).sum(dim=-1)
    return loss

data_path = 'live_whisperx_30-240s.jsonl'
lines = open(data_path).readlines()
num_lines = len(lines)
num_gpus = 8

def pure_lm_loss(device_id: int):
    device = f"cuda:{device_id}"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct",
        torch_dtype="auto",
        device_map=device,
        attn_implementation='flash_attention_2'
    ) 
    model.loss_function = BatchForCausalLMLoss
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", padding_side='left')
    im_start_id, assistant_id = tokenizer('<|im_start|>assistant').input_ids

    dataset = ConversationDataset(data_lines=lines[device_id::num_gpus])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x, num_workers=1)

    outputs = []
    for batch in tqdm.tqdm(dataloader, desc=device):
        conversations, tos_keys, durations, starts, ends = zip(*batch)
        inputs = tokenizer.apply_chat_template(conversations, return_tensors="pt", padding=True, return_dict=True, truncation=True)
        input_ids = inputs.input_ids
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        for batch_idx, assistant_idx in torch.argwhere(input_ids == assistant_id):
            if input_ids[batch_idx, assistant_idx-1] == im_start_id:
                labels[batch_idx, assistant_idx+2:-2] = input_ids[batch_idx, assistant_idx+2:-2] # do not count eos
        inputs['labels'] = labels
        inputs = inputs.to(device)

        with torch.inference_mode():
            losses = model(**inputs, reduction='none').loss.tolist() 
            outputs.extend(list(zip(tos_keys, durations, losses, starts, ends)))
    output_dir = os.path.splitext(data_path)[0] + f'_lmlosses'
    json.dump(outputs, open(f'{output_dir}/lmlosses_device{device_id}.json', 'w'))
    return outputs

def filter(lower_bound, upper_bound):
    get_video = lambda line: line[11:line.index('title')-4]
    input_dir = os.path.splitext(data_path)[0] + f'_lmlosses'
    lmlosses = [None] * num_lines
    for device_id in range(num_gpus):
        lmlosses[device_id::num_gpus] = json.load(open(f'{input_dir}/lmlosses_device{device_id}.json'))
    filtered_lines = []
    for (video, duration, loss, start, end), line in tqdm.tqdm(zip(lmlosses, lines)):
        assert get_video(line) == video
        if lower_bound <= loss <= upper_bound:
            filtered_lines.append(line)
    save_path = os.path.splitext(data_path)[0] + f'_lmloss{lower_bound}-{upper_bound}.jsonl'
    with open(save_path, 'w') as f:
        f.writelines(filtered_lines)

if __name__ == '__main__':
    # outputs = local_mp(list(range(8)), pure_lm_loss, desc='pure lm loss', num_workers=num_gpus)
    filter(1.5, 5)
    
    
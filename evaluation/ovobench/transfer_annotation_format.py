import json, argparse

class Transfer:
    @staticmethod
    def format_crr(datum: dict):
        question = f"""You're responsible of answering questions based on the video content. The following question are relevant to the latest frames, i.e. the end of the video.\n\n{datum['question']}\n\nDecide whether existing visual content, especially latest frames, i.e frames that near the end of the video, provide enough information for answering the question.\nReturn "Yes" if existing visual content has provided enough information;\nReturn "No" otherwise."""
        options = ["No", "Yes"]
        video_start = datum['ask_time']
        annos = [dict(
            id=datum['id'],
            task=datum['task'],
            question=question,
            # options=options,
            video_start=video_start, 
            video_end=test_info['realtime'], 
            answer=options[test_info['type']],
            video=datum['video'],
        ) for i, test_info in enumerate(datum['test_info'])]
        return annos

    @staticmethod
    def format_rec(datum: dict):
        question = f"""You're watching a video in which people may perform a certaintype of action repetitively. The person performing are referred to as 'they' in the following statement. You're task is to count how many times did different people in the video perform this kind of action in total.\nNow, answer the following question:\n\nHow many times did they {datum['activity']}?\n\nYour response type should be INT, for example, 0/1/2/3.."""
        options = [str(i) for i in range(11)]
        annos = [dict(
            id=datum['id'],
            task=datum['task'],
            question=question,
            # options=options,
            video_start=0,  
            video_end=test_info['realtime'],  
            answer=options[test_info['count']], 
            video=datum['video'],
        ) for i, test_info in enumerate(datum['test_info'])]
        return annos
    
    @staticmethod
    def format_ssr(datum):
        options = ["No", "Yes"]
        annos = [dict(
            id=datum['id'],
            task=datum['task'],
            question=f"""You're watching a tutorial video which contain a sequential of steps. The following is one step from the whole procedures:\n\n{test_info['step']}\n\nYour task is to decide: Is the man/woman in the video currently carrying out this step?\nReturn "Yes" if the man/woman in the video is currently performing this step;\nReturn "No" if not.""",
            # options=options,
            video_start=0, 
            video_end=test_info['realtime'], 
            answer=options[test_info['type']],
            video=datum['video'],
        ) for i, test_info in enumerate(datum['test_info'])]
        return annos

    @staticmethod
    def format_other(datum):
        datum['video_start'] = 0
        datum['video_end'] = datum.pop('realtime')
        choices = ['A', 'B', 'C', 'D', 'E']
        datum['options'] = [f'{choices[i]}. {option}' for i, option in enumerate(datum['options'])]
        datum['answer'] = choices[datum.pop('gt')]
        return datum

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Format OVO-Bench dataset JSONL file.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save formatted JSONL file.")
    args = parser.parse_args()

    annos = []
    for datum in json.load(open(args.input)):
        if hasattr(Transfer, 'format_' + datum['task'].lower()):
            formatter = getattr(Transfer, 'format_' + datum['task'].lower())
            annos.extend(formatter(datum))
        else:
            formatter = Transfer.format_other
            annos.append(formatter(datum))

    with open(args.output, 'w') as f:
        for anno in annos:
            f.write(json.dumps(anno) + '\n')
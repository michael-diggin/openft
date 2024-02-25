import json


def load_from_file(path: str, split: str | None = '\n\n') -> list[str]:
    """
    load_from_file reads data from the given path
    and splits it into a list.
    If split is None, the data read is returned as a list itself.
    """
    try:
        with open(path, 'r') as f:
            data = f.read()
    except FileNotFoundError as ex:
        print(f"Could not find {path}: check if the training_dir is correct and that the path does exist")
        raise ex

    if split:
        data_points = data.split(split)
    else:
        data_points = [data]
    return data_points

def create_single_ft_message(system_prompt: str, question: str, answer: str) -> dict:
    """
    The format the OpenAI expects for fine tuning is
    {'messages': [
        {'role': 'system', 'content': 'system_prompt, text'}
        {'role': 'user', 'content': 'user input, text'}
        {'role': 'assistant', 'content': 'desired AI response to the user input, text'}
        ]
    }

    The list of messages can go on as long as desired, eg to simulate a back and forth conversation.
    For this use case, it's QA at the moment so just one user and one assistant message.
    """
    sys_msg = {'role': 'system', 'content': system_prompt}
    user_msg = {'role': 'user', 'content': question}
    assistant_msg = {'role': 'assistant', 'content': answer}
    msg = {'messages': [sys_msg, user_msg, assistant_msg]}
    return msg

def write_dataset_to_jsonl(dataset: list[dict], path: str):
    '''
    OpenAI files are expected to be JSONL
    Where each line is a valid JSON object
    '''
    with open(path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example))
            f.write('\n')
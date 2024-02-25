import tiktoken


def calc_total_tokens(dataset: list[dict], encoding: tiktoken.Encoding, tokens_per_msg: int = 3) -> int:
    '''
    Given a dataset for fine tuning, this function returns an estimated
    for the total number of tokens needed for the dataset
    It is an estimation based on https://cookbook.openai.com/examples/chat_finetuning_data_prep#token-counting-utilities
    '''
    num_tokens = 0
    for example in dataset:
        num_tokens += _tokens_for_messages(example['messages'], encoding, tokens_per_msg)
    return num_tokens

def _tokens_for_messages(messages: list[dict], encoding: tiktoken.Encoding, tokens_per_msg: int = 3) -> int:
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_msg
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 3
    return num_tokens

def calc_cost_of_training(dataset: list[dict], encoding: tiktoken.Encoding, num_epochs: int = 1, cost_per_1k: float = 0.008) -> float:
    '''
    Cost per 1k tokens can be found https://openai.com/pricing
    This calculates the cost of training for num_epochs.
    This is an estimation.
    '''
    total_tokens = calc_total_tokens(dataset, encoding)
    cost = (total_tokens / 1000.) * num_epochs * cost_per_1k
    return cost


def check_all_examples_are_bounded(dataset: list[dict], encoding: tiktoken.Encoding, bound: int = 4096) -> tuple[bool, int]:
    '''
    https://platform.openai.com/docs/guides/fine-tuning/token-limits
    When fine tuning the maximum example length if 16385 tokens for gpt-3.5-turbo-1106
    and 4096 for gpt-3.5-turbo-0613 (default)
    '''
    all_good = True
    max_tokens = 0
    for i, example in enumerate(dataset):
        num_tokens = _tokens_for_messages(example['messages'], encoding)
        if num_tokens > bound:
            all_good = False
            print(f"Example {i} exceeds the bound of {bound} tokens and will be truncated")
        if num_tokens > max_tokens:
            max_tokens = num_tokens
    return all_good, max_tokens
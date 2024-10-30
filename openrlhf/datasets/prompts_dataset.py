import json

from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none


def preprocess_data(data, input_template=None, input_key="input", ref_key=None, task=None, apply_chat_template=None) -> dict:
    if apply_chat_template:
        if isinstance(data[input_key], list):
            # 适配ultra_interact的trajectory
            if len(data[input_key])>0 and 'role' not in data[input_key][0]:
                prompt=apply_chat_template([{'role':piece['from'], 'content':piece['value']} for piece in data[input_key]], tokenize=False, add_generation_prompt=True)
            else:
                prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
        else:
            prompt = apply_chat_template([{"role": "user", "content": data[input_key]}], tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt
    # if ref_key:
    #     ref = data[ref_key]
    # else:
    #     ref = None
    # return {"prompt": prompt, "reference": ref, "task": task, "completions": None}


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.n_samples_per_prompt = getattr(self.strategy.args, "n_samples_per_prompt", 1)

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        task = getattr(self.strategy.args, "task", None)
        ref_key = getattr(self.strategy.args, "ref_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.original =[]
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt_dict = preprocess_data(data, input_template, input_key, ref_key, task, apply_chat_template) # cheat collate function to pass the whole data entry
            self.prompts.append(prompt_dict)
            self.original.append(json.dumps(data,ensure_ascii=False))


    def __len__(self):
        length = len(self.prompts)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        return {'prompt':self.prompts[idx // self.n_samples_per_prompt],'original':self.original[idx // self.n_samples_per_prompt]}

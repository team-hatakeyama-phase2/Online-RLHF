import json
import os
import requests
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from accelerate import Accelerator

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="iter2_K64.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="iter2_K64_Mreward.json",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of responses per prompt"},
    )
    reward_api_hostname: Optional[str] = field(
        default="slurm0-a3-ghpc-1",
        metadata={"help": "hostname of reward_api"},
    )
    reward_api_port: Optional[int] = field(
        default=8000,
        metadata={"help": "port of reward_api"},
    )


accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device

ds_dir = script_args.dataset_name_or_path
ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")

reward_name_or_path = script_args.reward_name_or_path
reward_api_hostname = script_args.reward_api_hostname
reward_api_port = script_args.reward_api_port


"""
We process the data format here and query the reward model to get the rewards.
"""

# def get_reward(test_texts):
#     pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
#     rewards = [output[0]["score"] for output in pipe_outputs]
#     return rewards

def get_reward_from_api(test_texts):
    url = f"http://{reward_api_hostname}:{reward_api_port}:/classify/{reward_name_or_path.split('/')[-1]}"
    #url = f"http://slurm0-a3-ghpc-1:8000/classify/{reward_name_or_path.split('/')[-1]}"
    #url = f"http://slurm0-a3-ghpc-0:8000/classify/{reward_name_or_path.split('/')[-1]}"
    headers = {"Content-Type": "application/json"}
    input_json = {
        "prompt": test_texts,
    }

    response = requests.post(url, headers=headers, json=input_json).json()
    rewards = [output[0]["score"] for output in response]
    return rewards


# def change_of_format(prom, resp):
#     # To be modified according to the reward model and the LLM you use
#     # Be careful about multi-turn conversions
#     """
#     prom = prom.replace("<s>GPT4 Correct User: ", "").replace("<|end_of_turn|>GPT4 Correct Assistant:", "")

#     final_resp = resp.split("GPT4 Correct User")[0]
#     """
#     prom = prom
#     final_resp = resp

#     message = [
#         {"role": "user", "content": prom},
#         {"role": "assistant", "content": final_resp},
#     ]
#     return rm_tokenizer.apply_chat_template(message, tokenize=False).replace(rm_tokenizer.bos_token, "")


data = []

# tqdm is used to show the progress bar
for sample in tqdm(ds):
    # The VLLM may not generate responses for some prompts because it is too long, we skip them
    if len(sample["responses"]) < script_args.K:
        continue
    # test_texts = [change_of_format(sample['prompt'], tmp_output) for tmp_output in sample['responses']]
    test_texts = [
        sample["prompt"] + script_args.input_output_delimiter + tmp_output.strip()
            for tmp_output in sample["responses"]
    ]

    rewards = get_reward_from_api(test_texts)
    data.append({"prompt": sample["prompt"], "responses": sample["responses"], "rewards": rewards})


all_rewards = [sample["rewards"] for sample in data]
top1_scores = np.mean(np.max(all_rewards, axis=1))
mean_scores = np.mean(all_rewards)


output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = data
with open(script_args.output_dir, "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

if script_args.record_dir is not None:
    with open(script_args.record_dir, "a") as f:
        f.write(str(mean_scores) + "\t" + str(top1_scores) + "\n")

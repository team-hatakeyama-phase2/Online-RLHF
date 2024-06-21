import os
import argparse
from functools import reduce
import numpy as np
import pandas as pd
import datasets


parser = argparse.ArgumentParser()
parser.add_argument("--input_prompt", help="input_prompt")
parser.add_argument("--predict_dir", help="predict_dir")
parser.add_argument("--iteration_prefix", help="iteration_prefix")
args = parser.parse_args()

input_prompt = args.input_prompt
predict_dir = args.predict_dir
iteration_prefix = args.iteration_prefix

# input_prompt = "/content/drive/MyDrive/Geniac/datasets/hh-rlhf-12k-ja_mod"
# predict_dir = "./predicts"
# iteration_prefix = "karasu-1.1B_hh-rlhf-12k-ja_mod_mistral_rm"


use_columns = ['prompt', 'responses', 'rewards', 'avg_rewards']

original_df = datasets.load_from_disk(input_prompt).to_pandas().drop_duplicates()
original_df["prompt"] = original_df["input"]

reward_files = [x for x in os.listdir("./predicts") if x.startswith(iteration_prefix) and x.endswith("_reward.json")] 

dfs = []
for reward in reward_files:
    print(reward)
    df = pd.read_json(f"{predict_dir}/{reward}")
    df['prompt'] = df['instances'].apply(lambda x: x['prompt'])
    df['responses'] = df['instances'].apply(lambda x: x['responses'])
    df['rewards'] = df['instances'].apply(lambda x: x['rewards'])
    df['avg_rewards'] = df['rewards'].apply(lambda x: np.mean(x))
    dfs.append(df[use_columns])

merged_df = original_df

for i, df in enumerate(dfs):
    merged_df = merged_df.merge(df, on="prompt", suffixes=(f'_{i}', f'_{i+1}'))

merged_df['original_index'] = merged_df.index
explode_columns = [x for x in merged_df.columns if x.startswith('responses_') or x.startswith('rewards_')]
sort_columns = ['original_index'] + [x for x in merged_df.columns if x.startswith('rewards')]
ascendings = [True] + [False] * (len(sort_columns) - 1)
#print(sort_columns)

#print(merged_df.explode(explode_columns).sort_values(sort_columns, ascending=ascendings))

#print(merged_df.head(2).explode(explode_columns).sort_values(sort_columns).drop(columns=['original_index', 'input']))
merged_df.explode(explode_columns).sort_values(sort_columns, ascending=ascendings).drop(columns=['original_index', 'input']).to_csv(f"{iteration_prefix}_summary.csv", index=False)
#merged_df.explode(explode_columns).drop(columns=['original_index', 'input']).to_csv(f"{iteration_prefix}_summary.csv", index=False)

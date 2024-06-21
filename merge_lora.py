import argparse
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--lora_model_path", help="lora_model_path")
parser.add_argument("--merged_dir", help="merged_dir")
args = parser.parse_args()

lora_model_path = args.lora_model_path
merged_dir = args.merged_dir


# モデルの読み込み
model = AutoPeftModelForCausalLM.from_pretrained(
    lora_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(
    lora_model_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# マージして保存
model = model.merge_and_unload()
model.save_pretrained(merged_dir, safe_serialization=True)

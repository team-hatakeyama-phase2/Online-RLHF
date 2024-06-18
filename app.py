import argparse
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import uvicorn
from accelerate import Accelerator
import torch
import deepspeed

parser = argparse.ArgumentParser()
parser.add_argument("--reward_model_path", help="reward_model_path")
args = parser.parse_args()


app = FastAPI()

pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
}

reward_model_path = args.reward_model_path
print(reward_model_path)

rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
rm_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_path,
    device_map="auto",
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    truncation=True,
)


@app.post(f"/classify/{reward_model_path.split('/')[-1]}")
async def classify_text(request: Request):
    data = await request.json()
    text = data['prompt']
    result = rm_pipe(text, **pipe_kwargs)

    return result

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

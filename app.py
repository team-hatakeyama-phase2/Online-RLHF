from fastapi import FastAPI, Request
from transformers import AutoTokenizer, pipeline
import uvicorn
from accelerate import Accelerator
import torch

app = FastAPI()

pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
}

reward_model = "/storage5/takagi/models/mistral_rm"

rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)
rm_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    device="cuda",
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    truncation=True,
)


@app.post("/classify")
async def classify_text(request: Request):
    data = await request.json()
    text = data['prompt']
    result = rm_pipe(text)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

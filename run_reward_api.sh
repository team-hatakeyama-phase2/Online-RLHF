#reward_model_path="/storage5/takagi/models/mistral_rm"
reward_model_path=/storage5/takagi/models/swallow_mx_rm_lora
#reward_model_path="sfairXC/FsfairX-LLaMA3-RM-v0.1"

python app.py --reward_model_path ${reward_model_path}

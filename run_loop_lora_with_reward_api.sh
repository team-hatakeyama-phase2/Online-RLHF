# source ~/.bashrc

# Initialize Conda environment
# eval "$(conda shell.bash hook)"


my_world_size=1 # how many gpu you use

# Base paths and settings
# initial_model="TODO: Set initial model path here"
# initial_model=TinyLlama/TinyLlama_v1.1
# initial_model=llm-jp/llm-jp-1.3b-v1.0
initial_model=lightblue/karasu-1.1B
#initial_model=CohereForAI/c4ai-command-r-v01

# initial_model=hatakeyama-llm-team/Tanuki-8B-Instruct-without-DPO
# initial_model=nk2t/Llama-3-8B-Instruct-japanese-nk2t-v0.3
# initial_model=mistralai/Mistral-7B-Instruct-v0.3 # ng
# initial_model=microsoft/Phi-3-mini-4k-instruct
# initial_model="hatakeyama-llm-team/Tanuki-8B-Instruct"
# initial_model=JINIAC/JINIAC-5B-sft_configuration-3_prod-checkpoint-500-dpo_merge_20240526_final
# initial_model=JINIAC/JINIAC-5B-sft_configuration-3_prod-checkpoint-500-dpo_merge_20240526_final

#prompt_path="/storage5/takagi/datasets/ELYZA-tasks-100"
#dataset_key="input"

#prompt_path="/storage5/takagi/datasets/hh-rlhf-12k-ja_mod"
#dataset_key="input"

prompt_path="/storage5/takagi/datasets/hh-rlhf-12k-ja_alpaca"
dataset_key="input"

#reward_model_path=/content/drive/MyDrive/Geniac/RLHFlow_reward_modeling/RLHF-Reward-Modeling/marged_model_full
# reward_model_path=/content/drive/MyDrive/Geniac/RLHFlow_reward_modeling/RLHF-Reward-Modeling/llama3_rm
reward_model_path=/storage5/takagi/models/mistral_rm
#reward_model_path=/storage5/takagi/models/swallow_mx_rm_lora

function get_last_dir() {
  path=$1

  array=( `echo $path | tr -s '/' ' '`)
  last_index=`expr ${#array[@]} - 1`
  echo ${array[${last_index}]}
  return 0
}

model_info=$(get_last_dir ${initial_model})
prompt_info=$(get_last_dir ${prompt_path})
reward_info=$(get_last_dir ${reward_model_path})
dpo_beta=0.2

echo "model:"${model_info}
echo "prompt:"${prompt_info}
echo "reward:"${reward_info}

predict_dir="./predicts"
model_dir="./models"

mkdir ${predict_dir}
mkdir ${model_dir}

iteration_prefix=${model_info}_${prompt_info}_${reward_info}_${dpo_beta}

generate_and_reward() {

    local iteration_name=$1
    local model_path=$2
    local input_promt=$3
    local output_generate=$4
    local output_reward=$5

    # sampling
    echo "sampling"
    python ./generation/get_hf2.py \
      --model_name_or_path ${model_path} \
      --dataset_name_or_path ${input_prompt} \
      --dataset_key ${dataset_key} \
      --output_dir ${output_generate} \
      --K 4 \
      --temperature 1.0 \
      --local_index 0 \
      --my_world_size ${my_world_size} \
      --max_new_tokens 2048 \
      --eos_ids 6 &
#      --eos_ids 128009 &

    wait
    python ./generation/merge_data.py \
      --base_path ${output_generate} \
      --output_dir ${output_generate} \
      --num_datasets ${my_world_size}

    # reward
    echo "reward"
#    python ./annotate_data/get_rewards_with_api.py \
    accelerate launch ./annotate_data/get_rewards.py \
       --dataset_name_or_path ${output_generate} \
       --output_dir ${output_reward} \
       --reward_name_or_path ${reward_model_path} \
       --K 4
}



# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration_name=$1
    local model_path=$2
    local input_promt=$3
    local output_generate=$4
    local output_reward=$5

    echo "iteration ${iteration_name}"
    echo "model_path ${model_path}"
    echo "input_prompt ${input_prompt}"
    echo "output_generate ${output_generate}"
    echo "output_reward ${output_reward}"

    generate_and_reward ${iteration_name} ${model_path} ${input_prompt} ${output_generate} ${output_reward}

    # train
    echo "train"
    accelerate launch \
      --config_file ./accelerate_configs/single_gpu.yaml \
      ./dpo_iteration/run_dpo_lora.py \
      --run_name ${iteration_name} \
      --output_dir ${iteration_name}_lora \
      --model_name_or_path ${model_path} \
      --ref_model ${initial_model} \
      --learning_rate 2e-7 \
      --num_train_epochs 1 \
      --choose_type max_min \
      --train_dir ${output_reward} \
      --eval_dir ${output_reward} \
      --loss_type sigmoid \
      --lr_scheduler_type cosine \
      --beta ${dpo_beta}

#    accelerate launch \
#      --config_file ./accelerate_configs/multi_gpu.yaml \
#      ./dpo_iteration/run_dpo_lora.py \
#      --run_name ${iteration_name} \
#      --output_dir ${iteration_name}_lora \
#      --model_name_or_path ${model_path} \
#      --ref_model ${initial_model} \
#      --learning_rate 2e-7 \
#      --max_steps 1200 \
#      --choose_type max_min \
#      --train_dir ${output_reward} \
#      --eval_dir ${output_reward} \
#      --loss_type sigmoid \
#      --lr_scheduler_type cosine \
#      --beta ${dpo_beta}


    # lora merge
    echo "merge"
    echo from ${iteration_name}_lora
    python ./merge_lora.py \
      --lora_model_path ${iteration_name}_lora \
      --merged_dir ${iteration_name}
    cp ./${iteration_name}_lora/tokenizer* ./${iteration_name}/
    cp ./${iteration_name}_lora/special_tokens_map.json ./${iteration_name}/

}

# Function to run a set of operations for a model iteration
last_predict() {
    local iteration_name=$1
    local model_path=$2
    local input_promt=$3
    local output_generate=$4
    local output_reward=$5

    echo "iteration ${iteration_name}"
    echo "model_path ${model_path}"
    echo "input_prompt ${input_prompt}"
    echo "output_generate ${output_generate}"
    echo "output_reward ${output_reward}"

    generate_and_reward ${iteration_name} ${model_path} ${input_prompt} ${output_generate} ${output_reward}
}


# Main loop for iterations
for i in {1..3}
#for i in 1
do
    echo "i="${i}
    iteration_name="${model_dir}/${iteration_prefix}/iter${i}"
    input_prompt=${prompt_path} # json format
    dataset_key=${dataset_key}
    output_generate="${predict_dir}/${iteration_prefix}_${i}.json"
    output_reward="${predict_dir}/${iteration_prefix}_${i}_reward.json"

    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        model_path=${initial_model}
    else
        previous_iteration=$((i-1))
        model_path="${model_dir}/${iteration_prefix}/iter${previous_iteration}"
    fi

    run_iteration ${iteration_name} ${model_path} ${input_prompt} ${output_generate} ${output_reward}
done

echo "end loop"

echo "last predict"

i=$((i+1))
echo "i="${i}
previous_iteration=$((i-1))
model_path="${model_dir}/${iteration_prefix}/iter${previous_iteration}"

iteration_name="${model_dir}/${iteration_prefix}/iter${i}"
input_prompt=${prompt_path} # json format
dataset_key=${dataset_key}
output_generate="${predict_dir}/${iteration_prefix}_${i}.json"
output_reward="${predict_dir}/${iteration_prefix}_${i}_reward.json"

last_predict ${iteration_name} ${model_path} ${input_prompt} ${output_generate} ${output_reward}

sleep 10
echo "make_csv"

python make_summary.py \
  --input_prompt ${input_prompt} \
  --predict_dir ${predict_dir} \
  --iteration_prefix ${iteration_prefix}

echo "dump at "${iteration_prefix}_summary.csv

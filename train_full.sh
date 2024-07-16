wizard_path="/GPFS/data/shuotang-1/Matrix/datasets/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.json"
lima_path="/GPFS/data/shuotang-1/Matrix/datasets/lima/train.json"
alpaca_gpt4_path="/GPFS/data/yaxindu-1/FastChat/dataset/alpaca_gpt4.json"
sharegpt_path="/GPFS/data/ruiye-1/instruction_tuning/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
wildchat_path="/GPFS/data/shuotang-1/Matrix/datasets/wildchat/wildchat_rand50k.json"
ultrachat_path="/GPFS/data/zexiliu-1/LargeScaleSim/dataset/ultrachat_200k/ultrachat_rand50k.json"
openhermes_path="/GPFS/data/shuotang-1/Matrix/datasets/openhermes/openhermes.json"
ours_profile_path="/GPFS/data/shuotang-1/Matrix/datasets/ours_profile_50k.json"
ours_twitter_path="/GPFS/data/shuotang-1/Matrix/datasets/ours_twitter_50k.json"
##############################
vicuna_ram_path="/dev/shm/shuotang_ramdisk/vicuna-7b-v1.5/"
model_path="/GPFS/data/shuotang-1/LLM/llama3/llama3-8b-hf/"
llama3_resized_path="/GPFS/data/xhpang-1/LLM/llama3-8b-resized/"
llama2_path="/dev/shm/shuotang_ramdisk/llama2"
# model_ram_path="/dev/shm/shuotang_ramdisk/llama3-8b-hf/"
model_ram_path="/dev/shm/shuotang_ramdisk/llama3-8b-resized/"
##############################
output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0704_lima_llama3_temp_seq1024_fullV3_lr/"
wizard_output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0705_wizard_llama3_temp_seq1024_fullV3_lr/"
lima_output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0707_lima_llama3_temp_seq1024_fullV3_lr_lora/"
sharegpt_output_path="/GPFS/data/xhpang-1/FastChat/checkpoint/llama3_0710_sharegpt50k_llama3_temp_seq1024_fullV3_lr/"
ours_profile_10k_output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0711_profile10k_seq1024_fullV3/"
ours_profile_50k_output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0711_profile50k_true_seq1024_fullV3/"
ours_twitter_10k_output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0708_twitter10k_seq1024_fullV3/"
ours_twitter_50k_output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0711_twitter50k_seq1024_fullV3/"
# wildchat_10k_output_path="/GPFS/data/sihengchen-1/llm/Matrix/Fastchat/checkpoint/llama3_0711_wildchat10k_seq1024_fullV3/"
wildchat_10k_output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0715_wizard_llama3_temp_seq1024_loraV3_lr"
wildchat_50k_output_path="/GPFS/data/zexiliu-1/LargeScaleSim/FastChat/checkpoint/llama3_0711_wildchat50k_seq1024_fullV3/"
openhermes_10k_output_path="/GPFS/data/zexiliu-1/LargeScaleSim/FastChat/checkpoint/llama3_0712_openhermes10k_seq1024_fullV3/"
openhermes_50k_output_path="/GPFS/data/zexiliu-1/LargeScaleSim/FastChat/checkpoint/llama3_0712_openhermes50k_seq1024_fullV3/"
ultrachat_10k_output_path="/GPFS/data/zexiliu-1/LargeScaleSim/FastChat/checkpoint/llama3_0712_ultrachat10k_seq1024_fullV3/"
ultrachat_50k_output_path="/GPFS/data/zexiliu-1/LargeScaleSim/FastChat/checkpoint/llama3_0712_ultrachat50k_seq1024_fullV3/"

export WANDB_DISABLED=true

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=20004 fastchat/train/train_full.py \
    --model_name_or_path ${model_ram_path} \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ${wildchat_path} \
    --data_size 10000 \
    --fp16 True \
    --output_dir ${wildchat_10k_output_path} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 100 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing \
    --deepspeed playground/deepspeed_config_s3.json \


# deepspeed --include localhost:4,5,6,7 --master_port=20004 fastchat/train/train_lora.py \
#     --model_name_or_path ${llama2_path} \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --data_path ${alpaca_gpt4_path} \
#     --bf16 True \
#     --output_dir ${output_path} \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_steps 1200 \
#     --save_total_limit 100 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 1024 \
#     --q_lora True \
#     --gradient_checkpointing \
#     --deepspeed playground/deepspeed_config_s2.json \

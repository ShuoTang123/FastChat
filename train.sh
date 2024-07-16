wizard_path="/GPFS/data/shuotang-1/Matrix/datasets/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.json"
lima_path="/GPFS/data/shuotang-1/Matrix/datasets/lima/train.json"
alpaca_gpt4_path="/GPFS/data/yaxindu-1/FastChat/dataset/alpaca_gpt4.json"
sharegpt_path="/GPFS/data/ruiye-1/instruction_tuning/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
##############################
vicuna_ram_path="/dev/shm/shuotang_ramdisk/vicuna-7b-v1.5/"
model_path="/GPFS/data/shuotang-1/LLM/llama3/llama3-8b-hf/"
llama3_resized_path="/GPFS/data/xhpang-1/LLM/llama3-8b-resized/"
llama2_path="/dev/shm/shuotang_ramdisk/llama2"
# model_ram_path="/dev/shm/shuotang_ramdisk/llama3-8b-hf/"
model_ram_path="/dev/shm/shuotang_ramdisk/llama3-8b-resized/"
##############################
output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0704_lima_llama3_temp_seq1024_fullV3_lr/"
wizard_output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0708_wizard_llama3_temp_seq1024_loraV3_lr/"
lima_output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0707_lima_llama3_temp_seq1024_fullV3_lr_lora/"
sharegpt_output_path="/GPFS/data/shuotang-1/Matrix/FastChat/checkpoint/llama3_0705_sharegpt_llama3_temp_seq1024_fullV3_lr/"


deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=20004 fastchat/train/train_lora.py \
    --model_name_or_path ${llama3_resized_path} \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ${wizard_path} \
    --fp16 True \
    --output_dir ${wizard_output_path} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1200 \
    --save_total_limit 100 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
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

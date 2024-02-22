# Swallow TART

## Features
- DeepSpeed
  - To install it, see [here](../../../install-deepspeed.sh).
- Flash-Attention
  - To install it, see [here](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).

## Training

```bash
$ deepspeed examples/training/swallow-tart/run_train.py \
  --hf_dataset_dir $HF_DATA \
  --data_dir $DATA \
  --task_names $TASKS \
  --max_length $MAX_LENGTH \
  --n_dev_sample $N_DEV_SAMPLE \
  --num_proc $N_PROC \
  --model_name $MODEL_NAME \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size $BATCH \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --weight_decay $WEIGHT_DECAY \
  --warmup_steps $WARMUP \
  --logging_steps $LOG_STEP \
  --save_steps $SAVE_STEP \
  --save_total_limit $SAVE_LIMIT \
  --bf16 \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --gradient_checkpointing \
  --use_flash_attention \
  --remove_unused_columns False \
  --deepspeed $DS_CONFIG
```

If you want to use LoRA, add `--peft_config_path $LORA_CONFIG` to the command.

## Configs
- DeepSpeed config
  - Zero3 (cpu-offload): [ds_config_zero3.json](./configs/ds_config_zero3.json)
- LoRA config
  - LoRA: [lora_config.json](./configs/lora_config.json)

set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 32768 \
   --dataset /home/wangxiaorong/workspace/o1/datasets/UltraInteract_sft/train.jsonl \
   --input_key instruction \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 288579 \
   --pretrain /home/wangxiaorong/workspace/o1/checkpoints/Mistral-7B-Instruct-v0.2 \
   --save_path /home/wangxiaorong/workspace/o1/trained_llms/mistral-7b-sft-ui \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 2e-5 \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

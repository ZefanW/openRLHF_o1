# 需要确定reward model的prompt
set -x
# export TORCH_CPP_LOG_LEVEL=INFO
read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain /home/wangxiaorong/workspace/o1/checkpoints/mistral-7b-sft-ui \
   --reward_pretrain /home/wangxiaorong/workspace/o1/checkpoints/Eurus-RM-7b \
   --save_path /home/wangxiaorong/workspace/o1/trained_llms/mistral-7b-eurus-7b-ppo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 1024 \
   --max_epochs 1 \
   --prompt_max_len 8192 \
   --generate_max_len 8192 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /home/wangxiaorong/workspace/o1/datasets/UltraInteract_pair/train_ppo.jsonl \
   --input_key context_messages \
   --apply_chat_template \
   --max_samples 219522 \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

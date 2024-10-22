set -x 

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/wangxiaorong/workspace/o1/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain /home/wangxiaorong/workspace/o1/checkpoints/llama3-8b-sft-ui \
   --reward_pretrain /home/wangxiaorong/workspace/o1/checkpoints/Eurus-RM-7b \
   --save_path /home/wangxiaorong/workspace/o1/trained_llms/llama3-8b-eurus-7b-ppo \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --max_samples 219522 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /home/wangxiaorong/workspace/o1/datasets/UltraInteract_pair/train_ppo.jsonl \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   # --use_wandb fa8a70323203ed1a98ed385e2b0fb1391b1e4812

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
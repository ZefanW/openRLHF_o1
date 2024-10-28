set -x 
ray stop --force
ray start --head --num-gpus 8 --num-cpus 64
# 测试基本ray代码中，貌似是tensor_parallel不能设置为1，设为1有卡死问题。

RAY_ADDRESS='http://127.0.0.1:8265' ray job submit \
   --runtime-env /home/wangzefan/data/OpenRLHF/examples/env_vars/runtime-env.json \
   -- python3 /home/wangzefan/data/OpenRLHF/openrlhf/cli/train_ppo_ray.py \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --save_steps -1 \
   --logging_steps 1 \
   --pretrain /data/public/wangshuo/LongContext/model/meta-llama/Llama3.1-8b-instruct \
   --reward_pretrain /home/wangzefan/huggingface/Skywork-Reward-Llama-3.1-8B \
   --save_path /home/wangzefan/data/OpenRLHF/jobs/llama-ppo-rs-ui-math-only-001kl \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 128 \
   --rollout_batch_size 1024 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /home/wangzefan/data/OpenRLHF/datasets/UltraInteract_pair_math/train_ppo.jsonl \
   --input_key trajectory \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb True \
   --reward_shaping_function qwen_math_ui \
#   --colocate_critic_reward \
#   --colocate_actor_ref \
#   --ref_reward_offload \
#   --load_checkpoint \
   # --use_wandb fa8a70323203ed1a98ed385e2b0fb1391b1e4812

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]

# vllm_num_engines表示总共需要启动多少个engine。每个engine占用的gpu数量等同于tensor_parallel_size。
# 启动的vllm一般是不退出的。

# save_steps决定global step为多少的整数倍时进行ckpt存储，其存储目录为ckpt_path
# 最终ckpt存储在save_path
# 慎用相对路径，有时候会出问题

# wandb的存储路径依靠全局变量指定

# prm的用法：需要指定触发prm的字符串（比如Step ），然后指定触发prm计算的特殊token。prm同样存在hack问题，默认配合reward shaping修正一下reward。
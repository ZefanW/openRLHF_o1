set -x 
ray stop --force
ray start --head --num-gpus 8 --num-cpus 64
# 测试基本ray代码中，貌似是tensor_parallel不能设置为1，设为1有卡死问题。
# 本脚本使用prm训练，叠加reward shaping和delta功能。这两个功能都放在reward shaping里。
# 训练数据包含代码和数学

RAY_ADDRESS='http://127.0.0.1:8265' ray job submit \
   --runtime-env /home/wangzefan/data/OpenRLHF/examples/env_vars/runtime-env.json \
   -- python3 /home/wangzefan/data/OpenRLHF/openrlhf/cli/train_ppo_ray.py \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --save_steps -1 \
   --logging_steps 1 \
   --pretrain /home/wangzefan/data/OpenRLHF/checkpoints/Mistral-7B-Instruct-v0.2 \
   --process_reward_pretrain /home/wangzefan/data/OpenRLHF/checkpoints/Eurus-RM-7b \
   --save_path /home/wangzefan/data/OpenRLHF/jobs/mistral-prm-delta \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 128 \
   --rollout_batch_size 128 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /home/wangzefan/data/OpenRLHF/datasets/UltraInteract_0911/have_reference_toy.jsonl  \
   --input_key trajectory \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb True \
   --reward_shaping_function qwen_like \
   --prm_type shepherd
#   --prm_trigger "Step " \
#   --prm_token_id 12902

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
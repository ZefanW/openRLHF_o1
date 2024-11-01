import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, masked_mean
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray


logger = init_logger(__name__)


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.
    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.returns = self.returns.pin_memory()
        self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn


    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        value = self.critic(sequences, action_mask, attention_mask)

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)
            # r = self.reward_shaping(r, sequences, attention_mask)

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.
        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...
        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...
        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)
        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], original_data=None, reward_function=None,**generate_kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()

        # generate sequence
        start = time.time()
        sequences_all, attention_mask_all, action_mask_all = (
            self._generate_local(prompts, **generate_kwargs)
            if self.vllm_engines is None
            else self._generate_vllm(prompts, **generate_kwargs)
        )
        generate_time = time.time() - start

        num_actions = action_mask_all.size(1)
        sequences_cpu_all, attention_mask_cpu_all, action_mask_cpu_all = (
            sequences_all.to("cpu"),
            attention_mask_all.to("cpu"),
            action_mask_all.to("cpu"),
        )

        # TODO: 需要清理一些误导性注释。现在prm的step识别和预测规则全盘交给actor model，experience部分只做简单处理。

        # 服务于prm，计算插入过prm_token的结果
        # prm_trigger = getattr(self.strategy.args, 'prm_trigger', None) # 如果是None，后面只会在结尾处添加一个prm_token_id
        # prm_token_id = getattr(self.strategy.args, 'prm_token_id', None)
        # assert prm_token_id is not None

        # 只要trigger不是空串，就需要在sequences里插入token
        # PRM工作流程：将回答提取出来，反tokenize，监测prm_trigger的存在，根据trigger信号给文本分段，分段后分别tokenize在最后加上prm_token_id，然后合并起来。这样获得的结果再拿去跑reward model。只要返回的reward不是只有batch维度，就说明遇到了一个PRM。获取sequences中等于prm_trigger的位置，把这些位置的reward提出来，就是prm reward。这些reward的位置在后续会减去前序trigger token的数量，并且缩到合理范围内，然后加到reward trajectory上。
        # texts = self.tokenizer.batch_decode(sequences_cpu_all[:, -num_actions:], skip_special_tokens=True)
        # # 1024的prompt长度偶尔会导致user prompt不完整的问题
        # texts_splits=[text.split(prm_trigger) for text in texts]
        # token_lists=[]
        # for texts_split in texts_splits:
        #     cur_token_list=[]
        #     for split_id in range(len(texts_split)):
        #         cur_text=(prm_trigger if split_id>0 else '')+texts_split[split_id]
        #         cur_token_list.extend(self.tokenizer.encode(cur_text,add_special_tokens=False)+[prm_token_id])
        #     token_lists.append(cur_token_list)
        #
        # texts_prm=torch.nn.utils.rnn.pad_sequence([torch.tensor(token_list) for token_list in token_lists],batch_first=True,padding_value=self.tokenizer.pad_token_id)
        # sequences_cpu_prm=torch.cat([sequences_cpu_all[:,:-num_actions],texts_prm],dim=1)
        # attention_mask_cpu_prm=torch.cat([attention_mask_cpu_all[:,:-num_actions],texts_prm!=self.tokenizer.pad_token_id],dim=1)


        # prm_token = self.tokenizer.convert_ids_to_tokens([prm_token_id])[0].replace('▁','')
        # texts_prm = [texts_.replace(prm_trigger, prm_token + prm_trigger) for texts_ in texts]
        # response_prm_tokenized = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")
        # from IPython import embed; embed()
        # sequences_cpu_prm = torch.cat([sequences_cpu_all[:,:-num_actions],response_prm_tokenized['input_ids']], dim=1)
        # attention_mask_cpu_prm = torch.cat([attention_mask_cpu_all[:,:-num_actions],response_prm_tokenized['attention_mask']], dim=1)
        # chunk部分只需要保留每个sequence中id等于prm_token_id的即可。注意需要左pad，以确保reward不会出现在sequence外。


        # 防止huggingface inference成为micro rollout batch size的瓶颈，因此在这里把vllm的推理结果切成micro train batch size
        # 后面所有模型推理过程都遵循相同设定，

        chunk_size=generate_kwargs['micro_train_batch_size'] if 'micro_train_batch_size' in generate_kwargs else len(prompts)


        ref_values_list=[]
        action_log_probs_list=[]

        for i in range(0, sequences_all.size(0), chunk_size):
            sequences_cpu_chunk=sequences_cpu_all[i:i+chunk_size]
            attention_mask_cpu_chunk=attention_mask_cpu_all[i:i+chunk_size]
            action_mask_cpu_chunk=action_mask_cpu_all[i:i+chunk_size]
            sequences_chunk=sequences_all[i:i+chunk_size]
            attention_mask_chunk = attention_mask_all[i:i + chunk_size]
            # sequences_cpu_prm_chunk=sequences_cpu_prm[i:i+chunk_size]
            # attention_mask_cpu_prm_chunk=attention_mask_cpu_prm[i:i+chunk_size]

            if original_data is not None:
                original_data_chunk=original_data[i:i+chunk_size]

            # init log probs
            base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu_chunk, num_actions, attention_mask_cpu_chunk)

            # values
            value_ref = self.critic.forward.remote(sequences_cpu_chunk, action_mask_cpu_chunk, attention_mask_cpu_chunk)


            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])

            if self.strategy.args.colocate_actor_ref:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])

            # rewards, 可以叠加多个reward model，纯粹根据返回值区分是prm还是orm。gt部分总是塞进reward shaping中。
            r_refs = []
            # support remote RM API with ray
            if not self.remote_rm_url:
                for rm in self.reward_model:

                    if ray.get(rm.is_prm.remote()):
                        r_refs.append(rm.forward.remote(sequences_cpu_chunk, attention_mask_cpu_chunk, num_actions, ray.get(rm.find_step_end.remote(sequences_cpu_chunk, num_actions, self.tokenizer))))
                    else:
                        r_refs.append(rm.forward.remote(sequences_cpu_chunk, attention_mask_cpu_chunk, ))
            else:
                # remote RM
                for rm in self.remote_rm_url:
                    queries = self.tokenizer.batch_decode(sequences_all.cpu(), skip_special_tokens=False)
                    r = remote_rm_fn_ray.remote(rm, queries=queries)
                    r_refs.append(r)

            # log probs
            start = time.time()
            action_log_probs_chunk = self.actor(sequences_chunk, num_actions, attention_mask_chunk).cpu()
            actor_time = time.time() - start

            # get gt reward before waiting models done. if no reward_shaping method, gt_reward is always 0
            if reward_function is not None:
                gt_reward_chunk=reward_function.get_reward(sequences_chunk, attention_mask_chunk, original_data_chunk, num_actions)
            else:
                gt_reward_chunk=torch.zeros_like(sequences_chunk[:,0])

            # wait initial/critic/reward model done
            start = time.time()
            ref_values_chunk = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
            wait_time = time.time() - start

            # collect all results
            ref_values_chunk.append(gt_reward_chunk)
            ref_values_list.append(ref_values_chunk)
            action_log_probs_list.append(action_log_probs_chunk)

        # concat all results
        ref_values=[]
        for i in range(len(ref_values_list[0])):
            ref_values.append(torch.cat([ref_values_list_item[i] for ref_values_list_item in ref_values_list], dim=0))
        action_log_probs=torch.cat(action_log_probs_list, dim=0).to(device)

        base_action_log_probs, value, rewards, gt_rewards= ref_values[0], ref_values[1], ref_values[2:-1], ref_values[-1]
        base_action_log_probs, value, gt_rewards= base_action_log_probs.to(device), value.to(device), gt_rewards.to(device)



        # 处理rewards中的prm，将其转化为index和value的形式。value为了和orm可以直接叠加，进行左padding
        # rewards=[r if len(r.shape)==1 else reward_function.extract_process_rewards(r,sequences_cpu_prm, prm_token_id) for r in rewards]

        rewards = [r.to(device) for r in rewards]


        # if reward function is not None, call it. This should adapt to multi reward model.
        if reward_function is not None:
            # this will always return 2D tensor. we need to apply it back to the sequence.
            # the method is to add it back after compute_reward()
            r_last, process_r= reward_function(gt_rewards, rewards, action_mask_all)
        else:
            # Does not support PRM
            r_last = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]
            # r=r.unsqueeze(1)

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        reward, kl = compute_reward(
            r_last, # last step of r
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask_all,
        )

        if reward_function is not None:
            reward[~torch.isinf(process_r)]+=process_r[~torch.isinf(process_r)] # 这一步里面process_r为0的部分不会产生任何影响


        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask_all,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask_all, dim=-1),
            "reward": reward.sum(dim=1), # log total reward
            # "reward_rm": torch.zeros_like(r), # 只有reward model的不太好计算。
            "reward_gt": gt_rewards, # log gt reward
            "return": reward.sum(dim=-1), # 训练过程中记录的return是r-kl的结果，随着训练过程而逐渐增长没有问题。
            "response_length": action_mask_all.float().sum(dim=-1),
            "total_length": attention_mask_all.float().sum(dim=-1),
        }

        if self.strategy.args.perf: # 一般不用
            batch_size = 1 if isinstance(prompts, str) else len(prompts)
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

        experience = Experience(
            sequences_all,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask_all,
            action_mask_all,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        return self.actor.generate(**inputs, **kwargs)

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
        assert self.tokenizer.padding_side == "left", f"tokenizer padding_size should be left"
        pad_indices = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.int).argmax(dim=-1)
        prompt_token_ids = []
        for i, pad_index in enumerate(pad_indices.numpy()):
            prompt_token_ids.append(input_ids[i][pad_index:].tolist())
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in outputs:
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for output in outputs:
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

            if output_ids[output_len - 1] != eos_token_id:
                output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None
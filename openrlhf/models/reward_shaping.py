import json

# reward shaping functions，每一个函数都接收prompt, completion, evidence, prompt_token, completion_token作为参数，
# call函数每次载入一个完整数据（必须是完整数据，所有key都要保留），然后自己编写具体逻辑。
# 作为加速手段，gt评测和rm评测应当可以并发进行
import torch
from openrlhf.evaluation_utils.parallel_evaluation import process_completion
def qwen_math_ui(*args, **kwargs):
    class QwenMath:
        """
        模拟qwen2.5 math的reward shaping。
        r=sigmoid(0.5*r_m)+(r_v-1)
        为了防止模型回答了正确答案以后继续废话，这里的设置为response的最后50 token中必须exactly contains response。（粗略实现）
        具体逻辑应该根据SFT模型的回答风格来修改。
        """
        def __init__(self, *args, **kwargs):
            self.tokenizer=kwargs['tokenizer']
            self.math_keys=['Math_CoT']
            self.code_keys=[]

        def __call__(self, r, sequences, attention_mask, original_data, num_actions):
            # r是一维向量，在cuda上，shape=(micro_rollout_batch_size)。这么看来openRLHF默认不能做PRM
            reward_shaped=torch.zeros_like(r)
            for i, (r_, sequence_, attention_mask_, original_data_) in enumerate(zip(r, sequences, attention_mask, original_data)):
                original_data_dict=json.loads(original_data_)
                # print(original_data_dict)
                chosen=original_data_dict['chosen']
                success=0.5
                model_sequence = self.tokenizer.decode(sequence_[-num_actions:], skip_special_tokens=True)
                # print(model_sequence)
                if original_data_dict['task'] in self.math_keys:
                # if chosen.__contains__('Answer:\n\\boxed{'):
                    reference=chosen.split('Answer:\n\\boxed{')[-1].split('}')[0]
                    is_correct, format_correctness, extracted_model_output= process_completion(model_sequence, 'math',reference)
                    if is_correct:
                        success=1
                    else:
                        success=0
                elif original_data_dict['task'] in self.code_keys:
                    raise NotImplementedError

                reward_shaped[i]=torch.nn.functional.sigmoid(r_*0.5)+success-1

            # print(r,'->',reward_shaped)
            return reward_shaped

    return QwenMath(*args, **kwargs)

def qwen_like(*args, **kwargs):
    class QwenLike:
        """
        shaping方法为混合原有reward array和prm，gt有5倍权重，剩下的有1倍。
        如果用了prm，进行delta处理来防止对长回答的过度偏好。
        rm数值一律在做过sigmoid以后再进行处理。
        TODO code和mathPoT评测太慢，且code开多进程评测会报错。
        """
        def __init__(self, *args, **kwargs):
            self.tokenizer=kwargs['tokenizer']
            self.math_keys=['Math_CoT']
            # self.code_keys=['Coding']
            self.code_keys=[]
            # self.repl_keys=['Math_PoT']
            self.repl_keys=[]
            # 只有coding任务需要自行抽取代码

        def get_reward(self, sequences, attention_mask, original_data, num_actions):
            reward_gt = torch.zeros_like(sequences[:,0]).to(torch.bfloat16)
            for i, (sequence_, attention_mask_, original_data_) in enumerate(
                    zip(sequences, attention_mask, original_data)):
                original_data_dict = json.loads(original_data_)
                task = original_data_dict['task']
                success = 0.5
                is_correct = None
                model_sequence = self.tokenizer.decode(sequence_[-num_actions:], skip_special_tokens=True)

                if task in self.math_keys:
                    reference = original_data_dict['reference']
                    prediction = model_sequence
                    res = process_completion(prediction, 'math', reference)
                    is_correct = res[0]
                elif task in self.code_keys:
                    reference=json.loads(original_data_dict['reference'])
                    prediction=model_sequence.split('```python')[-1].split('```')[0]
                    res=process_completion(prediction,'code',reference)
                    is_correct = res[0]
                elif task in self.repl_keys:
                    reference=original_data_dict['reference']
                    prediction=model_sequence
                    res=process_completion(prediction,'repl',reference)
                    is_correct = res[0]

                if is_correct is not None:
                    if is_correct:
                        success = 1
                    else:
                        success = 0

                reward_gt[i] = success
            return reward_gt
        def __call__(self, reward_gt,rewards_rm):
            """
            always return 2D tensor [Batch, Step]
            """
            # rewards_rm为一个列表，包含多个rm的结果。其中shape=[B]的为orm，shape=[B,S]的为prm。prm至多有一个
            # check if there exist prm
            reward_shaped=None
            for reward_rm in rewards_rm:
                if len(reward_rm.shape)>1:
                    reward_shaped=torch.zeros_like(reward_rm)
            if reward_shaped is None:
                reward_shaped=torch.zeros_like(rewards_rm).unsqueeze(1)

            # adding rewards
            for reward_rm in rewards_rm:
                if len(reward_rm.shape)>1:
                    reward_shaped+=reward_rm
                else:
                    reward_shaped[:,-1]+=reward_rm

            # applying sigmoid
            reward_shaped=torch.sigmoid(reward_shaped)

            # applying weighting
            reward_shaped[:,-1]+=5*reward_gt

            return reward_shaped

    return QwenLike(*args, **kwargs)
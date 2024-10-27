import json

# reward shaping functions，每一个函数都接收prompt, completion, evidence, prompt_token, completion_token作为参数，
# call函数每次载入一个完整数据（必须是完整数据，所有key都要保留），然后自己编写具体逻辑。
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
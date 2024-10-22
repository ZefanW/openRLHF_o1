from transformers import AutoTokenizer, AutoModel
import torch


def test(model_path):
    dataset = [  # cases in webgpt; we use the same template as Mistral-Instruct-v0.2
        {
            "chosen": "[INST] Sural relates to which part of the body? [\INST] The sural region is the muscular swelling of the back of the leg below the knee, formed chiefly by the bellies of the gastrocnemius and soleus muscles [1,2].",
            "rejected": "[INST] Sural relates to which part of the body? [\INST] The Sural nerve runs down the side of the leg near the small saphenous vein, then passes forward below the lateral malleolus and continues on the outside of the foot as the lateral dorsal cutaneous nerve, which then communicates with the intermediate dorsal cutaneous nerve, which branches off to the side of the foot. [1]",
        }
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(example["chosen"], return_tensors="pt")
            chosen_reward = model(**inputs).item()
            inputs = tokenizer(example["rejected"], return_tensors="pt")
            rejected_reward = model(**inputs).item()
            print(f"Chosen reward: {chosen_reward}, Rejected reward: {rejected_reward}")
            print(chosen_reward - rejected_reward)


test("/home/wangxiaorong/workspace/o1/OpenRLHF/checkpoints/Eurus-RM-7b")
# Output: 47.4404296875

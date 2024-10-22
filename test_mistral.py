from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# def test(model_path):
#     dataset = [  # cases in webgpt; we use the same template as Mistral-Instruct-v0.2
#         {
#             "question": "[INST] Sural relates to which part of the body? [\INST]",
#             # The sural region is the muscular swelling of the back of the leg below the knee, formed chiefly by the bellies of the gastrocnemius and soleus muscles [1,2].
#         }
#     ]

#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
#     model.to("cuda")

#     with torch.no_grad():
#         for example in dataset:
#             generated_ids = model.generate(example["question"], max_new_tokens=1000, do_sample=True)
#             # decode with mistral tokenizer
#             result = tokenizer.decode(generated_ids[0].tolist())
#             print(result)


# test("/home/wangxiaorong/workspace/o1/trained_llms/mistral-7b-sft-ui")
# Output: 47.4404296875
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model_path = "/home/wangxiaorong/workspace/o1/checkpoints/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {"role": "user", "content": "Sural relates to which part of the body?"},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])


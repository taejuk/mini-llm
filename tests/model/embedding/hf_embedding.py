from transformers import AutoTokenizer, GPT2Model
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

sequence = "Hello, my dog is cute"
inputs = tokenizer(sequence, return_tensors="pt")
print(inputs)

# token embedding이 정상적으로 되는가?

# logit이 정상적으로 구해지는가?





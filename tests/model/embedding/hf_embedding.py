from transformers import AutoTokenizer, GPT2Model
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

sequence = "Hello, my dog is cute"
inputs = tokenizer(sequence, return_tensors="pt")
# embedding 결과이다.
outputs = model(**inputs)
# embedding 결과를 쓰고

# 
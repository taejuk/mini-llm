import os
import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel


def save_npy(path: str, tensor: torch.Tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    np.save(path, arr)
    print(f"saved {path}, shape={arr.shape}, dtype={arr.dtype}")


def main():
    out_dir = "hf_ref"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        output_hidden_states=True,
        output_attentions=True,
    ).to(device)

    model.eval()

    prompt = "Hello, my name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    print("prompt:", prompt)
    print("input_ids:", input_ids.tolist())

    # ------------------------------------------------------------
    # 1. Embedding 직접 계산
    # ------------------------------------------------------------
    batch_size, seq_len = input_ids.shape

    position_ids = torch.arange(
        0,
        seq_len,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        token_embeds = model.transformer.wte(input_ids)
        pos_embeds = model.transformer.wpe(position_ids)
        embedding_out = token_embeds + pos_embeds

    save_npy(f"{out_dir}/input_ids.npy", input_ids)
    save_npy(f"{out_dir}/token_embedding.npy", token_embeds)
    save_npy(f"{out_dir}/position_embedding.npy", pos_embeds)
    save_npy(f"{out_dir}/embedding_out.npy", embedding_out)

    # ------------------------------------------------------------
    # 2. Hook 등록: block 내부 중간 결과 저장
    # ------------------------------------------------------------
    captured = {}

    def make_hook(name):
        def hook(module, module_input, module_output):
            # module_output이 tuple인 경우 첫 번째만 저장
            if isinstance(module_output, tuple):
                module_output = module_output[0]
            captured[name] = module_output.detach()
        return hook

    # block 0 기준으로 먼저 검증
    block0 = model.transformer.h[0]

    hooks = []
    hooks.append(block0.ln_1.register_forward_hook(make_hook("block0_ln1_out")))
    hooks.append(block0.attn.c_attn.register_forward_hook(make_hook("block0_qkv_proj_out")))
    hooks.append(block0.attn.c_proj.register_forward_hook(make_hook("block0_attn_proj_out")))
    hooks.append(block0.ln_2.register_forward_hook(make_hook("block0_ln2_out")))
    hooks.append(block0.mlp.c_fc.register_forward_hook(make_hook("block0_mlp_fc_out")))
    hooks.append(block0.mlp.c_proj.register_forward_hook(make_hook("block0_mlp_proj_out")))

    # ------------------------------------------------------------
    # 3. 전체 forward
    # ------------------------------------------------------------
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
        )

    logits = outputs.logits
    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    past_key_values = outputs.past_key_values

    for h in hooks:
        h.remove()

    # ------------------------------------------------------------
    # 4. hidden states 저장
    # hidden_states[0] = embedding output
    # hidden_states[1] = block 0 output
    # ...
    # hidden_states[12] = block 11 output
    # ------------------------------------------------------------
    for i, h in enumerate(hidden_states):
        save_npy(f"{out_dir}/hidden_state_{i}.npy", h)

    save_npy(f"{out_dir}/logits.npy", logits)

    # attention weights
    for i, attn in enumerate(attentions):
        save_npy(f"{out_dir}/attention_{i}.npy", attn)

    # KV cache
    for layer_idx, (k, v) in enumerate(past_key_values):
        save_npy(f"{out_dir}/past_key_layer_{layer_idx}.npy", k)
        save_npy(f"{out_dir}/past_value_layer_{layer_idx}.npy", v)

    # block0 internal captures
    for name, tensor in captured.items():
        save_npy(f"{out_dir}/{name}.npy", tensor)

    # next token
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    print("next_token_id:", next_token.tolist())
    print("next_token_text:", tokenizer.decode(next_token[0]))

    np.save(f"{out_dir}/next_token.npy", next_token.detach().cpu().numpy())

    print("\nDone.")


if __name__ == "__main__":
    main()
"""第 1 步：用 transformers 加载本地 Qwen2.5-0.5B-Instruct，跑一句推理。

这是 P3（端侧大模型）的第一个里程碑：
验证「用 transformers 写代码调 Qwen」这条路在你机器上能跑通。

和 ollama 的区别（重点理解）：
- ollama：你调 API，模型当黑盒用，看不到内部。
- transformers：模型对象直接在你的 Python 进程里，你能改它、微调它、量化它。
  这正是 P3 要学的（LoRA 微调、量化）的前提。

模型已经下载到本地 HF 缓存（~/.cache/huggingface/hub/），本文件直接从缓存读，不联网。

前置（必须先装）：
    pip install transformers torch

运行：
    python 01_load_and_chat.py
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"        # 模型已在缓存，关掉联网检查：国内更稳、加载更快
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------- 1. 加载 tokenizer + 模型（从本地缓存读）----------
print(f"加载 {MODEL_ID}（首次读盘约几秒）...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)   # 默认 CPU + float32；0.5B 笔记本轻松跑
model.eval()

# 看一眼模型规模，建立直觉
n_params = sum(p.numel() for p in model.parameters())
print(f"参数量：{n_params/1e6:.0f}M ≈ {n_params/1e9:.2f}B")

# ---------- 2. 构造对话（指令模型必须套 chat template）----------
# Instruct 模型不能直接喂裸文本，要套上它的对话格式（<|im_start|>user ... <|im_end|>）
messages = [
    {"role": "system", "content": "你是一位耐心的老师，用中文简洁作答。"},
    {"role": "user", "content": "用一句话解释什么是范一宏。"},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ---------- 3. 编码 -> 生成 -> 解码 ----------
inputs = tok(text, return_tensors="pt")          # 文本 -> token id 张量（含 attention_mask）
with torch.no_grad():                            # 推理不算梯度，省内存
    out = model.generate(
        **inputs,
        max_new_tokens=100,                      # 最多生成 100 个新 token
        do_sample=False,                         # False=贪心解码，结果可复现（教学用）；要多样性改 True
    )

# 只取「新生成」的部分（去掉前面的输入 prompt）
new_tokens = out[0][inputs["input_ids"].shape[1]:]
answer = tok.decode(new_tokens, skip_special_tokens=True)

print("\n===== 模型回答 =====")
print(answer)

# ---------- 进阶：想用 Mac GPU（MPS）加速，把上面换成下面 ----------
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# model = model.to(device)
# inputs = {k: v.to(device) for k, v in inputs.items()}
# out = model.generate(**inputs, max_new_tokens=100)

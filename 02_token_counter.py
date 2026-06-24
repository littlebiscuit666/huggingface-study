"""第 2 步：Token 计数工具 —— 输入任意句子，看它被切成几个 token、每个 token 还原成什么。

为什么这个重要（实战 + 面试）：
- API 按 token 收费、模型按 token 算上下文长度（如 32K tokens），不是按字数。
- 中文 token ≠ 字 ≠ 词：常用组合压成 1 个 token，生僻字拆成多个。
- 估上下文/成本时要按 token 数，本工具就是干这个的。

用法：改下面 SENTENCES 列表里的句子，运行看结果。
    python 02_token_counter.py
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"        # 已在缓存，关掉联网检查更快更稳
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# 想测什么句子就加进来（中文、英文、标点都试试）
SENTENCES = [
    "什么是傅里叶变换",
    "什么是傅里叶变换？",
    "我是AIoT算法工程师",
    "床前明月光，疑是地上霜。",
    "Qwen2.5 is a small language model.",
]

print(f"模型：{MODEL_ID}")
print(f"词表大小 vocab_size = {tok.vocab_size}\n")

for s in SENTENCES:
    ids = tok(s, add_special_tokens=False)["input_ids"]   # 文本 -> token id
    toks = [tok.decode([i]) for i in ids]                 # 每个 id 还原成文本
    chars = len(s)
    ratio = len(ids) / chars if chars else 0
    print(f"「{s}」")
    print(f"  字符数={chars}  token数={len(ids)}  (token/字 = {ratio:.2f})")
    print(f"  切分: {toks}\n")

print("规律：常用组合（什么是/变换）压成 1 个 token；生僻字（傅/里/叶）各自 1 个。")
print("经验值：中文 1 字 ≈ 0.6~1.5 token，英文 1 词 ≈ 1~2 token。要精确必须像这样跑代码数。")

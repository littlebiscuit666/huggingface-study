"""第 5 步：对比微调前 / 后 —— 同一个问题，看模型回答变化。

这是验证微调「真有效」的关键一步（也是简历/面试最直观的证据）。
- 微调前：原版 Qwen2.5-0.5B-Instruct（通用风格，可能答不到点上）
- 微调后：加载刚训练的 LoRA adapter，应该带「测→诊→调→验」闭环风格、更专业

关键 API：PeftModel.from_pretrained(base_model, adapter_path)
- base_model 是原始模型（不修改）
- adapter 是 04 存的那 0.1% 小矩阵（几十 MB）
- 加载后 = 原模型 + 微调的增量，这就是 LoRA「即插即用」的优势

运行：
    python 05_compare.py
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "qwen_iot_lora"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DIR = os.path.dirname(os.path.abspath(__file__))

# 用来测试的问题（有些是训练集里的，有些是新问题，看泛化）
QUESTIONS = [
    "AIoT 设备运维的核心方法论是什么？",        # 训练集里直接有
    "边缘设备内存不足导致运行崩溃，怎么优化？",   # 训练集里有
    "传感器数据出现异常尖峰怎么处理？",           # 训练集里有
    "如何排查智能门锁的功耗异常？",               # 新问题（泛化测试）
]


def build_model(adapter_path=None):
    """加载模型；若给 adapter_path 则叠加 LoRA。"""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    if adapter_path:
        # 在原模型上挂载 LoRA adapter（即插即用，不改原权重）
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


def answer(tok, model, q, max_new=120):
    msgs = [{"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    new = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


def main():
    print("加载【微调前】模型……")
    tok, base = build_model(None)
    print("加载【微调后】模型（+ LoRA adapter）……")
    _, tuned = build_model(os.path.join(DIR, ADAPTER_DIR))

    for q in QUESTIONS:
        is_new = "智能门锁" in q
        print("\n" + "=" * 60)
        print(f"问题：{q}" + ("   [新问题-泛化测试]" if is_new else "   [训练集内]"))
        print("-" * 60)
        print(f"【微调前】\n{answer(tok, base, q)}")
        print(f"\n【微调后】\n{answer(tok, tuned, q)}")

    print("\n" + "=" * 60)
    print("观察重点：")
    print("  1. 训练集内问题：微调后应出现「测→诊→调→验」闭环说法、答案更结构化。")
    print("  2. 新问题：看微调是否带来「风格迁移」——即使没背过，也用闭环框架回答。")
    print("  3. 这就是 LoRA 微调的价值：花 0.6GB、15秒，让通用小模型变成领域专家。")


if __name__ == "__main__":
    main()

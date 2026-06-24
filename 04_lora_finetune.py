"""第 4 步：LoRA 微调 —— 把 Qwen2.5-0.5B 训成「IoT 故障问答」小专家。

这是 P3 的核心产出（简历最大亮点）。干的事：
- 读 iot_qa.jsonl（30 条 IoT 问答）
- 只对「答案」部分算 loss（提问部分 mask 掉，这是监督微调 SFT 标准做法）
- 用 LoRA 多轮训练（只动 0.1% 参数）
- 保存 LoRA adapter（几十 MB，不是整个 3GB 模型）

关键概念：label masking
- 输入 = [提问 tokens] + [答案 tokens]
- 我们只希望模型「学会怎么回答」，不希望它学「怎么提问」
- 所以 labels = [-100, -100, ...(提问部分)..., 答案token1, 答案token2, ...]
- -100 是 PyTorch 的「忽略」标记，对应位置不算 loss

运行：
    python 04_lora_finetune.py
产出：./qwen_iot_lora/ 目录（LoRA adapter 权重）
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
import resource
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_FILE = "iot_qa.jsonl"
OUTPUT_DIR = "qwen_iot_lora"
EPOCHS = 6               # 数据少，多跑几轮让它记住风格（小数据上略过拟合没关系，是为演示）
LR = 2e-4               # LoRA 常用学习率，比全量微调高一个量级
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DIR = os.path.dirname(os.path.abspath(__file__))


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def load_data(path):
    """读 jsonl，每条 {question, answer}。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def make_example(tok, q, a):
    """把一条问答编码成 (input_ids, labels)。

    构造：[提问 chat template] + [答案] + [结束符]
    labels：提问部分设 -100（不算 loss），只对答案部分算 loss。
    """
    eos = tok.eos_token_id
    prompt_msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    # 分别编码提问和完整文本，靠长度差定位答案起点
    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    full = prompt + a
    full_ids = tok(full, add_special_tokens=False).input_ids + [eos]
    # labels：提问段全 -100，答案段用真实 id
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # 对齐长度（防止 tokenize 边界差 1）
    labels = labels[:len(full_ids)] + [-100] * (len(full_ids) - len(labels))
    return full_ids, labels


def main():
    print(f"设备: {DEVICE}")
    data = load_data(os.path.join(DIR, DATA_FILE))
    print(f"读入 {len(data)} 条 IoT 问答")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # pad_token 兜底（Qwen 默认没有 pad token，设成 eos 避免报错）
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.train()

    # 预编码全部样本
    encoded = [make_example(tok, d["question"], d["answer"]) for d in data]
    max_len = max(len(ids) for ids, _ in encoded)
    print(f"最长样本 {max_len} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    torch.manual_seed(0)

    base_mem = rss_mb()
    t0 = time.time()
    history = []   # (step, loss)
    step = 0
    for epoch in range(EPOCHS):
        for ids, labels in encoded:
            # batch_size=1，padding 到 max_len（右侧补 pad）
            pad = max_len - len(ids)
            input_ids = torch.tensor([ids + [tok.pad_token_id] * pad], device=DEVICE)
            lab = torch.tensor([labels + [-100] * pad], device=DEVICE)
            attn = torch.tensor([[1] * len(ids) + [0] * pad], device=DEVICE)

            out = model(input_ids=input_ids, attention_mask=attn, labels=lab)
            out.loss.backward()
            opt.step()
            opt.zero_grad()

            history.append((step, out.loss.item()))
            if step % 10 == 0:
                print(f"  epoch {epoch} step {step:3d}  loss={out.loss.item():.3f}")
            step += 1

    dt = time.time() - t0
    peak_mem = rss_mb()

    # 保存 LoRA adapter（只存那 0.1% 的小矩阵，不是整个模型）
    model.save_pretrained(os.path.join(DIR, OUTPUT_DIR))
    tok.save_pretrained(os.path.join(DIR, OUTPUT_DIR))

    # 训练曲线小结：首尾平均 loss 对比
    first5 = sum(l for _, l in history[:5]) / 5
    last5 = sum(l for _, l in history[-5:]) / 5
    print("\n" + "=" * 50)
    print(f"训练样本 {len(data)} × {EPOCHS} 轮 = {step} 步")
    print(f"总耗时 {dt:.1f}s（每步 {dt/max(step,1):.2f}s）")
    print(f"峰值内存 {peak_mem:.0f} MB ({peak_mem/1024:.1f} GB)")
    print(f"loss：前5步均值 {first5:.3f}  →  后5步均值 {last5:.3f}  (下降 {(1-last5/first5)*100:.0f}%)")
    print(f"LoRA adapter 已保存到 ./{OUTPUT_DIR}/")
    print("=" * 50)
    print("loss 明显下降 = 模型在学 IoT 问答风格。下一步用 05 脚本看实际效果对比。")


if __name__ == "__main__":
    main()

"""第 7 步：LoRA 微调 —— 把 Qwen2.5-0.5B 训成「网易云风格乐评」小专家。

这是一个有趣的 NLP 应用：让模型学会网易云音乐评论的文风（感性、共情、
有故事感、常带画面感的文字）。

关键设计：
- 读 music_review.jsonl（标准格式：{messages: [system?, user, assistant]}）
- 只对 assistant 部分算 loss（user/system 部分 mask 掉）
- 用 LoRA 多轮训练（只动 0.1% 参数）
- 保存 LoRA adapter（几十 MB）

网易云乐评风格特点（可以写进你的训练数据）：
1. 画面感：用具体场景/细节触发回忆
2. 共情力：戳中普遍情绪（孤独、成长、怀念）
3. 故事感：有时间/人物/情节的小片段
4. 留白感：不说透，留想象空间
5. 音乐关联：和歌曲/歌手/专辑有呼应

运行：
    python 07_music_review_finetune.py
产出：./qwen_music_lora/ 目录（LoRA adapter 权重）
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
DATA_FILE = "music_review.jsonl"
OUTPUT_DIR = "qwen_music_lora"
EPOCHS = 16               # 乐评需要更细腻的风格学习，适当多跑几轮
LR = 2e-4               # LoRA 常用学习率
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DIR = os.path.dirname(os.path.abspath(__file__))


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def load_data(path):
    """读 jsonl，每条 {messages: [...]} 标准对话格式。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def make_example(tok, messages):
    """把一条对话编码成 (input_ids, labels)。

    标准 SFT 做法：
    - messages = [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
    - 只对 assistant 部分算 loss，其他部分设 -100

    实现思路：
    1. 先构造到 user 为止的 prompt，编码得到 user_part_len
    2. 构造完整对话（含 assistant），编码得到 full_ids
    3. labels = [-100] * user_part_len + full_ids[user_part_len:]
    """
    eos = tok.eos_token_id

    # 验证消息格式
    roles = [m["role"] for m in messages]
    if roles[-1] != "assistant":
        raise ValueError(f"最后一条消息必须是 assistant，实际: {roles[-1]}")

    # 步骤 1：构造到 user 为止的对话（用于计算需要 mask 的长度）
    # 注意：用 add_generation_prompt=False，因为完整对话里已经包含 assistant 的回复
    prompt_msgs = messages[:-1]  # 去掉最后一条 assistant
    prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids

    # 步骤 2：构造完整对话（user + assistant，不带 generation_prompt）
    full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tok(full_text, add_special_tokens=False).input_ids

    # 确保以 eos 结尾
    if full_ids[-1] != eos:
        full_ids = full_ids + [eos]

    # 步骤 3：构造 labels
    # prompt 部分全 -100，assistant 部分用真实 id
    # 注意：prompt_ids 长度可能和 full_ids 的前半部分有细微差异（因为 chat template）
    # 用 safer 的方式：先全设 -100，找到 assistant 开始的位置再填
    labels = [-100] * len(full_ids)

    # 找到 assistant 内容开始的位置（通过比较 prompt_text 和 full_text）
    # 或者更简单：用 prompt_ids 长度作为起点（大部分情况下准确）
    assistant_start = len(prompt_ids)
    if assistant_start < len(full_ids):
        labels[assistant_start:] = full_ids[assistant_start:]

    return full_ids, labels


def main():
    print(f"设备: {DEVICE}")
    data_path = os.path.join(DIR, DATA_FILE)

    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"⚠️  数据文件 {DATA_FILE} 不存在！")
        print("请先创建训练数据，参考格式见 music_review_example.jsonl")
        return

    data = load_data(data_path)
    print(f"读入 {len(data)} 条乐评数据")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj" , "v_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.train()

    # 预编码全部样本
    encoded = [make_example(tok, d["messages"]) for d in data]
    max_len = max(len(ids) for ids, _ in encoded)
    print(f"最长样本 {max_len} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    torch.manual_seed(0)

    base_mem = rss_mb()
    if DEVICE == "mps":
        try:
            torch.mps.reset_peak_memory()
        except:
            pass
    t0 = time.time()
    history = []
    step = 0
    for epoch in range(EPOCHS):
        for ids, labels in encoded:
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
    mps_peak = 0.0
    if DEVICE == "mps":
        try:
            mps_peak = torch.mps.max_memory_allocated() / 1e6
        except:
            pass

    # 保存 LoRA adapter
    model.save_pretrained(os.path.join(DIR, OUTPUT_DIR))
    tok.save_pretrained(os.path.join(DIR, OUTPUT_DIR))

    # 训练曲线小结
    first5 = sum(l for _, l in history[:5]) / 5
    last5 = sum(l for _, l in history[-5:]) / 5
    print("\n" + "=" * 50)
    print(f"训练样本 {len(data)} × {EPOCHS} 轮 = {step} 步")
    print(f"总耗时 {dt:.1f}s（每步 {dt/max(step,1):.2f}s）")
    print(f"峰值内存（进程 RSS）  {peak_mem:.0f} MB ({peak_mem/1024:.1f} GB)")
    if DEVICE == "mps":
        print(f"峰值显存（PyTorch mps） {mps_peak:.0f} MB ({mps_peak/1024:.1f} GB)")
    print(f"loss：前5步均值 {first5:.3f}  →  后5步均值 {last5:.3f}  (下降 {(1-last5/first5)*100:.0f}%)")
    print(f"LoRA adapter 已保存到 ./{OUTPUT_DIR}/")
    print("=" * 50)
    print("运行 08_compare_music.py 看微调前后的乐评对比！")


if __name__ == "__main__":
    main()

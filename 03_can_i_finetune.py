"""第 3 步：实测 —— 我的 Mac 到底能不能微调 Qwen2.5-0.5B？

与其空谈，不如跑一次「真训练的一步」（forward + backward + optimizer.step），
用三个真实数字回答：
  1. 可训参数占全量的多少（LoRA 的威力）
  2. 一步训练耗时
  3. 峰值内存占用（对比你笔记本的内存，判断能不能跑）

为什么这一步重要：证明「微调」在你机器上可行，是 P3（LoRA 微调做 IoT 问答）的前提。

Mac 注意：
- 优先用 MPS（Apple GPU）；若 MPS 反向传播报错，自动回退 CPU 重试（0.5B CPU 也跑得动）。
- 不用 bitsandbytes（那是 CUDA/英伟达专用），Mac 上量化另有方案（如 MLX/GGUF），本步先不量化。

运行：
    python 03_can_i_finetune.py
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import resource
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def rss_mb():
    """进程峰值物理内存（MB）。Mac 的 ru_maxrss 单位是字节。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def run_on(device):
    """在指定设备上加载模型 + 加 LoRA + 跑一步训练，返回统计。"""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

    # ---------- 给注意力的 q_proj / v_proj 加 LoRA（最常见的挂法）----------
    # LoRA 原理：冻结原始权重 W，只学两个小矩阵 A×B（ΔW=A·B）叠加在 W 上。
    config = LoraConfig(
        r=8,                       # LoRA 秩：越小越省，8 是常用值
        lora_alpha=16,            # 缩放系数，经验上设 2×r
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],   # 只给注意力的 q/v 挂 LoRA
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print(f"\n[设备={device}]")
    model.print_trainable_parameters()   # 自动打印：可训参数 / 全量 / 占比

    # ---------- 一个 mini batch（这里先用单条假数据，验证能跑通）----------
    batch = tok("IoT 传感器故障如何排查？", return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["input_ids"]

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    base = rss_mb()

    # ---------- 计时：一步完整训练 ----------
    torch.manual_seed(0)
    t0 = time.time()
    out = model(**batch, labels=labels)   # 前向：算 loss
    out.loss.backward()                   # 反向：算梯度
    opt.step()                            # 更新（只更新 LoRA 那些小矩阵）
    opt.zero_grad()
    dt = time.time() - t0

    peak = rss_mb()
    return {"device": device, "trainable_ratio": None, "step_sec": dt,
            "base_mb": base, "peak_mb": peak, "loss": out.loss.item()}


# ---------- 先试 MPS，失败回退 CPU ----------
device = "mps" if torch.backends.mps.is_available() else "cpu"
try:
    r = run_on(device)
except Exception as e:
    print(f"\n[!] {device} 上跑失败了：{e}")
    print("    自动回退到 CPU 重试……")
    r = run_on("cpu")

# ---------- 结论 ----------
mem = r["peak_mb"]
print("\n" + "=" * 50)
print(f"设备           : {r['device']}")
print(f"一步训练耗时   : {r['step_sec']:.2f} 秒")
print(f"初始内存(RSS)  : {r['base_mb']:.0f} MB")
print(f"峰值内存(RSS)  : {mem:.0f} MB  ({mem/1024:.1f} GB)")
print(f"这一步 loss    : {r['loss']:.3f}")
print("=" * 50)
print("判定：")
if mem < 6000:
    print(f"  ✅ 峰值仅 {mem/1024:.1f}GB，你的笔记本轻松能微调。P3 可以放心做 LoRA 微调。")
elif mem < 10000:
    print(f"  🟡 峰值 {mem/1024:.1f}GB，能跑，但别同时开太多东西。")
else:
    print(f"  🔴 峰值 {mem/1024:.1f}GB 偏高，建议减小 batch / 关掉其他程序。")
print("\n结论：一步训练（前向+反向+更新）成功跑通 = 你的 Mac 能微调 Qwen2.5-0.5B。")
print("下一步（04+）才真正喂 IoT 问答数据做多轮训练。")

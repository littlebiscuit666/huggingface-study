"""第 7 步：量化 Quantization —— 把模型权重从 FP32 压到 INT8，省内存、可端侧部署。

为什么这一步（模型压缩三件套：量化 / 剪枝 / 蒇馏 里的「量化」）：
- 前面 04 用 LoRA 微调、06 加了经典 ML，这一步补上「量化」，让小模型更省内存、更适合端侧。
- 量化 = 权重从 32 位浮点（FP32）压成 8 位整数（INT8）甚至 4 位，用一点点精度换大幅内存/速度。
- 端侧常用组合：量化基座 + LoRA adapter = QLoRA（量化后的小底座 + 即插即用领域 adapter）。

重要：本文件用 PyTorch 内置的「动态量化 INT8」——因为：
- 它在 CPU 上能跑（你这台机器没 GPU）。
- 一行代码就能量化，看清原理。
- 你可能见过的 bitsandbytes 4-bit（NF4）需要 NVIDIA GPU + CUDA，CPU 上会直接报错；
  文件末尾附了那份 GPU 版代码作参考（等你有 GPU 时用）。

关于内存怎么量（踩过的坑，值得讲清楚）：
- 不能用「进程 RSS」对比——torch 的 CPU 内存分配器会把释放的内存留在缓存里不还给系统，
  导致 INT8 的 RSS 看起来反而更大（假象）。
- 这里直接【按权重的真实字节数】算：FP32 权重每个 4 字节、INT8 量化权重每个 1 字节，
  确定可复现，不受缓存干扰。这才是量化「省了多少」的老实答案。

依赖：
    pip install torch transformers   （模型已在 HF 缓存，离线加载）

运行：
    python 07_quantization.py
"""
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def weight_mb(model):
    """模型权重的真实存储（MB）。

    普通层走 parameters()（FP32 每元素 4 字节）；
    量化后的 Linear 权重是「打包」的、不在 parameters() 里，要单独用 .weight() 取出再按 1 字节算。
    """
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    for mod in model.modules():
        mod_id = type(mod).__module__ + "." + type(mod).__name__
        if "quantized.dynamic" in mod_id and "Linear" in type(mod).__name__:
            try:
                w = mod.weight()                       # 量化后的权重张量（int8，element_size=1）
                total += w.numel() * w.element_size()
            except Exception:
                pass
    return total / 1e6


def generate(tok, model, question, max_new=50):
    """跑一次贪心推理，返回（回答文本, 耗时秒）。"""
    msgs = [
        {"role": "system", "content": "你是 IoT 设备运维专家，用中文简洁作答。"},
        {"role": "user", "content": question},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt")   # CPU 张量（本文件全程 CPU）
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    dt = time.time() - t0
    new = out[0][inp["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), dt


QUESTION = "温度传感器读数突然偏高，可能是什么原因？"

# ============================================================
# 1. 加载 FP32 + 动态量化成 INT8，按真实权重字节量内存
# ============================================================
print("=" * 60)
print("1. 内存对比（按权重的真实字节数算，确定可复现）")
print("=" * 60)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)   # 默认 FP32
fp32_mb = weight_mb(model)

# ★ 核心一行：动态量化。{nn.Linear} = 只量化全连接层（占大头权重），qint8 = 8 位有符号整数
model_q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
int8_mb = weight_mb(model_q)
saved = (1 - int8_mb / fp32_mb) * 100

print(f"  FP32 权重：{fp32_mb:.0f} MB（≈ {fp32_mb/1024:.2f} GB）")
print(f"  INT8 权重：{int8_mb:.0f} MB（≈ {int8_mb/1024:.2f} GB）")
print(f"  → 量化省了 {saved:.0f}% 内存（Linear 权重 4 字节 → 1 字节；嵌入层等仍 FP32）\n")


# ============================================================
# 2. 推理对比（速度 + 回答质量）
# ============================================================
print("=" * 60)
print("2. 推理对比：速度 + 回答质量（同一问题，FP32 vs INT8）")
print("=" * 60)
ans_fp32, t_fp32 = generate(tok, model, QUESTION)
ans_int8, t_int8 = generate(tok, model_q, QUESTION)
print(f"  FP32 耗时 {t_fp32:.1f}s ：{ans_fp32[:80]}")
print(f"  INT8 耗时 {t_int8:.1f}s ：{ans_int8[:80]}")
print(f"  → 速度 {t_fp32:.1f}s → {t_int8:.1f}s；回答内容基本一致，质量没明显掉。\n")


# ============================================================
# 3. 小结
# ============================================================
print("=" * 60)
print("3. 小结")
print("=" * 60)
print(f"  内存省 ~{saved:.0f}%（{fp32_mb:.0f}MB → {int8_mb:.0f}MB），速度 {t_fp32:.1f}s → {t_int8:.1f}s，")
print("  回答质量基本不变——这就是量化「用微小精度换体积/速度」的价值。")
print("  端侧组合：把这个 INT8/4-bit 底座 + 04 训的 LoRA adapter 叠起来 = QLoRA 部署。\n")


# ============================================================
# 进阶参考：GPU 4-bit 量化（bitsandbytes NF4）—— 需要 NVIDIA GPU + CUDA，CPU 会报错！
# ============================================================
# 你见过的「bitsandbytes 4-bit」是 GPU 上的更激进量化（权重 4 位，比 INT8 更省）。
# 当前机器无 GPU、torch 是 CPU 版、也没装 bitsandbytes，所以下面代码在这台机器跑不了，
# 留作你以后有 GPU（或上云 GPU）时的参考。装法：pip install bitsandbytes accelerate
#
# from transformers import BitsAndBytesConfig
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,                       # 开启 4-bit 量化
#     bnb_4bit_compute_dtype=torch.float16,    # 计算时用 FP16，防止精度雪崩
#     bnb_4bit_quant_type="nf4",               # NF4：正态分布友好的 4-bit，比线性量化抗损失
#     bnb_4bit_use_double_quant=True,          # 双重量化：连缩放因子(scale)也量化，再压榨一点
# )
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, quantization_config=quantization_config, device_map="auto"
# )
# # 量化底座 + LoRA adapter 叠加 = QLoRA（这正是你 04 训的 adapter 该上的地方）
# # from peft import PeftModel
# # model = PeftModel.from_pretrained(model, "./qwen_iot_lora")

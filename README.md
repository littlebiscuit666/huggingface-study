# HuggingFace-study · P3 端侧大模型

> 目标：从「调 ollama API」升级到「用 transformers 写代码控制模型」，
> 最终做出 **Qwen2.5-0.5B 量化 + LoRA 微调** 的 IoT 故障问答助手。
> 命中 JD：资格3（大模型架构与应用）。


| | ollama | transformers |
|---|---|---|
| 模型格式 | GGUF（压缩格式） | safetensors（HF 原生） |
| 用法 | 调 API，模型当黑盒 | 模型对象在进程里，能改/微调/量化 |
| 能学到的 | 推理调用 | **LoRA 微调、量化、模型内部** ← JD 要的 |
| 你现有的 | ✅ 已会（7 个模型） | ❌ 还没装库 |

> 你 ollama 里的 `qwen2.5:0.5b`（GGUF）和这里下的 `Qwen2.5-0.5B-Instruct`（HF 格式）是**同一个模型的不同打包**，transformers 只认后者。

## 环境（一次性）

```bash
# 1. 装库
pip install transformers torch accelerate

# 2. 模型已下载到 HF 缓存，确认一下：
hf cache ls   # 应看到 model/Qwen/Qwen2.5-0.5B-Instruct 约 1GB
```

## 文件

| 文件 | 学什么 | 状态 |
|---|---|---|
| `01_load_and_chat.py` | transformers 加载本地模型 + chat template + 生成；CPU 推理 | ✅ 可跑 |
| `02_*` | 量化（4-bit/8-bit）测显存/吞吐 | 待写 |
| `03_*` | LoRA 微调做 IoT 故障问答 | 待写 |

## 运行

```bash
cd huggingface-study
python 01_load_and_chat.py
```

## 下一步路线

1. **跑通 01**：确认 transformers + 本地模型 OK。
2. **02 量化**：把 0.5B 压到 4-bit，测内存/速度/质量权衡（对应 P2 的思路，但用在 LLM）。
3. **03 LoRA 微调**：造几十条 IoT 故障问答，让小模型变成「IoT 领域专家」，复用你的 RAG 经验。

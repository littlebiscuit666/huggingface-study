# HuggingFace-study · 端侧大模型 LoRA 微调

> 目标：从「调 ollama API」升级到「用 transformers 写代码控制模型」，
> 最终做出 **Qwen2.5-0.5B + LoRA 微调** 的 IoT 故障问答助手。
> 命中 JD：大模型架构与应用、模型微调与部署。

## 为什么用 transformers（而不是 ollama）

| | ollama | transformers |
|---|---|---|
| 模型格式 | GGUF（压缩格式） | safetensors（HF 原生） |
| 用法 | 调 API，模型当黑盒 | 模型对象在进程里，能改/微调/量化 |
| 能学到的 | 推理调用 | **LoRA 微调、tokenizer、模型内部** ← JD 要的 |

> ollama 里的 `qwen2.5:0.5b`（GGUF）和这里用的 `Qwen/Qwen2.5-0.5B-Instruct`（HF 格式）是**同一个模型的不同打包**，transformers 只认后者。

## 环境（一次性）

```bash
# 1. 装库（peft 做 LoRA 微调；scikit-learn + matplotlib 做 06 的经典 ML 分析）
pip install transformers torch peft scikit-learn matplotlib

# 2. 模型已下载到 HF 缓存，确认一下：
hf cache ls   # 应看到 Qwen/Qwen2.5-0.5B-Instruct 约 1GB
```

> 脚本里都设了 `HF_HUB_OFFLINE=1`，从本地缓存读，国内不联网更稳、加载更快。

## 文件（建议按顺序读 → 跑）

| 文件 | 学什么 | 关键概念 | 状态 |
|---|---|---|---|
| `01_load_and_chat.py` | transformers 加载本地模型 + chat template + CPU 推理 | Instruct 模型必须套对话模板；贪心解码 | ✅ 可跑 |
| `02_token_counter.py` | Token 计数工具：句子怎么被切成 token | 中文 token ≠ 字 ≠ 词；上下文/成本按 token 算 | ✅ 可跑 |
| `03_can_i_finetune.py` | 可行性实测：跑一步真训练 | LoRA 可训参数占比、一步耗时、峰值内存 | ✅ 可跑 |
| `04_lora_finetune.py` | **LoRA 微调**做 IoT 故障问答 | label masking 只对答案算 loss；r=8 / q,v_proj | ✅ 可跑 |
| `05_compare.py` | 微调前/后对比（含新问题泛化测试） | LoRA adapter 即插即用，不改动原权重 | ✅ 可跑 |
| `06_ml_analysis.py` | 经典 ML 分析：K-Means 故障分群 + 线性回归趋势预警 | 无监督聚类 / StandardScaler 标准化 / 趋势外推（ML 预处理层） | ✅ 可跑 |
| `07_quantization.py` | 量化：FP32 → INT8 动态量化 | 权重 4→1 字节省内存；动态量化原理；附 bitsandbytes NF4 GPU 版参考 | ✅ 可跑 |
| `08_gguf_ollama.py` | GGUF：量化模型的存盘+直接加载 | 用 ollama 调本地 GGUF（Q4）；GGUF vs torch 动态量化对比 | ✅ 可跑 |
| `07_music_review_finetune.py` | **LoRA 微调**做网易云风格乐评生成 | 风格迁移、文本生成、情感化写作 | ⏳ 等数据 |
| `08_compare_music.py` | 乐评微调前/后对比 | 泛化能力、风格一致性 | ⏳ 等数据 |

## 关键技术点（简历/面试可讲）

- **LoRA 低秩微调**：冻结原始权重 W，只学两个小矩阵 A×B（ΔW=A·B）叠加在 W 上。本项目只训 **0.1% 参数**（挂 `q_proj`/`v_proj`，r=8，alpha=16），省显存、adapter 只有几十 MB。
- **Label masking（SFT 标准做法）**：输入 = [提问] + [答案]，提问段 labels 设 `-100`（PyTorch 忽略标记，不算 loss）——让模型学「怎么回答」而不是「怎么提问」。
- **Chat template**：Instruct 模型不能喂裸文本，必须套 `<|im_start|>user ... <|im_end|>` 对话格式。
- **Adapter 即插即用**：`PeftModel.from_pretrained(base, adapter)` 在原模型上挂载增量，原权重零修改——上线可热切换、可多 adapter 共存。
- **量化（INT8/4-bit）**：把权重从 FP32 压成 INT8（CPU 动态量化，省约 47% 内存）甚至 4-bit（GPU 上 bitsandbytes NF4），用微小精度换体积/速度；量化底座 + LoRA adapter = **QLoRA** 端侧部署套路。

## 运行

```bash
cd huggingface-study
python 01_load_and_chat.py    # 1. 验证 transformers + 本地模型 OK
python 02_token_counter.py    # 2. 理解 tokenization
python 03_can_i_finetune.py   # 3. 实测能不能微调（打印峰值内存判定）
python 04_lora_finetune.py    # 4. LoRA 微调，产出 ./qwen_iot_lora/ adapter
python 05_compare.py          # 5. 看微调前后回答对比
python 06_ml_analysis.py      # 6. 经典 ML：K-Means 故障分群 + 回归趋势预警
python 07_quantization.py     # 7. 量化：FP32→INT8，省内存（附 GPU NF4 参考）
python 08_gguf_ollama.py      # 8. GGUF：量化模型存盘+直接加载（需 ollama 后台运行）
```

设备：优先 MPS（Apple GPU），不可用自动回退 CPU。Qwen2.5-0.5B 在普通笔记本上全程可跑，无需独显。

## 数据

### IoT 问答数据

`iot_qa.jsonl` —— 30 条 IoT 运维问答，`{question, answer}` 格式，答案统一是「**测 → 诊 → 调 → 验**」闭环风格（先复测确认、再诊断根因、再调整修复、最后验证）。

### 网易云风格乐评数据

`music_review.jsonl` —— 网易云风格乐评数据，`{messages: [...]}` 标准对话格式。

数据字段说明：
- `messages[0]`: system prompt（可选）
- `messages[1]`: user，歌曲 + 背景
- `messages[2]`: assistant，乐评正文（核心训练目标）

网易云乐评风格建议（写训练数据时参考）：
1. **画面感**：用具体场景/细节触发回忆（如"高三晚自习前的广播"、"凌晨三点的客厅"）
2. **共情力**：戳中普遍情绪（孤独、成长、怀念、遗憾）
3. **故事感**：有时间/人物/情节的小片段
4. **留白感**：不说透，留想象空间
5. **音乐关联**：和歌曲/歌手有呼应

示例数据在 `music_review_example.jsonl`，可以参考格式写自己的 `music_review.jsonl`。

## 训练指标（运行 04 自动输出）

`04_lora_finetune.py` 跑完会打印：训练步数、总耗时、**峰值内存**、**loss 前5步均值 → 后5步均值（下降百分比）**。把这几个数填进简历，就是微调「真有效」的量化证据。05 脚本再用「训练集内问题 + 新问题」对比微调前后回答，验证是否学到了领域风格与泛化能力。

## 关联项目

- 配套的**底层原理**：[deep-learning-study](https://github.com/littlebiscuit666/deep-learning-study)（纯 numpy 手写神经元/反向传播/剪枝——理解 LoRA 背后的梯度与压缩思想）

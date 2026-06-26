"""第 8 步：GGUF —— 量化模型的「标准存盘格式」，直接加载、跨平台部署。

为什么这一步（接着 07 量化）：
- 07 的 torch 动态 INT8 是「每次加载 FP32 → 现量化」，存不了盘（save_pretrained 会报错）。
- 真正工业上「量化一次、存成一个文件、以后直接加载」用的是 GGUF 格式：把模型量化
  （常见 Q4 = 4-bit）后存成一个 .gguf 文件，用运行时（ollama / llama.cpp）直接加载，
  不用每次重算量化，跨设备、跨语言都能用。
- 你 ollama 里的模型就全是 GGUF。本文件用 ollama 的 Python API，调本地已存盘的 GGUF
  小模型，体验「量化后存盘、直接加载」的完整闭环——这正是你之前问的「量化后能不能直接加载」。

注意：
- 需要 ollama 桌面端在后台运行（它会在 localhost:11434 提供服务）。
- 这里用的是 qwen3:0.6b（你本地已有的 4-bit GGUF，约 498MB）。它和前面 01~07 用的
  Qwen2.5-0.5B-Instruct（HF 格式）不是同一个模型，但本课重点是 GGUF「存盘+加载」机制，
  用哪个 Qwen 都能说明问题。

依赖：
    pip install ollama    （纯 Python，秒装；ollama 桌面端需另行安装并运行）

运行：
    python 08_gguf_ollama.py
"""
import time
import ollama

MODEL = "qwen3:0.6b"   # 本地已存盘的 4-bit GGUF 小模型（~498MB）
QUESTION = "温度传感器读数突然偏高，可能是什么原因？"


# ============================================================
# 1. 看本地已存盘的 GGUF 模型（量化后存成一个文件，直接可加载）
# ============================================================
print("=" * 60)
print("1. 本地已存盘的 GGUF 模型（ollama 管理，存在 ~/.ollama/models/blobs/）")
print("=" * 60)
target = None
for m in ollama.list().models:
    size_mb = getattr(m, "size", 0) // 1024 // 1024
    mark = "  <-- 本课用这个" if m.model == MODEL else ""
    print(f"  {m.model:24} {size_mb:6} MB{mark}")
    if m.model == MODEL:
        target = m
if target is None:
    raise SystemExit(f"\n没找到 {MODEL}，先在终端跑 `ollama pull {MODEL}` 拉下来再运行本文件")
print(f"\n  → {MODEL} 是个 ~{getattr(target,'size',0)//1024//1024}MB 的量化 GGUF，已存盘、随时直接加载\n")


# ============================================================
# 2. 直接加载这个 GGUF 模型并推理（冷启动会先把 GGUF 读进内存）
# ============================================================
print("=" * 60)
print("2. 直接加载 GGUF 并推理")
print("=" * 60)
t0 = time.time()
r = ollama.chat(
    model=MODEL,
    think=False,   # 关掉 Qwen3 思考模式，直接给可见答案（否则 token 全花在 <think> 里，content 为空）
    messages=[
        {"role": "system", "content": "你是 IoT 设备运维专家，用中文简洁作答。"},
        {"role": "user", "content": QUESTION},
    ],
    options={"num_predict": 150},
)
dt = time.time() - t0
print(f"  耗时 {dt:.1f}s（含冷启动：把 .gguf 从磁盘读进内存）")
print(f"  回答：{r['message']['content'].strip()[:120]}\n")


# ============================================================
# 3. 小结：两条量化路线对比
# ============================================================
print("=" * 60)
print("3. 小结：两条量化路线对比")
print("=" * 60)
print("  torch 动态 INT8（07）：进程内「加载 FP32 → 现量化」，存不了盘，每次重算。")
print(f"  GGUF（本课，{MODEL}）：量化一次 → 存成 .gguf 文件 → 运行时直接加载，跨平台跨语言。")
print("  → 回答你之前的问题：量化后想「存盘、直接加载」，GGUF（ollama / llama.cpp）就是答案；")
print("    你 ollama 里每个模型都是这么干的。GGUF 是端侧部署的事实标准。")

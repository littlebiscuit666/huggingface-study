"""第 8 步：对比乐评微调前/后的效果。

运行：
    python 08_compare_music.py

会输出：
- 微调前（原始 Qwen）的乐评
- 微调后（Qwen + LoRA adapter）的乐评
- 两者并列对比，感受风格变化
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "qwen_music_lora"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DIR = os.path.dirname(os.path.abspath(__file__))


def generate(tok, model, prompt, max_new_tokens=200, temperature=0.8):
    """给定 prompt，生成回复。"""
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return tok.decode(generated, skip_special_tokens=True)


def build_prompt(song, artist, context=""):
    """构建乐评生成 prompt。"""
    if context and context.strip():
        return f"请为歌曲《{song}》（{artist}）写一段网易云风格的乐评。背景：{context}"
    else:
        return f"请为歌曲《{song}》（{artist}）写一段网易云风格的乐评。"


def main():
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # 检查 adapter 是否存在
    adapter_path = os.path.join(DIR, ADAPTER_DIR)
    if not os.path.exists(adapter_path):
        print(f"⚠️  Adapter 目录 {ADAPTER_DIR} 不存在！")
        print("请先运行 07_music_review_finetune.py 进行微调")
        return

    # 测试用例：一半是训练集里的歌曲，一半是新歌曲
    test_cases = [
        ("晴天", "周杰伦", "校园回忆"),
        ("夜曲", "周杰伦", "深夜独处"),
        ("稻香", "周杰伦", "童年回忆"),  # 新歌曲
        ("遇见", "孙燕姿", "地铁邂逅"),
        ("开始懂了", "孙燕姿", "成长与释怀"),  # 新歌曲
        ("海阔天空", "Beyond", "奋斗与坚持"),  # 新歌曲
    ]

    # 加载 base 模型和 tokenizer（只加载一次）
    print("正在加载模型...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    base_model.eval()

    # 加载 LoRA adapter
    lora_model = PeftModel.from_pretrained(base_model, adapter_path).to(DEVICE)
    lora_model.eval()

    print("模型加载完成\n")

    for song, artist, context in test_cases:
        prompt = build_prompt(song, artist, context)

        print("=" * 60)
        print(f"🎵 歌曲：《{song}》- {artist}")
        print(f"📝 背景：{context}")
        print("=" * 60)

        # 微调前（原始模型）
        print("\n📄【微调前（原始 Qwen）】")
        print("-" * 40)
        base_out = generate(tok, base_model, prompt)
        print(base_out)

        # 微调后（LoRA 模型）
        print("\n✨【微调后（网易云风格）】")
        print("-" * 40)
        lora_out = generate(tok, lora_model, prompt)
        print(lora_out)

        print("\n")

    print("=" * 60)
    print("💡 观察点：")
    print("  - 微调后是否更有「画面感」和「故事感」？")
    print("  - 是否更接近网易云乐评的感性风格？")
    print("  - 对没见过的歌曲（泛化），是否也能保持风格？")
    print("=" * 60)


if __name__ == "__main__":
    main()

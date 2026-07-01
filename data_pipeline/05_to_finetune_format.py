"""数据工程 · 第 5 步：转换为微调格式 —— 把精标数据装配成大模型能吃的燃料。

对应岗位职责：*为大模型训练提供高质量的「燃料」*。
清洗 + 质检后的候选，最后要按 00_annotation_guideline.md 第 3 节的 SFT 标准格式装配：
每条 = system（固定人设）+ user（反推的「歌曲+情绪背景」提问）+ assistant（乐评正文）。

这一步做的事：
  1. 读 cleaned_reviews.jsonl（已清洗的高质量候选）
  2. 给每条自动打情绪标签，反推出 user 侧的提问（歌曲 + 背景）
  3. 组装成 {messages:[system, user, assistant]} 标准格式
  4. 输出 music_review_from_pipeline.jsonl —— 可直接喂给 07_music_review_finetune.py

注意：真实流程里 user 提问、assistant 润色由标注员人工完成（见规范）。
这里用规则自动反推，是为了把 pipeline 跑通、形成闭环 demo；
真实交付时这一步的产物需人工过一遍再入库。

运行：
    python 05_to_finetune_format.py
输入：cleaned_reviews.jsonl
产出：music_review_from_pipeline.jsonl（SFT 格式，衔接 07 微调脚本）
"""
import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DIR = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(DIR, "cleaned_reviews.jsonl")
OUT = os.path.join(DIR, "music_review_from_pipeline.jsonl")

SYSTEM = "你是一个擅长写网易云风格乐评的助手，你的文字富有画面感和故事性，能触动人心。"

# 情绪词典（与 04_analyze.py 一致）—— 用于反推 user 提问里的「背景」
EMOTION_LEXICON = {
    "怀念": ["怀念", "想起", "回来", "旧", "曾经", "多年", "从前", "记忆", "照片"],
    "孤独": ["孤独", "一个人", "深夜", "凌晨", "沉默", "安静", "无人"],
    "遗憾": ["遗憾", "错过", "没能", "最后一面", "告别", "分手", "再也", "没说"],
    "成长": ["成长", "长大", "远方", "回家", "释然", "明白", "懂得", "平凡", "坚持"],
    "温暖": ["温暖", "陪", "谢谢", "幸运", "阳光", "喜欢", "牵手"],
    "悲伤": ["眼泪", "流泪", "哭", "泪", "心碎", "疼", "难过"],
}


def infer_background(text, topn=2):
    """从乐评正文反推 2 个情绪关键词作为「背景」。"""
    scores = {emo: sum(1 for w in words if w in text)
              for emo, words in EMOTION_LEXICON.items()}
    ranked = [emo for emo, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0]
    if not ranked:
        return "回忆、感动"
    return "、".join(ranked[:topn])


def main():
    with open(IN, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    out_rows = []
    seen = set()  # 按正文去重，避免模板重复挤进训练集
    for r in rows:
        text = r["clean_text"].strip()
        if text in seen:
            continue
        seen.add(text)
        song = r.get("song", "")
        artist = r.get("artist", "")
        bg = infer_background(text)
        user = f"请为歌曲《{song}》（{artist}）写一段乐评，背景是{bg}。"
        out_rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": text},
            ]
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 已装配 {len(out_rows)} 条 SFT 训练数据 → {os.path.basename(OUT)}")
    print(f"   （输入候选 {len(rows)} 条，正文去重后保留 {len(out_rows)} 条）")
    print("\n示例（第 1 条）：")
    print(json.dumps(out_rows[0], ensure_ascii=False, indent=2))
    print("\n下一步：把它作为 07_music_review_finetune.py 的 DATA_FILE，即可 LoRA 微调验证。")


if __name__ == "__main__":
    main()

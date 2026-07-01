"""数据工程 · 第 2 步：自动清洗 pipeline —— 把「原始脏数据」过一遍规则漏斗。

严格对应 00_annotation_guideline.md 第 2 节「必须打回的 7 条硬规则」和第 4 节「清洗细则」，
把能用规则自动判掉的噪声先清掉，剩下的高质量候选再交给人工标注（宁缺毋滥）。

这一步展示的能力（简历 / 面试）：
- 把「标注规范」翻译成「可复现、可复用的代码规则链」（每条规则独立、可解释）。
- 输出「清洗漏斗」：每一步过滤掉多少、剩多少 —— 数据清洗最直观的量化成果。
- 用 raw 里预埋的 noise_tag，反过来算清洗的「召回率」（真脏数据被清掉的比例）。

清洗顺序（漏斗）：
    去空白规整 → 去重复标点 → 繁转简 → 去广告 → 去跑题
    → 去表情/乱码堆砌 → 去超短无信息 → 去重 → 剩余为高质量候选

运行：
    python 02_clean.py
输入：raw_reviews.jsonl
产出：cleaned_reviews.jsonl（清洗后的高质量候选，待人工标注）
"""
import os
import sys
import re
import json
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DIR = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(DIR, "raw_reviews.jsonl")
OUT = os.path.join(DIR, "cleaned_reviews.jsonl")

# ---------------------------------------------------------------
# 规则词表 / 正则（对应规范第 2 节）
# ---------------------------------------------------------------
# 广告 / 引流关键词
AD_KEYWORDS = [
    "加v", "加V", "微信", "vx", "V信", "刷赞", "刷评论", "日入", "过千",
    "代练", "上分", "出二手", "面交", "非诚勿扰", "关注我", "抖音号",
    "私聊", "有意", "福利", "点我头像", "中介", "出租", "联系电话",
]
AD_PHONE = re.compile(r"1[3-9]\d{9}")  # 手机号

# 灌水 / 抢楼词（整条就是这些）
FLOOR_WORDS = {
    "前排", "沙发", "沙发！", "板凳", "抢个前排", "999+", "顶上去", "留名",
    "路过", "顶", "mark", "早", "有人吗", "第一个评论嘿嘿", "打卡第一天",
}

# 跑题关键词（和音乐/情绪无关的生活闲聊）
OFFTOPIC_KEYWORDS = [
    "螺蛳粉", "几点上课", "花盆", "手机壳", "画的饼", "打完球",
    "借我", "五块钱", "楼上说得对", "怎么都在发广告",
]

# 表情 / 特殊符号（用于统计占比）
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF❤♥♡"
    "♪♫★☆→←♡♥]"
)
# 只保留：中日文、基本标点、字母数字；其余算“杂符号”
NON_TEXT_RE = re.compile(
    r"[^一-鿿぀-ヿa-zA-Z0-9，。！？、；：""''《》…—\s]"
)

# 繁体 → 简体最小映射（真实项目用 opencc，这里手写高频字避免装库）
T2S = str.maketrans({
    "這": "这", "聽": "听", "會": "会", "愛": "爱", "淚": "泪", "歡": "欢",
    "覺": "觉", "萬": "万", "們": "们", "個": "个", "麼": "么", "來": "来",
    "時": "时", "點": "点", "見": "见", "捨": "舍", "麗": "丽", "後": "后",
    "藝": "艺", "夢": "梦", "溫": "温", "陽": "阳", "陰": "阴",
})

# 火星文常见替换
MARS_MAP = {
    "呮媞": "只是", "埘暒": "某些", "倳情": "事情", "dё": "的", "叭叭叭叭": "",
}


# ---------------------------------------------------------------
# 清洗步骤（每个函数只做一件事，返回 (是否保留, 处理后文本, 丢弃原因)）
# ---------------------------------------------------------------
def normalize(text):
    """规整：去首尾/全角空白、去多余换行。"""
    text = text.replace("　", " ").replace("　", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedup_punct(text):
    """去重复标点：，，→，  。。。→…"""
    text = re.sub(r"，{2,}", "，", text)
    text = re.sub(r"。{3,}", "…", text)
    text = re.sub(r"！{2,}", "！", text)
    text = re.sub(r"？{2,}", "？", text)
    return text


def to_simplified(text):
    """繁转简 + 火星文修正。"""
    text = text.translate(T2S)
    for k, v in MARS_MAP.items():
        text = text.replace(k, v)
    return text


def is_ad(text):
    if AD_PHONE.search(text):
        return True
    return any(k in text for k in AD_KEYWORDS)


def is_floor(text):
    t = text.strip().rstrip("！!。.")
    return t in FLOOR_WORDS or t in {w.rstrip("！!") for w in FLOOR_WORDS}


def is_offtopic(text):
    return any(k in text for k in OFFTOPIC_KEYWORDS)


def emoji_junk_ratio(text):
    """表情 + 杂符号占比。占比过高判为乱码/表情堆砌。"""
    if not text:
        return 1.0
    junk = len(EMOJI_RE.findall(text)) + len(NON_TEXT_RE.findall(text))
    return junk / len(text)


def hanzi_count(text):
    """去掉标点后的纯汉字数（衡量信息量）。"""
    return len(re.findall(r"[一-鿿]", text))


def clean_symbols(text):
    """删除表情和杂符号（在判定保留后，做最终净化）。"""
    text = EMOJI_RE.sub("", text)
    text = NON_TEXT_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------
# 主流程：漏斗式过滤，逐级统计
# ---------------------------------------------------------------
def main():
    with open(IN, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    total = len(rows)
    funnel = []            # (步骤名, 过滤后剩余条数)
    dropped_reason = Counter()   # 丢弃原因统计
    dropped_tags = Counter()     # 被丢弃的真实 noise_tag（算召回）

    kept = []
    seen_texts = set()     # 去重用

    for r in rows:
        text = r["raw_text"]

        # 1) 规整 + 去重标点 + 繁转简（先净化格式，再判定）
        text = to_simplified(dedup_punct(normalize(text)))

        # 2) 硬规则过滤（对应规范第 2 节）
        if is_ad(text):
            dropped_reason["广告引流"] += 1
            dropped_tags[r["noise_tag"]] += 1
            continue
        if is_offtopic(text):
            dropped_reason["无关跑题"] += 1
            dropped_tags[r["noise_tag"]] += 1
            continue
        if is_floor(text):
            dropped_reason["灌水抢楼"] += 1
            dropped_tags[r["noise_tag"]] += 1
            continue
        if emoji_junk_ratio(text) > 0.3:
            dropped_reason["表情/乱码堆砌"] += 1
            dropped_tags[r["noise_tag"]] += 1
            continue

        # 净化符号后再判信息量
        text = clean_symbols(text)
        if hanzi_count(text) < 8:
            dropped_reason["超短无信息"] += 1
            dropped_tags[r["noise_tag"]] += 1
            continue

        # 3) 去重（精确去重：净化后完全相同的只留第一条）
        if text in seen_texts:
            dropped_reason["重复内容"] += 1
            dropped_tags[r["noise_tag"]] += 1
            continue
        seen_texts.add(text)

        # 保留：写回净化后的文本
        r["clean_text"] = text
        kept.append(r)

    # 写出
    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -------------------- 打印清洗漏斗 --------------------
    print("=" * 56)
    print("清洗漏斗（Cleaning Funnel）")
    print("=" * 56)
    print(f"  原始数据                {total:5d} 条")
    running = total
    for reason, cnt in dropped_reason.most_common():
        running -= cnt
        print(f"  ├─ 剔除[{reason:12s}] -{cnt:4d}  →  剩 {running:5d} 条")
    print(f"  最终保留（高质量候选）   {len(kept):5d} 条  "
          f"（保留率 {len(kept)/total*100:.1f}%）")

    # -------------------- 清洗效果验证（用预埋标签算召回）--------------------
    print("\n" + "=" * 56)
    print("清洗效果验证（真实 noise_tag 视角）")
    print("=" * 56)
    # 统计原始里各 tag 的总数
    orig_tags = Counter(r["noise_tag"] for r in rows)
    kept_tags = Counter(r["noise_tag"] for r in kept)
    print(f"  {'噪声类型':12s}{'原始':>6}{'保留':>6}{'清掉率':>8}")
    for tag in orig_tags:
        o = orig_tags[tag]
        k = kept_tags.get(tag, 0)
        rate = (o - k) / o * 100
        print(f"  {tag:12s}{o:>6}{k:>6}{rate:>7.0f}%")
    good_kept = kept_tags.get("good", 0)
    print(f"\n✅ 优质(good)保留 {good_kept}/{orig_tags['good']}，"
          f"噪声清除率整体 {(1 - len(kept)/total)*100:.0f}% → {os.path.basename(OUT)}")
    print("   （剩余候选仍需人工按规范打分精标，见 00_annotation_guideline.md）")


if __name__ == "__main__":
    main()

"""数据工程 · 第 4 步：数据分析 —— 挖掘乐评数据的潜在价值。

对应岗位职责：*协助进行数据清洗、整理和分析工作，挖掘数据中的潜在价值*。
清洗完不是终点。数据分析能回答：这批数据覆盖了哪些情绪 / 主题？分布均不均衡？
有没有哪一类严重不足需要补标？—— 这些洞察直接指导「下一轮补什么数据」。

本步做三件分析（不引入 jieba/wordcloud，仅用 sklearn + matplotlib，开箱即用）：
  A. 情绪分布统计：用情绪词典给每条乐评打情绪标签，看六大情绪的分布是否均衡
  B. TF-IDF + K-Means 主题聚类：字符 n-gram 特征，把乐评自动聚成几个主题簇
  C. 高频关键词：统计高频二字词，反映这批数据的核心意象（雨/深夜/告别…）

依赖：pip install scikit-learn matplotlib（06 步已装）
运行：
    python 04_analyze.py
输入：cleaned_reviews.jsonl（也可分析 music_review.jsonl）
产出：emotion_dist.png（情绪分布）、topic_clusters.png（主题聚类）
"""
import os
import sys
import re
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(DIR, "cleaned_reviews.jsonl")

# 情绪词典（简易版，真实项目可换成情感模型）
EMOTION_LEXICON = {
    "怀念": ["怀念", "想起", "回来", "旧", "曾经", "多年", "从前", "记忆", "照片"],
    "孤独": ["孤独", "一个人", "深夜", "凌晨", "沉默", "空", "安静", "无人"],
    "遗憾": ["遗憾", "错过", "没能", "最后一面", "告别", "分手", "再也", "说不出口", "没说"],
    "成长": ["成长", "长大", "远方", "回家", "释然", "明白", "懂得", "平凡", "坚持"],
    "温暖": ["温暖", "陪", "谢谢", "幸运", "阳光", "喜欢", "爱", "牵手"],
    "悲伤": ["眼泪", "流泪", "哭", "泪", "心碎", "疼", "难过"],
}


def load_texts(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "clean_text" in obj:
                items.append(obj["clean_text"])
            elif "messages" in obj:
                a = [m for m in obj["messages"] if m["role"] == "assistant"]
                if a:
                    items.append(a[-1]["content"])
            elif "raw_text" in obj:
                items.append(obj["raw_text"])
    return items


def tag_emotion(text):
    """给一条文本打情绪标签：命中词最多的情绪；都没命中记 '其他'。"""
    scores = {emo: sum(1 for w in words if w in text)
              for emo, words in EMOTION_LEXICON.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "其他"


def analyze_emotion(texts):
    """A. 情绪分布。"""
    dist = Counter(tag_emotion(t) for t in texts)
    print("=" * 50)
    print("A. 情绪分布")
    print("=" * 50)
    for emo, cnt in dist.most_common():
        bar = "█" * int(cnt / max(1, len(texts)) * 50)
        print(f"  {emo:6s}{cnt:5d}  {bar}")

    # 画柱状图
    labels = [e for e, _ in dist.most_common()]
    values = [dist[e] for e in labels]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values, color="#c62828")
    plt.title("乐评数据 · 情绪分布")
    plt.ylabel("条数")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center")
    out = os.path.join(DIR, "emotion_dist.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  → 图已保存 emotion_dist.png")
    # 给出可执行洞察
    least = dist.most_common()[-1]
    print(f"  💡 洞察：'{least[0]}'类最少（{least[1]}条），下一轮补标可优先补这类，均衡分布。")
    return dist


def analyze_topics(texts, k=4):
    """B. TF-IDF + K-Means 主题聚类（字符 2-gram，免分词）。"""
    print("\n" + "=" * 50)
    print(f"B. 主题聚类（K-Means, k={k}）")
    print("=" * 50)
    if len(texts) < k:
        print("  样本太少，跳过聚类。")
        return
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(X)

    # 每簇的代表词（该簇质心上权重最高的 n-gram）
    terms = np.array(vec.get_feature_names_out())
    for c in range(k):
        centroid = km.cluster_centers_[c]
        top = terms[centroid.argsort()[::-1][:8]]
        top = [t.strip() for t in top if t.strip()]
        cnt = int((labels == c).sum())
        print(f"  簇{c}（{cnt}条）关键词: {' '.join(top[:6])}")

    # 用 SVD 降到 2 维可视化
    svd = TruncatedSVD(n_components=2, random_state=0)
    pts = svd.fit_transform(X)
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="tab10", s=18, alpha=0.7)
    plt.title(f"乐评主题聚类（K-Means k={k}, TF-IDF+SVD降维）")
    plt.xlabel("SVD-1")
    plt.ylabel("SVD-2")
    plt.legend(*scatter.legend_elements(), title="簇", loc="best")
    out = os.path.join(DIR, "topic_clusters.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  → 图已保存 topic_clusters.png")


def analyze_keywords(texts, topn=15):
    """C. 高频二字词（简易 bigram 频次，反映核心意象）。"""
    print("\n" + "=" * 50)
    print("C. 高频关键词（二字词）")
    print("=" * 50)
    stop = set("的了是我你他她在也就都很和与而又把被着过吗呢啊吧这那有没")
    counter = Counter()
    for t in texts:
        hanzi = re.sub(r"[^一-鿿]", "", t)
        for i in range(len(hanzi) - 1):
            bg = hanzi[i:i + 2]
            if bg[0] not in stop and bg[1] not in stop:
                counter[bg] += 1
    for word, cnt in counter.most_common(topn):
        print(f"  {word}  ×{cnt}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    if not os.path.isabs(path):
        path = os.path.join(DIR, path)
    texts = load_texts(path)
    print(f"分析文件：{os.path.basename(path)}，共 {len(texts)} 条\n")
    analyze_emotion(texts)
    analyze_topics(texts, k=4)
    analyze_keywords(texts)
    print("\n✅ 分析完成。情绪分布 + 主题聚类图可直接放进简历/报告。")


if __name__ == "__main__":
    main()

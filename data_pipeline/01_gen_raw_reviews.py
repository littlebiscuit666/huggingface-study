"""数据工程 · 第 1 步：生成「原始脏数据」—— 模拟从网易云热评爬下来的真实语料。

为什么要造脏数据（简历 / 面试要能讲）：
- 真实的数据标注 / 清洗岗位，拿到的从来不是干净成品，而是充满噪声的原始语料。
- 只有先有「脏数据」，才能展示「清洗 → 标注 → 质检」这条真正值钱的能力。
- 这里刻意按真实网易云热评的噪声分布造数据：灌水、超短、表情乱码、火星文、
  繁体、错别字、广告引流、无关闲聊 …… 再把少量真正优质的乐评埋进去等待被筛出。

输出：raw_reviews.jsonl —— 每条一个原始评论对象，字段贴近真实爬虫产物：
    {
      "id":        评论唯一 id,
      "song":      歌曲名,
      "artist":    歌手,
      "raw_text":  原始评论文本（含各种噪声）,
      "likes":     点赞数,
      "user":      用户昵称,
      "noise_tag": 这条数据真实的噪声类型（仅用于后续验证清洗效果，真实场景没有这一列）
    }

运行：
    python 01_gen_raw_reviews.py
产出：raw_reviews.jsonl（约 1800 条）
"""
import os
import sys
import json
import random

# Windows 终端默认 GBK，输出 emoji / 特殊符号会崩，统一切成 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 固定随机种子 —— 数据可复现（面试常被问：你的流程能复现吗？能。）
random.seed(42)

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "raw_reviews.jsonl")
TARGET = 1800  # 目标条数

# ---------------------------------------------------------------
# 歌曲池（歌名, 歌手, 适合的情绪背景）—— 优质乐评会围绕这些背景写
# ---------------------------------------------------------------
SONGS = [
    ("晴天", "周杰伦", "校园青春、暗恋"),
    ("夜曲", "周杰伦", "深夜、孤独、怀念"),
    ("遇见", "孙燕姿", "地铁、缘分、错过"),
    ("后来", "刘若英", "初恋、遗憾、成长"),
    ("小幸运", "田馥甄", "毕业、感恩、青春"),
    ("光年之外", "邓紫棋", "宇宙、坚定的爱"),
    ("起风了", "买辣椒也用券", "漂泊、归来、成长"),
    ("平凡之路", "朴树", "迷茫、坚持、释然"),
    ("说散就散", "袁娅维", "分手、洒脱、不舍"),
    ("体面", "于文文", "分手、尊严、告别"),
    ("董小姐", "宋冬野", "文艺、漂泊、遗憾"),
    ("成都", "赵雷", "离别、城市、温柔"),
    ("南山南", "马頔", "远方、思念、孤独"),
    ("演员", "薛之谦", "假装、心碎、释怀"),
    ("模特", "李荣浩", "都市、疲惫、自嘲"),
]

USERS = [
    "云村村民", "听风的人", "匿名用户", "深夜网抑云", "路过的猫",
    "一只柠檬", "橘子汽水", "北岛的海", "沉默是金", "夏日限定",
    "user_88231", "★彡", "小K", "AAA房产中介小王", "游客7788",
]

# ===============================================================
# 各类「脏」文本生成器 —— 每类对应真实热评里的一种噪声
# ===============================================================

# 1) 灌水 / 抢楼：占楼党，毫无内容
SPAM_FLOOR = [
    "前排", "沙发！", "板凳", "第一个评论嘿嘿", "抢个前排",
    "999+", "顶上去", "留名", "路过", "打卡第一天",
    "2024来听的扣1", "有人吗", "顶", "mark", "早",
]

# 2) 超短无信息量：情绪词但没内容
TOO_SHORT = [
    "好听", "爱了", "绝了", "泪目", "❤", "yyds", "神曲",
    "循环", "单曲循环中", "百听不厌", "awsl", "!!!", "。。。", "😭",
]

# 3) 表情 / 乱码 / 特殊符号堆砌
EMOJI_JUNK = [
    "😭😭😭😭😭😭😭😭", "❤️❤️❤️❤️❤️", "🎵🎵🎵🎶🎶", "[大哭][大哭][大哭]",
    "( >﹏< )", "T_T T_T T_T", "555555555", "→_→ ←_←",
    "♡♡♡♡♡♡♡♡♡", "【破涕为笑】【破涕为笑】", "🥀🥀",  # 半个 surrogate 制造乱码
]

# 4) 火星文 / 繁体 / 错别字
WEIRD = [
    "這首歌真dё好聽啊嗚嗚嗚", "呮媞想埘暒⒈些倳情",  # 火星文
    "這是我聽過最美的旋律，每次聽都會流淚",  # 繁体（需转简）
    "聽了整整一個晚上都捨不得停",  # 繁体
    "这首歌真的好听的不要不要的", "泪奔了都", "听着听着就哭了呜呜呜呜呜呜呜",
    "太好听了叭叭叭叭", "耳朵怀孕了", "DNA动了",
]

# 5) 广告 / 引流 / 无关（必须清掉）
ADS = [
    "刷赞刷评论加V信：xxx8888 全网最低价",
    "招收学徒，日入过千，有意私聊",
    "点我头像有惊喜，福利多多",
    "出二手吉他，九成新，同城可面交，V：guitar666",
    "本人房产中介，城西两室一厅出租，非诚勿扰联系电话13800000000",
    "关注我抖音号：music_life，每天分享好歌",
    "代练上分，各种游戏都接，便宜靠谱",
]

# 6) 无关闲聊（跑题，和音乐 / 情绪无关）
OFF_TOPIC = [
    "今天中午吃的螺蛳粉太香了", "有没有人知道明天几点上课",
    "我家猫又把花盆打翻了", "求问这个手机壳哪买的",
    "老板画的饼又大又圆", "刚打完球累死了", "谁能借我五块钱吃饭",
    "这歌评论区怎么都在发广告", "楼上说得对",
]

# 7) 优质乐评「原料」（画面感 + 情绪 + 音乐关联）—— 埋进脏数据里，等待被标注筛出
GOOD_TEMPLATES = [
    "{bg}的时候单曲循环这首《{song}》，{scene}，眼泪就掉下来了。",
    "高三那年最爱《{song}》，{scene}。现在再听，恍如隔世。",
    "《{song}》前奏一响，{scene}，那些以为忘掉的人和事全都回来了。",
    "深夜戴耳机听《{song}》，{scene}。有些话终究没能说出口。",
    "{scene}。谢谢{artist}的《{song}》，陪我走过那段最难的日子。",
]
SCENES = [
    "想起那个再也见不到的人", "窗外的雨下了一整夜", "地铁上偷偷红了眼眶",
    "翻到高中的旧照片", "一个人走在回家的老街", "路灯把影子拉得很长",
    "手机里还留着没发出去的消息", "站台的风把票根吹走了",
]

# 8) 带前后空白 / 全角空格 / 换行的脏格式（内容其实还行，考清洗的“规整”能力）
def messy_format(text):
    """给一段文本随机加上真实脏格式：首尾空格、全角空格、多余换行、重复标点。"""
    if random.random() < 0.5:
        text = "  " + text + "   "
    if random.random() < 0.3:
        text = text.replace("，", "，，")  # 重复标点
    if random.random() < 0.3:
        text = "　　" + text  # 全角空格缩进
    if random.random() < 0.2:
        text = text + "\n\n"
    return text


def make_good_review(song, artist, bg):
    """生成一条优质乐评原料。"""
    tpl = random.choice(GOOD_TEMPLATES)
    return tpl.format(song=song, artist=artist, bg=bg, scene=random.choice(SCENES))


# ===============================================================
# 按真实分布混合各类噪声，生成整个数据集
# ===============================================================
# 各噪声类型的目标占比（模拟真实热评区：优质内容其实是少数）
DISTRIBUTION = [
    ("good", 0.20),        # 优质乐评（真正想要的燃料）
    ("spam_floor", 0.14),  # 抢楼灌水
    ("too_short", 0.16),   # 超短无信息
    ("emoji_junk", 0.10),  # 表情乱码
    ("weird", 0.12),       # 火星文/繁体/错别字
    ("ads", 0.08),         # 广告引流
    ("off_topic", 0.08),   # 无关闲聊
    ("dup", 0.12),         # 重复（复制粘贴的爆款评论）
]


def gen_one(noise_type, song, artist, bg, dup_pool):
    """根据噪声类型生成一条 raw_text，返回 (text, noise_tag)。"""
    if noise_type == "good":
        return messy_format(make_good_review(song, artist, bg)), "good"
    if noise_type == "spam_floor":
        return random.choice(SPAM_FLOOR), "spam_floor"
    if noise_type == "too_short":
        return random.choice(TOO_SHORT), "too_short"
    if noise_type == "emoji_junk":
        return random.choice(EMOJI_JUNK), "emoji_junk"
    if noise_type == "weird":
        return messy_format(random.choice(WEIRD)), "weird"
    if noise_type == "ads":
        return random.choice(ADS), "ads"
    if noise_type == "off_topic":
        return random.choice(OFF_TOPIC), "off_topic"
    if noise_type == "dup":
        # 重复：从已生成的“爆款”里复制一条（模拟大量用户复制粘贴同一句）
        if dup_pool:
            return random.choice(dup_pool), "dup"
        return random.choice(TOO_SHORT), "too_short"
    return "好听", "too_short"


def main():
    # 展开分布为一个类型序列
    types = []
    for name, ratio in DISTRIBUTION:
        types += [name] * int(TARGET * ratio)
    # 补齐到 TARGET
    while len(types) < TARGET:
        types.append("good")
    random.shuffle(types)

    # “爆款”池：几条会被大量复制的评论，用来制造重复
    dup_pool = [
        "网抑云时间到，一起emo",
        "每个人心里都有一首这样的歌",
        "多年以后我还是会为这首歌流泪",
    ]

    rows = []
    for i, noise_type in enumerate(types):
        song, artist, bg = random.choice(SONGS)
        text, tag = gen_one(noise_type, song, artist, bg, dup_pool)
        rows.append({
            "id": f"r{i:05d}",
            "song": song,
            "artist": artist,
            "raw_text": text,
            "likes": random.choice([0, 0, 1, 2, 5, 12, 88, 233, 1024, 9999]),
            "user": random.choice(USERS),
            "noise_tag": tag,  # 真实场景没有这列，这里留作后续验证清洗召回率
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印分布，让人一眼看清“脏数据长什么样”
    from collections import Counter
    dist = Counter(r["noise_tag"] for r in rows)
    print(f"✅ 已生成 {len(rows)} 条原始脏数据 → {os.path.basename(OUT)}")
    print("\n噪声类型分布（模拟真实热评区）：")
    for tag, cnt in dist.most_common():
        bar = "█" * int(cnt / 10)
        print(f"  {tag:12s} {cnt:5d}  {bar}")
    print(f"\n其中真正优质（good）仅 {dist['good']} 条 —— 这就是「清洗+标注」要从噪声里捞出的燃料。")


if __name__ == "__main__":
    main()

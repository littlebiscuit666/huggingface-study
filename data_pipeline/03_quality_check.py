"""数据工程 · 第 3 步：质量检查（QC）—— 给清洗后的候选数据做体检。

对应岗位职责：*确保标注数据的质量和准确性*、*及时反馈标注过程中发现的问题*。
清洗规则不可能 100% 干净，QC 这一层负责「抽检 + 报告 + 揪出漏网样本」，
是保证数据质量的最后一道关口，也是标注流程里最能体现「严谨」的一环。

检查项（对应规范第 5 节质检要求）：
    A. 字段完整性       —— 每条是否都有必需字段、有没有空文本
    B. 长度分布         —— 太短（信息不足）/ 太长（可能跑题灌水）的比例
    C. 重复率           —— 精确重复 + 高相似（前缀）重复
    D. 疑似漏网噪声      —— 用规范里的关键词二次扫描，揪出清洗没抓到的
    E. 生成质检报告      —— 打印汇总 + 导出不合格样本清单 qc_report.json

运行：
    python 03_quality_check.py                # 默认检查 cleaned_reviews.jsonl
    python 03_quality_check.py music_review.jsonl   # 也可检查最终训练集（自动识别格式）
产出：qc_report.json（质检报告 + 不合格样本清单）
"""
import os
import sys
import re
import json
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(DIR, "cleaned_reviews.jsonl")
REPORT = os.path.join(DIR, "qc_report.json")

# 质量阈值（可按规范调整）
MIN_HANZI = 8      # 少于这么多汉字 → 信息量不足
MAX_LEN = 140      # 超过这么长 → 可能是灌水/跑题
DUP_PREFIX = 12    # 前 N 字相同视为高相似重复

# 疑似漏网噪声关键词（QC 二次扫描，比清洗更宽松，宁可误报也别漏）
SUSPECT = ["加v", "微信", "vx", "刷赞", "私聊", "抖音", "代练",
           "螺蛳粉", "上课", "花盆", "借我"]


def hanzi_count(text):
    return len(re.findall(r"[一-鿿]", text))


def load_texts(path):
    """兼容两种格式：
       - cleaned_reviews.jsonl：取 clean_text
       - music_review.jsonl（SFT格式）：取 messages[-1].content（assistant 正文）
    """
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "clean_text" in obj:
                items.append({"id": obj.get("id", ""), "text": obj["clean_text"]})
            elif "messages" in obj:
                assistant = [m for m in obj["messages"] if m["role"] == "assistant"]
                text = assistant[-1]["content"] if assistant else ""
                items.append({"id": obj.get("id", ""), "text": text})
            elif "raw_text" in obj:
                items.append({"id": obj.get("id", ""), "text": obj["raw_text"]})
    return items


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    if not os.path.isabs(path):
        path = os.path.join(DIR, path)
    items = load_texts(path)
    total = len(items)

    report = {"file": os.path.basename(path), "total": total, "issues": {}}
    bad_samples = []  # 不合格样本清单

    # A. 字段完整性 —— 空文本
    empty = [it for it in items if not it["text"].strip()]
    report["issues"]["空文本"] = len(empty)

    # B. 长度分布
    lengths = [hanzi_count(it["text"]) for it in items]
    too_short = [it for it in items if hanzi_count(it["text"]) < MIN_HANZI]
    too_long = [it for it in items if len(it["text"]) > MAX_LEN]
    report["issues"]["过短(<%d汉字)" % MIN_HANZI] = len(too_short)
    report["issues"]["过长(>%d字)" % MAX_LEN] = len(too_long)
    report["length_stat"] = {
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "avg": round(sum(lengths) / total, 1) if total else 0,
    }

    # C. 重复率
    text_counter = Counter(it["text"] for it in items)
    exact_dup = sum(c - 1 for c in text_counter.values() if c > 1)
    prefix_counter = Counter(it["text"][:DUP_PREFIX] for it in items if it["text"])
    prefix_dup = sum(c - 1 for c in prefix_counter.values() if c > 1)
    report["issues"]["精确重复"] = exact_dup
    report["issues"]["高相似重复(前%d字)" % DUP_PREFIX] = prefix_dup
    report["dup_rate"] = round(exact_dup / total * 100, 1) if total else 0

    # D. 疑似漏网噪声
    suspect_hits = []
    for it in items:
        hit = [kw for kw in SUSPECT if kw in it["text"].lower()]
        if hit:
            suspect_hits.append({"id": it["id"], "text": it["text"], "hit": hit})
    report["issues"]["疑似漏网噪声"] = len(suspect_hits)

    # 汇总不合格样本（去重）
    seen = set()
    for tag, group in [("过短", too_short), ("过长", too_long)]:
        for it in group:
            key = (it["id"], tag)
            if key not in seen:
                seen.add(key)
                bad_samples.append({"id": it["id"], "reason": tag, "text": it["text"][:50]})
    for s in suspect_hits:
        bad_samples.append({"id": s["id"], "reason": "疑似噪声:" + ",".join(s["hit"]),
                            "text": s["text"][:50]})

    report["bad_sample_count"] = len(bad_samples)
    report["bad_samples"] = bad_samples[:50]  # 报告里最多存 50 条示例

    # 通过率
    problem_ids = {b["id"] for b in bad_samples} | {it["id"] for it in empty}
    passed = total - len(problem_ids)
    report["pass_rate"] = round(passed / total * 100, 1) if total else 0

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # -------------------- 打印报告 --------------------
    print("=" * 56)
    print(f"质检报告 QC · {report['file']}（共 {total} 条）")
    print("=" * 56)
    print("检查项                        问题数")
    print("-" * 40)
    for k, v in report["issues"].items():
        flag = "⚠️" if v > 0 else "✅"
        print(f"  {k:24s}{v:6d}  {flag}")
    print("-" * 40)
    ls = report["length_stat"]
    print(f"  长度(汉字)  min={ls['min']}  avg={ls['avg']}  max={ls['max']}")
    print(f"  重复率      {report['dup_rate']}%")
    print(f"\n  合格率      {report['pass_rate']}%  "
          f"（{passed}/{total} 条通过，{len(problem_ids)} 条需复核）")

    if bad_samples:
        print(f"\n需人工复核的样本（前 5 条，完整见 qc_report.json）：")
        for b in bad_samples[:5]:
            print(f"    [{b['reason']}] {b['text']}")
    print(f"\n📄 完整报告已导出 → {os.path.basename(REPORT)}")


if __name__ == "__main__":
    main()

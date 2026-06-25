"""第 6 步：经典机器学习分析 —— 给 IoT 助手叠一层 ML（无监督分群 + 趋势预测）。

为什么加这一层（简历 / 面试要能讲）：
- 前面 01~05 都是「大模型微调」，但「机器学习」这块一直缺。
- 真实 IoT 场景里，海量传感器数据不可能全丢给 LLM；先用经典 ML 做无监督分群
  （哪些设备已经在异常）和趋势预测（读数会不会越界），再交给微调后的小模型
  解释原因、给处置方案 —— 这才是「ML 预处理 + LLM 决策」的工程闭环。

本步做两件经典 ML（都用 scikit-learn）：
  A. K-Means 无监督故障分群：给一堆设备的多维遥测（温度 / 振动 / 电流 / 功耗），
     不告诉它谁有故障，看它能不能把异常设备自动聚成一类（再对比真实标签算准确率）。
  B. 线性回归趋势预测：对某传感器读数的时间序列拟合趋势线，
     外推预测未来会不会越过预警阈值（趋势预警）。

依赖：
    pip install scikit-learn numpy matplotlib

运行：
    python 06_ml_analysis.py
产物：kmeans_clusters.png、regression_forecast.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面环境也能存图
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Windows / Mac 中文字体兼容
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(0)
DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# A. K-Means 无监督故障分群
# ============================================================
print("=" * 60)
print("A. K-Means 无监督故障分群")
print("=" * 60)

N_NORMAL, N_FAULTY = 140, 60
# 正常设备：4 维遥测都在正常区间
normal = np.random.normal(
    loc=[55, 2.0, 1.2, 18], scale=[5, 0.3, 0.1, 2], size=(N_NORMAL, 4)
)
# 故障设备：温度 / 振动 / 电流 / 功耗 整体偏高
faulty = np.random.normal(
    loc=[85, 5.5, 2.3, 38], scale=[6, 0.5, 0.2, 4], size=(N_FAULTY, 4)
)
X = np.vstack([normal, faulty])
y_true = np.array([0] * N_NORMAL + [1] * N_FAULTY)   # 真实标签（K-Means 看不见）
feat = ["温度(℃)", "振动(mm/s)", "电流(A)", "功耗(W)"]

# K-Means 对量纲敏感（功耗几十、电流个位），必须先标准化
Xs = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=2, n_init=10, random_state=0)
labels = km.fit_predict(Xs)

# 簇标签 0/1 是任意分配的，用「与真实标签的最大一致率」评估
acc = max((labels == y_true).mean(), (labels == 1 - y_true).mean())
print(f"  设备数 {len(X)}（正常 {N_NORMAL} / 故障 {N_FAULTY}），特征 {feat}")
print(f"  K-Means(k=2) 分群与真实故障标签一致率：{acc*100:.1f}%")
print("  → 无监督也能把异常设备分开，说明遥测特征有区分度。")

# 可视化：取区分度最直观的两维（温度 vs 振动）画散点
plt.figure(figsize=(7, 5))
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s=18, alpha=0.7, label="簇0")
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s=18, alpha=0.7, label="簇1")
plt.xlabel(feat[0]); plt.ylabel(feat[1])
plt.title(f"K-Means 无监督故障分群（与真实标签一致率 {acc*100:.1f}%）")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(DIR, "kmeans_clusters.png"), dpi=110)
print("  图已保存 -> kmeans_clusters.png\n")


# ============================================================
# B. 线性回归趋势预测 + 越界预警
# ============================================================
print("=" * 60)
print("B. 线性回归趋势预测 + 越界预警")
print("=" * 60)

# 一个传感器的时序读数：温度缓慢上升（潜在故障趋势）+ 噪声
T_HIST = 60
t = np.arange(T_HIST).reshape(-1, 1)
temp = 0.50 * t.ravel() + 40 + np.random.normal(0, 1.0, T_HIST)

reg = LinearRegression().fit(t, temp)
r2 = reg.score(t, temp)

# 外推预测未来 20 个采样点
T_FUTURE = 20
t_future = np.arange(T_HIST, T_HIST + T_FUTURE).reshape(-1, 1)
temp_pred = reg.predict(t_future)

THRESHOLD = 70.0
cross_idx = np.where(temp_pred > THRESHOLD)[0]
will_cross = len(cross_idx) > 0
cross_at = T_HIST + int(cross_idx[0]) if will_cross else None

print(f"  历史样本 {T_HIST} 点，趋势斜率 {reg.coef_[0]:.3f} ℃/采样，拟合优度 R^2={r2:.3f}")
print(f"  预测未来 {T_FUTURE} 点，末点温度 {temp_pred[-1]:.1f}℃（预警阈值 {THRESHOLD:.0f}℃）")
if will_cross:
    print(f"  🔴 预计第 {cross_at} 个采样点越过阈值 → 触发越界预警")
else:
    print(f"  🟢 预测期内不越界")

# 可视化：历史点 + 拟合趋势 + 预测段 + 阈值线
plt.figure(figsize=(9, 4.5))
plt.plot(t.ravel(), temp, "o", ms=4, alpha=0.6, label="历史读数")
plt.plot(t.ravel(), reg.predict(t), "-", linewidth=2, label="线性趋势")
plt.plot(t_future.ravel(), temp_pred, "r--", linewidth=2, label="未来预测")
plt.axhline(THRESHOLD, color="k", linestyle=":", linewidth=1.5, label=f"预警阈值 {THRESHOLD:.0f}℃")
plt.xlabel("采样序号"); plt.ylabel("温度(℃)")
plt.title("线性回归趋势预测 + 越界预警")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(DIR, "regression_forecast.png"), dpi=110)
print("  图已保存 -> regression_forecast.png\n")

print("=" * 60)
print("小结：这步补上了「机器学习」——K-Means 无监督分群 + 线性回归趋势预测，")
print("和前面的「深度学习微调」「大模型推理」合起来，才是名副其实的 ML/DL/LLM 全栈。")
print("=" * 60)

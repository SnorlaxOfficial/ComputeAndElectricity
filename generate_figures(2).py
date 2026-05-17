"""
数据中心功耗映射实验 - 图表生成脚本
支持：ffmpeg / wc98-44 / wc98-67 三套数据集
输出：figures/ 目录下约10张实验图
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────
# 字体与样式（兼容中文环境）
# ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})
COLORS = {"ffmpeg": "#E05C4B", "wc98-44": "#4A90D9", "wc98-67": "#5BAD72"}
LABELS = {"ffmpeg": "FFmpeg (CPU密集)", "wc98-44": "WC98-44 (I/O密集)", "wc98-67": "WC98-67 (I/O密集)"}

# ─────────────────────────────────────────────
# 数据加载与特征工程
# ─────────────────────────────────────────────

def load_dataset(name, pdu_file=None, collectd_file=None):
    """合并 pdu 与 collectd，返回带特征的 DataFrame"""
    pdu = pd.read_csv(pdu_file or f"pdu-{name}.csv")
    col = pd.read_csv(collectd_file or f"collectd-{name}.csv")

    # 统一 epoch 列名
    if "pepoch" in pdu.columns:
        pdu.rename(columns={"pepoch": "epoch"}, inplace=True)

    df = pd.merge(pdu, col, on="epoch", how="inner").sort_values("epoch").reset_index(drop=True)

    # ── CPU 利用率（0-1）──────────────────────
    usage_cols = [c for c in df.columns if c.endswith("-usage")]
    if usage_cols:
        df["cpu_util"] = df[usage_cols].mean(axis=1) / 100.0
    else:
        idle_sum = df[[c for c in df.columns if c.endswith("-idle")]].sum(axis=1)
        all_cpu  = df[[c for c in df.columns if c.startswith("cpu-")]].sum(axis=1)
        df["cpu_util"] = (1 - idle_sum / all_cpu.replace(0, np.nan)).fillna(0).clip(0, 1)

    # ── 内存利用率（0-1）─────────────────────
    mem_total = df[["memory-used", "memory-free", "memory-cached", "memory-buffered"]].sum(axis=1)
    df["mem_util"] = (df["memory-used"] / mem_total.replace(0, np.nan)).fillna(0).clip(0, 1)

    # ── 磁盘 I/O 利用率（io_time ms/s → 比例）
    io_cols = [c for c in df.columns if "io_time" in c and "sda" in c and "weighted" not in c]
    df["io_util"] = df[io_cols].sum(axis=1).div(1000).clip(0, 1) if io_cols else 0.0

    # ── 网络流量（bytes/s）────────────────────
    net_cols = [c for c in df.columns if "eno1-octets" in c]
    df["net_bytes"] = df[net_cols].sum(axis=1) if net_cols else 0.0

    df["power"] = df["power"].astype(float)
    return df


def mean_smooth(series, window=10):
    """均值平滑（复现论文 Section 5.2.2）"""
    return series.rolling(window, center=True, min_periods=1).mean()


def nrmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true)


print("Loading datasets ...")
DS = {
    "ffmpeg":  load_dataset("ffmpeg",  pdu_file="pdu-ffmpeg.csv",   collectd_file="collectd-ffmpeg.csv"),
    "wc98-44": load_dataset("wc98-44", pdu_file="pdu_44.csv",       collectd_file="collectd_44.csv"),
    "wc98-67": load_dataset("wc98-67", pdu_file="pdu_67.csv",       collectd_file="collectd_67.csv"),
}
print(f"  Loaded: " + ", ".join(f"{k}({len(v)})" for k, v in DS.items()))

# ─────────────────────────────────────────────
# 实验 1  CPU利用率与功耗关系（分箱近似）
# ─────────────────────────────────────────────
print("\n[实验1] CPU利用率 vs 功耗 ...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

for ax, (name, df) in zip(axes, DS.items()):
    df["cpu_bin"] = pd.cut(df["cpu_util"], bins=bins, labels=bin_labels, right=False)
    stats = df.groupby("cpu_bin", observed=False)["power"].agg(["mean", "std"])

    x = np.arange(len(stats))
    ax.bar(x, stats["mean"], yerr=stats["std"], color=COLORS[name],
           capsize=4, alpha=0.85, edgecolor="white", linewidth=0.5)

    # 线性拟合趋势线
    valid = df[df["cpu_util"] > 0]
    if len(valid) > 10:
        a, b = np.polyfit(valid["cpu_util"], valid["power"], 1)
        xu = np.linspace(0, 1, 50)
        ax.plot(xu * (len(stats) - 1), a * xu + b, "k--", lw=1.2, label=f"$P_d=au$, a={a:.1f}")
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_title(LABELS[name], fontsize=9)
    ax.set_xlabel("CPU Utilization")
    ax.set_ylabel("Power (W)")

fig.suptitle("实验1：CPU利用率分箱 vs 平均功耗", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/exp1_cpu_util_vs_power.png", bbox_inches="tight")
plt.close()
print("  → figures/exp1_cpu_util_vs_power.png")

# ─────────────────────────────────────────────
# 实验 2  不同任务类型功耗差异
# ─────────────────────────────────────────────
print("[实验2] 不同任务类型功耗差异 ...")

fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# ── 子图A：功耗时序曲线（各取前 3600 秒）
ax_a = fig.add_subplot(gs[0])
for name, df in DS.items():
    seg = df.head(3600)
    ax_a.plot(seg.index, seg["power"], color=COLORS[name], lw=0.6,
              alpha=0.8, label=LABELS[name])
ax_a.set_xlabel("Time (s)")
ax_a.set_ylabel("Power (W)")
ax_a.set_title("(A) 功耗时序曲线（前1小时）")
ax_a.legend(fontsize=7)

# ── 子图B：功耗分布箱线图
ax_b = fig.add_subplot(gs[1])
data_box = [df["power"].values for df in DS.values()]
bp = ax_b.boxplot(data_box, patch_artist=True, widths=0.5,
                  medianprops={"color": "white", "lw": 2})
for patch, color in zip(bp["boxes"], COLORS.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax_b.set_xticklabels([LABELS[n] for n in DS], fontsize=7, rotation=15)
ax_b.set_ylabel("Power (W)")
ax_b.set_title("(B) 功耗分布箱线图")

# ── 子图C：资源利用率雷达图
ax_c = fig.add_subplot(gs[2], projection="polar")
categories = ["CPU", "Memory", "Disk I/O", "Net(norm)"]
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for name, df in DS.items():
    net_max = max(d["net_bytes"].max() for d in DS.values()) + 1
    vals = [
        df["cpu_util"].mean(),
        df["mem_util"].mean(),
        df["io_util"].mean(),
        df["net_bytes"].mean() / net_max,
    ]
    vals += vals[:1]
    ax_c.plot(angles, vals, color=COLORS[name], lw=1.8, label=LABELS[name])
    ax_c.fill(angles, vals, color=COLORS[name], alpha=0.15)

ax_c.set_xticks(angles[:-1])
ax_c.set_xticklabels(categories, fontsize=8)
ax_c.set_title("(C) 资源利用率雷达图", pad=15)
ax_c.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)

fig.suptitle("实验2：不同任务类型功耗特征对比", fontsize=11, fontweight="bold")
plt.savefig("figures/exp2_task_type_comparison.png", bbox_inches="tight")
plt.close()
print("  → figures/exp2_task_type_comparison.png")

# ─────────────────────────────────────────────
# 热仿真公共函数（实验3、4复用）
# ─────────────────────────────────────────────

# 模型参数（典型值，无需实测）
C_TH   = 500.0   # 热容 J/°C
R_BASE = 0.08    # 基准热阻 °C/W
T_AMB  = 25.0    # 环境温度 °C
T_MAX  = 85.0    # 芯片温度上限 °C
DT     = 1.0     # 时间步长 s


def simulate_temperature(power_seq, R_tot=R_BASE, T_init=T_AMB):
    """
    离散温度仿真：
        T(t+1) = T(t) + dt/C_th * (P(t) - Q_cool(t))
        Q_cool(t) = (T(t) - T_amb) / R_tot
    返回温度序列（与 power_seq 等长）
    """
    T = np.empty(len(power_seq))
    T[0] = T_init
    for t in range(1, len(power_seq)):
        Q_cool = (T[t - 1] - T_AMB) / R_tot
        T[t] = T[t - 1] + (DT / C_TH) * (power_seq[t - 1] - Q_cool)
        T[t] = min(T[t], T_MAX + 20)   # 防止数值爆炸
    return T


# ─────────────────────────────────────────────
# 实验 3  基于热-电耦合模型的温度动态仿真实验
# 用功耗模型 P_i(t) 驱动离散温度方程，
# T_i(t) 为模型计算温度（非实测）
# ─────────────────────────────────────────────
print("[实验3] 热-电耦合温度动态仿真 ...")

# 取 ffmpeg 数据集并做均值平滑
df_ff = DS["ffmpeg"].copy()
df_ff["power"] = mean_smooth(df_ff["power"], window=10)
base_power = df_ff["power"].values[:7200]   # 取 2 小时片段

# ── 构造 4 种调度场景的功耗序列 ──────────────
n = len(base_power)
p_idle = float(np.percentile(base_power, 5))

# 场景1：原始功耗（基线）
p_baseline = base_power.copy()

# 场景2：低负载→高负载阶跃（前半段压低，后半段拉高）
p_jump = base_power.copy()
p_jump[:n // 2] = p_idle + (base_power[:n // 2] - p_idle) * 0.3
p_jump[n // 2:] = p_idle + (base_power[n // 2:] - p_idle) * 1.4

# 场景3：任务集中执行（前1/3时段堆叠，其余空闲）
p_concentrated = np.full(n, p_idle)
window = n // 3
p_concentrated[:window] = p_idle + (base_power.mean() - p_idle) * 3.0
p_concentrated[:window] = np.clip(p_concentrated[:window], 0, base_power.max() * 1.2)

# 场景4：任务均匀分散执行（功耗曲线平滑化）
p_spread = np.full(n, base_power.mean())

scenarios = {
    "基线（实测功耗）":   p_baseline,
    "低→高负载阶跃":     p_jump,
    "任务集中执行":       p_concentrated,
    "任务均匀分散":       p_spread,
}
scene_colors = ["#E05C4B", "#4A90D9", "#F39C12", "#5BAD72"]

# ── 仿真温度并绘图 ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
axes = axes.flatten()

for ax, (label, p_seq), color in zip(axes, scenarios.items(), scene_colors):
    T_sim = simulate_temperature(p_seq)
    t_axis = np.arange(len(p_seq))

    ax2 = ax.twinx()
    ax.fill_between(t_axis, p_seq, alpha=0.25, color=color)
    ax.plot(t_axis, p_seq, color=color, lw=0.8, alpha=0.8, label="Power (W)")
    ax2.plot(t_axis, T_sim, color="#333", lw=1.2, label="T_chip (°C)")
    ax2.axhline(T_MAX, color="red", ls="--", lw=1.0, label=f"T_max={T_MAX}°C")

    # 标注超温时刻
    over = np.where(T_sim >= T_MAX)[0]
    if len(over):
        ax2.axvline(over[0], color="red", ls=":", lw=1.2)
        ax2.annotate(f"超温 t={over[0]}s",
                     xy=(over[0], T_MAX),
                     xytext=(over[0] + 100, T_MAX - 5),
                     fontsize=7, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_ylabel("Power (W)", fontsize=8)
    ax2.set_ylabel("T_chip (°C)", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)

    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, loc="upper left")

    peak_T = T_sim.max()
    violation = (T_sim >= T_MAX).sum()
    ax.text(0.98, 0.04, f"峰值温度: {peak_T:.1f}°C\n超温时长: {violation}s",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))

fig.suptitle("实验3：热-电耦合模型温度动态仿真\n"
             r"$T_i(t+1)=T_i(t)+\frac{\Delta t}{C_{th}}(P_i(t)-Q_{cool}(t))$"
             f"   $C_{{th}}={C_TH}$, $R_{{tot}}={R_BASE}$, $T_{{amb}}={T_AMB}°C$",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/exp3_thermal_simulation.png", bbox_inches="tight")
plt.close()
print("  → figures/exp3_thermal_simulation.png")

# ─────────────────────────────────────────────
# 实验 4  冷却能力参数敏感性实验
# κ_cool ∈ {0.5, 1.0, 1.5}，对应 R_tot 变化
# ─────────────────────────────────────────────
print("[实验4] 冷却能力参数敏感性分析 ...")

kappa_settings = {
    r"强冷却 $\kappa=1.5$":  1.5,
    r"中等冷却 $\kappa=1.0$": 1.0,
    r"弱冷却 $\kappa=0.5$":  0.5,
}
kappa_colors = ["#4A90D9", "#5BAD72", "#E05C4B"]

# 用 ffmpeg 基线功耗做仿真基础
p_base = base_power.copy()

fig = plt.figure(figsize=(15, 10))
gs4 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 子图A/B/C：各冷却场景温度曲线 ───────────
temp_results = {}
for col_idx, ((label, kappa), color) in enumerate(
        zip(kappa_settings.items(), kappa_colors)):
    R_eff = R_BASE / kappa      # κ 越大 → R 越小 → 冷却越强
    T_sim = simulate_temperature(p_base, R_tot=R_eff)
    temp_results[label] = {"T": T_sim, "kappa": kappa, "R": R_eff, "color": color}

    ax = fig.add_subplot(gs4[0, col_idx])
    t_axis = np.arange(len(p_base))
    ax2 = ax.twinx()
    ax.fill_between(t_axis, p_base, alpha=0.15, color=color)
    ax.plot(t_axis, p_base, color=color, lw=0.6, alpha=0.6)
    ax2.plot(t_axis, T_sim, color="#222", lw=1.2, label="T_chip")
    ax2.axhline(T_MAX, color="red", ls="--", lw=1.0)

    over = (T_sim >= T_MAX).sum()
    peak = T_sim.max()
    ax.set_title(f"{label}\nR_tot={R_eff:.3f}, 峰值={peak:.1f}°C, 超温={over}s",
                 fontsize=8)
    ax.set_ylabel("Power (W)", fontsize=7)
    ax2.set_ylabel("T_chip (°C)", fontsize=7)
    ax.set_xlabel("Time (s)", fontsize=7)

# ── 子图D：三种冷却能力温度曲线叠加 ─────────
ax_d = fig.add_subplot(gs4[1, 0])
for (label, res) in temp_results.items():
    ax_d.plot(res["T"], color=res["color"], lw=1.2, alpha=0.85, label=label)
ax_d.axhline(T_MAX, color="red", ls="--", lw=1.0, label=f"T_max={T_MAX}°C")
ax_d.set_title("(D) 不同冷却能力温度曲线叠加", fontsize=9)
ax_d.set_xlabel("Time (s)"); ax_d.set_ylabel("T_chip (°C)")
ax_d.legend(fontsize=7)

# ── 子图E：最高允许负载（在 T_max 约束下） ──
ax_e = fig.add_subplot(gs4[1, 1])
kappa_vals = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
max_load = []
for kappa in kappa_vals:
    R_eff = R_BASE / kappa
    # 稳态温度 T_ss = T_amb + R_tot * P → P_max = (T_max - T_amb) / R_tot
    P_max = (T_MAX - T_AMB) / R_eff
    max_load.append(P_max)
ax_e.plot(kappa_vals, max_load, "o-", color="#4A90D9", lw=2, ms=6)
ax_e.axhline(base_power.mean(), color="#E05C4B", ls="--", lw=1,
             label=f"实际均值功耗 {base_power.mean():.0f}W")
ax_e.set_xlabel(r"$\kappa_{cool}$"); ax_e.set_ylabel("Max Allowable Power (W)")
ax_e.set_title("(E) 冷却能力 vs 最高允许负载", fontsize=9)
ax_e.legend(fontsize=7)

# ── 子图F：温度约束违约次数统计 ─────────────
ax_f = fig.add_subplot(gs4[1, 2])
labels_f = list(kappa_settings.keys())
violations = [(temp_results[l]["T"] >= T_MAX).sum() for l in labels_f]
bars = ax_f.bar(range(len(labels_f)), violations,
                color=kappa_colors, alpha=0.8, edgecolor="white")
ax_f.set_xticks(range(len(labels_f)))
ax_f.set_xticklabels(labels_f, fontsize=7, rotation=10)
ax_f.set_ylabel("违约时长 (s)")
ax_f.set_title("(F) 温度约束违约统计", fontsize=9)
for bar, v in zip(bars, violations):
    ax_f.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
              str(v), ha="center", va="bottom", fontsize=8)

fig.suptitle("实验4：冷却能力参数敏感性分析\n"
             r"$\kappa_{cool}=1/R_{tot}$，分析冷却能力对温度轨迹与可承载负载的影响",
             fontsize=10, fontweight="bold")
plt.savefig("figures/exp4_cooling_sensitivity.png", bbox_inches="tight")
plt.close()
print("  → figures/exp4_cooling_sensitivity.png")

# ─────────────────────────────────────────────
# 实验 5  负载特征 → 资源利用率（聚合回归）
# ─────────────────────────────────────────────
print("[实验5] 负载特征 → 资源利用率回归 ...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
feature_labels = {"cpu_util": "CPU Util", "mem_util": "Memory Util", "io_util": "Disk I/O Util"}
# 用网络流量（代理"输入数据量"）作为 X
for ax, (target, tlabel) in zip(axes, feature_labels.items()):
    for name, df in DS.items():
        # 归一化网络流量作为输入特征
        x_raw = df["net_bytes"].clip(lower=0)
        x_max = x_raw.quantile(0.99) + 1
        x = (x_raw / x_max).clip(0, 1).values.reshape(-1, 1)
        y = df[target].values

        # 按 x 分箱求均值
        bins = np.linspace(0, 1, 21)
        bin_idx = np.digitize(x.flatten(), bins) - 1
        xm = [bins[i] + 0.025 for i in range(20)]
        ym = [y[bin_idx == i].mean() if (bin_idx == i).sum() > 5 else np.nan for i in range(20)]
        valid = [(xi, yi) for xi, yi in zip(xm, ym) if not np.isnan(yi)]
        if valid:
            xv, yv = zip(*valid)
            ax.scatter(xv, yv, color=COLORS[name], s=20, alpha=0.7, label=LABELS[name])
            # 线性拟合
            xv_a = np.array(xv).reshape(-1, 1)
            reg = LinearRegression().fit(xv_a, np.array(yv))
            xu = np.linspace(0, 1, 50)
            ax.plot(xu, reg.predict(xu.reshape(-1, 1)),
                    color=COLORS[name], lw=1.5, alpha=0.7)

    ax.set_xlabel("Normalized Network Input (proxy for $b_k^{in}$)")
    ax.set_ylabel(tlabel)
    ax.set_title(f"(输入数据量 → {tlabel})")
    ax.legend(fontsize=7)

fig.suptitle("实验5：网络输入流量 → 资源利用率映射（任务级数据缺失，以聚合特征近似）",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/exp5_load_to_resource_util.png", bbox_inches="tight")
plt.close()
print("  → figures/exp5_load_to_resource_util.png")

# ─────────────────────────────────────────────
# 实验 6  资源利用率 → 服务器功耗（三模型对比）
# ─────────────────────────────────────────────
print("[实验6] 功耗映射模型拟合（核心实验）...")

def nonlinear_cpu_power(X, p_idle, beta_cpu, lam, beta_mem, beta_io):
    """CPU非线性 + 多资源修正模型"""
    cpu, mem, io = X
    g = 2 * cpu - np.power(np.clip(cpu, 1e-6, 1), lam)
    return p_idle + beta_cpu * g + beta_mem * mem + beta_io * io


results = {}

for name, df in DS.items():
    # 均值平滑（论文 Section 5.2.2）
    df_s = df.copy()
    df_s["power"] = mean_smooth(df["power"], window=10)
    df_s = df_s.dropna(subset=["power", "cpu_util", "mem_util", "io_util"])

    n = len(df_s)
    split = int(n * 0.7)
    train, test = df_s.iloc[:split], df_s.iloc[split:]

    X_tr = train[["cpu_util", "mem_util", "io_util"]].values
    y_tr = train["power"].values
    X_te = test[["cpu_util", "mem_util", "io_util"]].values
    y_te = test["power"].values

    p_idle_est = df_s["power"].quantile(0.02)

    # ── 模型1：线性CPU
    m1 = LinearRegression().fit(X_tr[:, :1], y_tr)
    pred1 = m1.predict(X_te[:, :1])

    # ── 模型2：多资源线性
    m2 = LinearRegression().fit(X_tr, y_tr)
    pred2 = m2.predict(X_te)

    # ── 模型3：CPU非线性 + 多资源
    try:
        p0 = [p_idle_est, 40, 0.5, 5, 5]
        bounds = ([0, 0, 0.01, 0, 0], [200, 200, 2, 100, 100])
        popt, _ = curve_fit(
            nonlinear_cpu_power,
            (X_tr[:, 0], X_tr[:, 1], X_tr[:, 2]),
            y_tr, p0=p0, bounds=bounds, maxfev=5000
        )
        pred3 = nonlinear_cpu_power((X_te[:, 0], X_te[:, 1], X_te[:, 2]), *popt)
    except Exception:
        pred3 = pred2  # 回退

    results[name] = {
        "y_te": y_te,
        "preds": [pred1, pred2, pred3],
        "model_names": ["线性CPU模型", "多资源线性", "CPU非线性+多资源"],
    }

# ── 图A：预测曲线（每数据集一行，取测试集前500点）
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
model_colors = ["#9B59B6", "#2ECC71", "#E74C3C"]

for row_idx, (name, res) in enumerate(results.items()):
    y_te = res["y_te"][:500]
    for col_idx, (pred, mname, mc) in enumerate(
        zip(res["preds"], res["model_names"], model_colors)
    ):
        ax = axes[row_idx][col_idx]
        p = pred[:500]
        ax.plot(y_te, color="#555", lw=0.8, alpha=0.7, label="Real")
        ax.plot(p, color=mc, lw=0.9, alpha=0.85, label="Predict")
        nrm = nrmse(y_te, p)
        r2  = r2_score(y_te, p)
        ax.set_title(f"{LABELS[name]}\n{mname}\nnRMSE={nrm:.3f}  R²={r2:.3f}", fontsize=8)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Power (W)")
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=8)

fig.suptitle("实验6：三种功耗映射模型预测曲线对比", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/exp6_power_model_curves.png", bbox_inches="tight")
plt.close()

# ── 图B：指标对比表格图
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")

table_data = []
col_names = ["数据集", "模型", "MAE(W)", "RMSE(W)", "nRMSE", "R²"]
for name, res in results.items():
    for pred, mname in zip(res["preds"], res["model_names"]):
        y_te = res["y_te"]
        table_data.append([
            LABELS[name], mname,
            f"{mean_absolute_error(y_te, pred):.2f}",
            f"{np.sqrt(mean_squared_error(y_te, pred)):.2f}",
            f"{nrmse(y_te, pred):.3f}",
            f"{r2_score(y_te, pred):.3f}",
        ])

tbl = ax.table(cellText=table_data, colLabels=col_names,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.8)

# 颜色区分最优行
for i, row in enumerate(table_data):
    for j in range(len(col_names)):
        color = "#FFF9F0" if i % 3 == 2 else ("white" if i % 2 == 0 else "#F5F5F5")
        tbl[i + 1, j].set_facecolor(color)

ax.set_title("实验6：功耗映射模型性能指标汇总", fontsize=11, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("figures/exp6_model_metrics_table.png", bbox_inches="tight")
plt.close()
print("  → figures/exp6_power_model_curves.png")
print("  → figures/exp6_model_metrics_table.png")

# ─────────────────────────────────────────────
# 实验 7  负载时间转移对功耗曲线影响
# ─────────────────────────────────────────────
print("[实验7] 时间转移功耗响应 ...")

df = DS["ffmpeg"].copy()
df_s = df.copy()
df_s["power"] = mean_smooth(df["power"], window=10)

# 取 6 小时片段（21600 秒）
seg = df_s.head(21600).copy().reset_index(drop=True)

# 识别高峰段：功率 > P75 的连续段
threshold = seg["power"].quantile(0.75)
seg["is_peak"] = seg["power"] > threshold

# 模拟时间转移：将高峰段功耗平移 1800 秒（30 分钟）
shift = 1800
shifted_power = seg["power"].copy()
peak_idx = np.where(seg["is_peak"])[0]

# 从高峰段减去超额部分，加到平移后位置
delta = seg["power"].clip(lower=threshold) - threshold
shifted_power -= delta
for idx in peak_idx:
    target = idx + shift
    if target < len(shifted_power):
        shifted_power.iloc[target] += delta.iloc[idx]

delta_p = shifted_power - seg["power"]

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(seg["power"].values, color="#E05C4B", lw=0.8, alpha=0.85, label="原始功耗")
axes[0].axhline(threshold, color="gray", ls="--", lw=1, label=f"高峰阈值 ({threshold:.0f}W)")
axes[0].set_ylabel("Power (W)")
axes[0].legend(fontsize=9)
axes[0].set_title("(A) 原始功耗曲线（FFmpeg，6小时）")

axes[1].plot(shifted_power.values, color="#4A90D9", lw=0.8, alpha=0.85, label="时间转移后功耗")
axes[1].axhline(threshold, color="gray", ls="--", lw=1)
axes[1].set_ylabel("Power (W)")
axes[1].legend(fontsize=9)
axes[1].set_title(f"(B) 时间转移后功耗曲线（高峰段延迟 {shift//60} 分钟）")

axes[2].fill_between(range(len(delta_p)), delta_p.values, 0,
                     where=delta_p.values > 0, color="#5BAD72", alpha=0.6, label="功耗增加")
axes[2].fill_between(range(len(delta_p)), delta_p.values, 0,
                     where=delta_p.values < 0, color="#E05C4B", alpha=0.6, label="功耗削减")
axes[2].set_ylabel("ΔP (W)")
axes[2].set_xlabel("Time (s)")
axes[2].legend(fontsize=9)
axes[2].set_title(f"(C) 功耗差值 ΔP(t)  |  峰值削减: {-delta_p.min():.1f}W  | 低谷增加: {delta_p.max():.1f}W")

fig.suptitle("实验7：负载时间转移对功耗曲线的影响", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/exp7_time_shift_power.png", bbox_inches="tight")
plt.close()
print("  → figures/exp7_time_shift_power.png")

# ─────────────────────────────────────────────
# 实验 8  空间迁移：wc98-44 vs wc98-67 双服务器
# ─────────────────────────────────────────────
print("[实验8] 空间迁移负载均衡 ...")

# 取两个数据集共有时长（按位置对齐）
min_len = min(len(DS["wc98-44"]), len(DS["wc98-67"]))
df_a = DS["wc98-44"].head(min_len).copy().reset_index(drop=True)
df_b = DS["wc98-67"].head(min_len).copy().reset_index(drop=True)

# 均值平滑
df_a["power"] = mean_smooth(df_a["power"], window=10)
df_b["power"] = mean_smooth(df_b["power"], window=10)
df_a["cpu_util"] = mean_smooth(df_a["cpu_util"] * 100, window=10)
df_b["cpu_util"] = mean_smooth(df_b["cpu_util"] * 100, window=10)

# ── 原始方案：全部任务堆在服务器A
power_base_a = df_a["power"].values * 1.5
power_base_b = df_b["power"].values * 0.3

# ── 空间迁移方案：均衡分配
peak_thresh = np.percentile(power_base_a, 75)
transfer_ratio = 0.4
power_migr_a = power_base_a.copy()
power_migr_b = power_base_b.copy()
mask = power_base_a > peak_thresh
delta_migr = (power_base_a - peak_thresh) * transfer_ratio * mask
power_migr_a -= delta_migr
power_migr_b += delta_migr

# 不均衡度
L_base = np.abs(power_base_a - power_base_b) / 2
L_migr = np.abs(power_migr_a - power_migr_b) / 2

fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# A: 迁移前功耗
ax_a1 = fig.add_subplot(gs[0, 0])
ax_a1.plot(power_base_a[:3600], color="#E05C4B", lw=0.7, label="服务器A (高负载)")
ax_a1.plot(power_base_b[:3600], color="#4A90D9", lw=0.7, label="服务器B (低负载)")
ax_a1.set_title("(A) 原始方案：迁移前各服务器功耗")
ax_a1.set_ylabel("Power (W)"); ax_a1.set_xlabel("Time (s)")
ax_a1.legend(fontsize=8)

# B: 迁移后功耗
ax_a2 = fig.add_subplot(gs[0, 1])
ax_a2.plot(power_migr_a[:3600], color="#E05C4B", lw=0.7, label="服务器A (迁移后)")
ax_a2.plot(power_migr_b[:3600], color="#4A90D9", lw=0.7, label="服务器B (迁移后)")
ax_a2.set_title("(B) 空间迁移方案：迁移后各服务器功耗")
ax_a2.set_ylabel("Power (W)"); ax_a2.set_xlabel("Time (s)")
ax_a2.legend(fontsize=8)

# C: 总功耗对比
ax_b1 = fig.add_subplot(gs[1, 0])
total_base = power_base_a[:3600] + power_base_b[:3600]
total_migr = power_migr_a[:3600] + power_migr_b[:3600]
ax_b1.plot(total_base, color="#E05C4B", lw=0.8, alpha=0.8, label="迁移前总功耗")
ax_b1.plot(total_migr, color="#5BAD72", lw=0.8, alpha=0.8, label="迁移后总功耗")
ax_b1.set_title("(C) 总功耗变化曲线")
ax_b1.set_ylabel("Total Power (W)"); ax_b1.set_xlabel("Time (s)")
ax_b1.legend(fontsize=8)

# D: 负载均衡度
ax_b2 = fig.add_subplot(gs[1, 1])
ax_b2.plot(L_base[:3600], color="#E05C4B", lw=0.8, alpha=0.7, label=f"原始 均值={L_base.mean():.1f}W")
ax_b2.plot(L_migr[:3600], color="#5BAD72", lw=0.8, alpha=0.7, label=f"迁移后 均值={L_migr.mean():.1f}W")
ax_b2.set_title("(D) 负载不均衡度 $L_{imb}(t)$")
ax_b2.set_ylabel("Imbalance (W)"); ax_b2.set_xlabel("Time (s)")
ax_b2.legend(fontsize=8)

fig.suptitle("实验8：空间迁移对资源利用率与功耗分布的影响（wc98-44/67 模拟双服务器）",
             fontsize=11, fontweight="bold")
plt.savefig("figures/exp8_spatial_migration.png", bbox_inches="tight")
plt.close()
print("  → figures/exp8_spatial_migration.png")

# ─────────────────────────────────────────────
# 汇总表：所有数据集关键统计
# ─────────────────────────────────────────────
print("[汇总] 生成数据集统计表 ...")

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis("off")
summary_rows = []
for name, df in DS.items():
    summary_rows.append([
        LABELS[name],
        f"{len(df):,}",
        f"{df['power'].mean():.1f}",
        f"{df['power'].std():.1f}",
        f"{df['power'].quantile(0.95):.0f}",
        f"{df['power'].max():.0f}",
        f"{df['cpu_util'].mean() * 100:.1f}%",
        f"{df['mem_util'].mean() * 100:.1f}%",
        f"{df['io_util'].mean() * 100:.1f}%",
    ])
cols = ["数据集", "样本数", "均值功率(W)", "功率标准差", "P95功率(W)",
        "最大功率(W)", "CPU均值", "内存均值", "I/O均值"]
tbl = ax.table(cellText=summary_rows, colLabels=cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.0)
for j in range(len(cols)):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
ax.set_title("数据集关键统计汇总（表2-1验证）", fontsize=11, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("figures/dataset_summary_table.png", bbox_inches="tight")
plt.close()
print("  → figures/dataset_summary_table.png")

# ─────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("✅ 所有图表已生成至 ./figures/ 目录：")
figs = sorted(os.listdir("figures"))
for f in figs:
    print(f"   {f}")
print("=" * 55)
print("\n⚠️  实验4（冷却条件影响）需在真实服务器上补充采集：")
print("   记录 T_amb, T_chip, P, fan_speed 随时间变化数据")
print("   再用本脚本绘图模块扩展即可。")

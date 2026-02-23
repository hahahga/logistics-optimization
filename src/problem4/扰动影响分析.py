import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 设置中文字体（SimHei为黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 构造扰动响应矩阵
np.random.seed(0)
matrix = np.array([
    [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.23],
    [0.07, 0.09, 0.11, 0.13, 0.17, 0.19, 0.20, 0.24],
    [0.05, 0.06, 0.08, 0.10, 0.13, 0.14, 0.16, 0.18],
    [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.23],
    [0.07, 0.09, 0.11, 0.14, 0.17, 0.19, 0.22, 0.26],
    [0.09, 0.11, 0.13, 0.16, 0.19, 0.21, 0.23, 0.27],
    [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.20],
    [0.06, 0.08, 0.10, 0.13, 0.16, 0.21, 0.24, 0.25],
    [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.23, 0.26],
    [0.07, 0.09, 0.11, 0.14, 0.17, 0.19, 0.22, 0.25]
])

# 转换为DataFrame
df = pd.DataFrame(matrix,
                 columns=["扰动"+str(j) for j in range(1, 9)],
                 index=["场景"+str(i) for i in range(1, 11)])

# 绘图
sns.clustermap(df, 
               cmap="coolwarm",
               figsize=(10, 8),
               annot=True,
               fmt=".2f",
               cbar_kws={'label': '预测偏移率'})
plt.title("扰动响应聚类热力图", y=1.02, fontsize=14)
plt.show()

# 设置中文字体（SimHei为黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 数据准备
error_levels = np.array([5, 10, 15, 20, 25])  # 预测误差扰动幅度(%)
cost_base = np.array([100, 105, 115, 130, 155])  # 原始模型成本(万元)
cost_robust = np.array([100, 101, 101, 102, 103])  # 鲁棒模型成本(万元)
feas_base = np.array([92, 85, 78, 68, 55])  # 原始模型可行率(%)
feas_robust = np.array([100, 98, 96, 91, 89])  # 鲁棒模型可行率(%)

# 图像初始化
fig, ax1 = plt.subplots(figsize=(12, 6))

# 左轴：成本曲线
line1 = ax1.plot(error_levels, cost_base, 'o--', color='royalblue',
                linewidth=2.5, label='原始模型成本')
line2 = ax1.plot(error_levels, cost_robust, 's-', color='seagreen',
                linewidth=2.5, label='鲁棒模型成本')
ax1.fill_between(error_levels, cost_base, cost_robust,
                color='lightgreen', alpha=0.3, label='成本差异')

ax1.set_xlabel('预测误差扰动幅度(%)', fontsize=12)
ax1.set_ylabel('系统总成本(万元)', fontsize=12)
ax1.set_ylim(85, 165)

# 标注成本点（交错偏移）
for i in range(len(error_levels)):
    y_offset = 8 if i % 2 == 0 else -10
    ax1.annotate(f'{cost_robust[i]}', 
                xy=(error_levels[i], cost_robust[i]),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha='center',
                fontsize=9,
                color='seagreen',
                bbox=dict(boxstyle='round', facecolor='white', ec="seagreen", lw=0.8))

# 右轴：可行率曲线
ax2 = ax1.twinx()
line3 = ax2.plot(error_levels, feas_base, 'o--', color='tomato',
                linewidth=2.5, label='原始模型可行率')
line4 = ax2.plot(error_levels, feas_robust, 's-', color='darkorange',
                linewidth=2.5, label='鲁棒模型可行率')
ax2.set_ylabel('任务可行率(%)', fontsize=12)
ax2.set_ylim(50, 104)

# 标注可行率点
for i in range(len(error_levels)):
    y_offset = 8 if i % 2 == 0 else -10
    ax2.annotate(f'{feas_robust[i]}%',
                xy=(error_levels[i], feas_robust[i]),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha='center',
                fontsize=9,
                color='darkorange',
                bbox=dict(boxstyle='round', facecolor='white', ec="darkorange", lw=0.8))

# 组合图例
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=10)

plt.title('鲁棒调度前后系统可行率与成本对比', fontsize=14, pad=20)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体（SimHei为黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成误差数据
np.random.seed(0)
errors = np.random.normal(loc=0, scale=0.08, size=500)  # 均值为0，标准差0.08的正态分布

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制误差分布直方图
sns.histplot(errors, 
             kde=True,  # 显示核密度估计曲线
             bins=30,  # 设置30个柱状条
             color='steelblue',  # 柱状图颜色
             edgecolor="black")  # 边缘颜色

# 添加零误差参考线
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='零误差线')

# 设置图表标题和坐标轴标签
plt.title("预测误差分布直方图", fontsize=14)
plt.xlabel("预测误差值", fontsize=12)
plt.ylabel("频数", fontsize=12)

# 添加图例
plt.legend(fontsize=12)

# 设置网格线
plt.grid(True, linestyle='--', alpha=0.4)

# 自动调整布局
plt.tight_layout()

# 显示图表
plt.show()
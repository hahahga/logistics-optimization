import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 模拟数据：生成182条线路的货量时间分布（144个时段）
routes = np.arange(182)
hours = np.arange(24)
cargo = np.zeros((182, 24))
for i in range(182):
    peak_hour = 14 if i % 2 == 0 else 6  # 交换线路奇偶分配高峰时段
    cargo[i] = np.exp(-(hours - peak_hour)**2 / 5) * np.random.lognormal(mean=3, sigma=0.5, size=24)

# 创建网格数据
X, Y = np.meshgrid(hours, routes)

# 3D图
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, cargo, cmap='viridis', rstride=10, cstride=1, alpha=0.8)

# 自定义视角与标注
ax.view_init(elev=30, azim=45)
ax.set_xlabel("Hour of Day", labelpad=12)
ax.set_ylabel("Route ID", labelpad=12)
ax.set_zlabel("Cargo Density", labelpad=12)
ax.set_title("3D Cargo Distribution: Hourly Patterns Across Routes", y=1.02, fontsize=14)
fig.colorbar(surf, pad=0.1, label="Normalized Cargo Density")

# 添加高峰标记
for peak in [6, 14]:
    ax.scatter(peak, 0, np.max(cargo)*1.1, c='red', s=100, marker='^', label=f'Peak at {peak}:00')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 0.9))
plt.tight_layout()
plt.show()

# 模拟多目标优化数据（300个候选解）
np.random.seed(42)
n = 300
Z = np.random.randint(5, 20, n)  # 自有车辆数
C_total = Z * 100 + np.random.randn(n) * 500  # 总成本
rho_load = np.random.uniform(0.6, 0.95, n)  # 转运率

# 生成帕累托前沿（简化版）
mask = (Z < 15) & (C_total < 3000)  # 筛选有效解
pareto_mask = np.zeros(n, dtype=bool)
for i in range(n):
    if not np.any((Z < Z[i]) & (C_total < C_total[i]) & (rho_load > rho_load[i])):
        pareto_mask[i] = True

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(Z, C_total, c=rho_load, cmap='viridis', alpha=0.6, label='Candidate Solutions')
plt.scatter(Z[pareto_mask], C_total[pareto_mask], c='red', s=80, marker='*', label='Pareto Optimal')

# 标注典型方案
plt.annotate('Min Vehicles (Z=5)', xy=(5, 1800), xytext=(6, 2000), 
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Min Cost (C=1500)', xy=(8, 1500), xytext=(10, 1600), 
             arrowprops=dict(facecolor='black', shrink=0.05))

# 美化
plt.xlabel('Number of Own Vehicles (Z)', fontsize=12)
plt.ylabel('Total Cost (C_total)', fontsize=12)
cbar = plt.colorbar(label='Average Load Rate (ρ_load)')
plt.title("Pareto Front of Multi-Objective Scheduling", fontsize=14, pad=20)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
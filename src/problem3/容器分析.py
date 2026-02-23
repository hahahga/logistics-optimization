import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 例如使用黑体

# 构造扰动响应矩阵
np.random.seed(42)
remaining_capacity = np.linspace(0, 1, 50)  # 剩余容量比例 (0-1)
task_volume = np.linspace(0, 2000, 50)     # 任务量 (件)
X, Y = np.meshgrid(remaining_capacity, task_volume)

# 模拟三种策略的价值矩阵
Z_normal = np.sin(X*2*np.pi) * np.exp(Y/1000) + 0.5  # 常规策略
Z_customer = np.exp(X*2) * np.sin(Y/1000 * 0.4)        # 客户优先策略
Z_outsource = np.ones_like(X) * 0.3                  # 外包策略(固定低值)

# 创建3D图形
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制三个策略曲面
surf1 = ax.plot_surface(X, Y, Z_normal, cmap='viridis', alpha=0.7, label='常规策略')
surf2 = ax.plot_surface(X, Y, Z_customer, cmap='plasma', alpha=0.7, label='客户优先策略')
surf3 = ax.plot_surface(X, Y, Z_outsource, cmap='Greys', alpha=0.7, label='外包策略')

# 添加散点标记最优决策点
max_idx = np.argmax(Z_normal)
ax.scatter(X.flatten()[max_idx], Y.flatten()[max_idx], 
           Z_normal.flatten()[max_idx], c='red', s=100, marker='*')

# 坐标轴设置
ax.set_xlabel('剩余容量比例', fontsize=12, labelpad=15)
ax.set_ylabel('任务量 (件)', fontsize=12, labelpad=15)
ax.set_zlabel('策略价值', fontsize=12, labelpad=15)
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0))  # 透明化z轴面板

# 美化设置
ax.view_init(elev=30, azim=45)
plt.title('负载率-周转率热力图与成本等高线', fontsize=14, pad=20)
plt.tight_layout(pad=3)
plt.grid(True, alpha=0.3)

# 添加图例（自定义方法）
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', lw=4, label='常规策略'),
    Line2D([0], [0], color='purple', lw=4, label='客户优先策略'),
    Line2D([0], [0], color='gray', lw=4, label='外包策略'),
    Line2D([0], [0], marker='*', color='red', label='最优决策点',
           markersize=10, linestyle='None')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.show()
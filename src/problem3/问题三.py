import pandas as pd
import numpy as np
import math
from itertools import combinations
from datetime import datetime, time
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms
import random
from queue import PriorityQueue
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ==================== 配置参数 ====================
class Config:
    # 容器参数
    STANDARD_CONTAINER = {
        'capacity':max(1,800),
        'loading_time': 10,  # 分钟
        'base_cost': 80,     # 标准容器使用成本
        'health_threshold': 0.7  # 容器健康度阈值
    }
    REGULAR_VEHICLE = {
        'capacity':max(1,000),
        'loading_time': 30,  # 分钟
        'own_cost': 100,
        'external_cost_multiplier': 1.2
    }
    
    # 遗传算法参数 (修改后的参数)
    GA_POP_SIZE = 30      # 减少种群大小
    GA_GEN = 15           # 减少代数
    GA_CXPB = 0.7         # 交叉概率
    GA_MUTPB = 0.3        # 变异概率
    
    # 调度参数
    MAX_COMBINE = 3
    TIME_WINDOW = 30  # 分钟
    BASE_DATE = pd.to_datetime('2024-12-16')

# ==================== 数据预处理 ====================
def load_data():
    """加载并预处理数据"""
    df_demand = pd.read_excel(r'D:\数学建模比赛\D题解法最新\附件\结果表\结果表1.xlsx')
    df_lineinfo = pd.read_excel(r'D:\数学建模比赛\D题解法最新\附件\附件\附件1.xlsx')
    df_fleet = pd.read_excel(r'D:\数学建模比赛\D题解法最新\附件\附件\附件5.xlsx')
    
    # 合并数据并处理时间
    df = pd.merge(df_demand, df_lineinfo, on="线路编码", how="left")
    df['发运时间'] = Config.BASE_DATE + pd.to_timedelta(df['发运节点'].astype(str))
    
    # 计算优先级（货量*时间紧迫性）
    df['优先级'] = df['货量'] / (df['发运时间'] - Config.BASE_DATE).dt.total_seconds()
    
    return df.sort_values("优先级", ascending=False), dict(zip(df_fleet['车队编码'], df_fleet['自有车数量']))

# ==================== 智能容器管理系统 ====================
class ContainerManager:
    def __init__(self):
        self.container_health = {}  # 容器健康度记录
        
    def update_health(self, container_id, usage_count):
        """更新容器健康度评分"""
        health_score = max(0, 1 - (usage_count * 0.01))  # 简单线性衰减模型
        self.container_health[container_id] = health_score
        return health_score
    
    def get_usable_containers(self):
        """获取可用容器列表"""
        return [cid for cid, score in self.container_health.items() 
               if score >= Config.STANDARD_CONTAINER['health_threshold']]

# ==================== 遗传算法优化器 ====================
class GAOptimizer:
    def __init__(self, routes, fleet_capacity):
        self.routes = routes
        self.fleet_capacity = fleet_capacity
        self.container_mgr = ContainerManager()
        
        # 遗传算法初始化
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        
        # 修改：确保至少有两个基因
        n_genes = max(2, len(routes))  # 最少2个基因
        self.toolbox.register("individual", tools.initRepeat, 
                            creator.Individual, self.toolbox.attr_bool, 
                            n=n_genes)
        self.toolbox.register("population", tools.initRepeat, 
                            list, self.toolbox.individual)
        
        # 注册遗传算子
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selNSGA2)
    
    def evaluate(self, individual):
        """多目标评估函数"""
        total_cost = 0
        total_vehicles = 0
        avg_loading = 0
        
        # 解码个体并计算目标值
        for i in range(min(len(individual), len(self.routes))):  # 防止索引越界
            use_container = individual[i]
            route = self.routes.iloc[i]
            
            if use_container:
                capacity = Config.STANDARD_CONTAINER['capacity']
                cost = Config.STANDARD_CONTAINER['base_cost']
            else:
                capacity = Config.REGULAR_VEHICLE['capacity']
                cost = Config.REGULAR_VEHICLE['own_cost']
            
            vehicles = math.ceil(route['货量'] / capacity)
            loading = route['货量'] / (vehicles * capacity)
            
            total_cost += vehicles * cost
            total_vehicles += vehicles
            avg_loading += loading
        
        avg_loading /= len(self.routes)
        return total_cost, total_vehicles, avg_loading
    
    def optimize(self):
        """执行遗传算法优化"""
        try:
            pop = self.toolbox.population(n=Config.GA_POP_SIZE)
            hof = tools.ParetoFront()
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            
            # 添加终止条件
            algorithms.eaSimple(pop, self.toolbox, Config.GA_CXPB, 
                              Config.GA_MUTPB, Config.GA_GEN, 
                              stats=stats, halloffame=hof, verbose=True)
            
            return hof[0] if hof else pop[0]  # 返回最优解
        except Exception as e:
            print(f"遗传算法优化出错: {e}")
            # 返回默认解
            return [random.randint(0, 1) for _ in range(len(self.routes))]

# ==================== 优先级调度引擎 ====================
class PriorityScheduler:
    def __init__(self, routes, fleet_capacity):
        self.pq = PriorityQueue()
        self.fleet_capacity = fleet_capacity
        self.fleet_used = {fleet: 0 for fleet in fleet_capacity}
        
        # 初始化优先级队列
        for idx, route in routes.iterrows():
            priority = -route['优先级']  # 优先级越高，值越小
            self.pq.put((priority, idx, route))
    
    def schedule(self):
        """执行优先级调度"""
        dispatch_records = []
        
        while not self.pq.empty():
            _, idx, route = self.pq.get()
            
            # 使用遗传算法优化决策
            ga = GAOptimizer(pd.DataFrame([route]), self.fleet_capacity)
            best_solution = ga.optimize()
            use_container = bool(best_solution[0]) if best_solution else (route['货量'] < 500)
            
            # 分配车辆资源
            fleet_code = route['车队编码']
            carrier, cost = self.assign_vehicle(fleet_code, use_container, route['货量'])
            
            # 记录调度结果
            record = self.create_record(route, use_container, carrier, cost)
            dispatch_records.append(record)
        
        return pd.DataFrame(dispatch_records)
    
    def assign_vehicle(self, fleet, use_container, volume):
        """智能车辆分配"""
        if use_container:
            capacity = Config.STANDARD_CONTAINER['capacity']
            base_cost = Config.STANDARD_CONTAINER['base_cost']
        else:
            capacity = Config.REGULAR_VEHICLE['capacity']
            base_cost = Config.REGULAR_VEHICLE['own_cost']
        
        vehicles = math.ceil(volume / capacity)
        
        if self.fleet_used[fleet] + vehicles <= self.fleet_capacity[fleet]:
            self.fleet_used[fleet] += vehicles
            if use_container:
                return f"{fleet}-自有车(容器){self.fleet_used[fleet]}", base_cost * vehicles
            else:
                return f"{fleet}-自有车{self.fleet_used[fleet]}", base_cost * vehicles
        else:
            cost_multiplier = Config.REGULAR_VEHICLE['external_cost_multiplier']
            if use_container:
                return "外部容器车", base_cost * cost_multiplier * vehicles
            else:
                return "外部车", base_cost * cost_multiplier * vehicles
    
    def create_record(self, route, use_container, carrier, cost):
        """创建调度记录"""
        if use_container:
            capacity = Config.STANDARD_CONTAINER['capacity']
            load_time = Config.STANDARD_CONTAINER['loading_time']
        else:
            capacity = Config.REGULAR_VEHICLE['capacity']
            load_time = Config.REGULAR_VEHICLE['loading_time']
        
        vehicles = max(1, math.ceil(route['货量'] / capacity))  # 确保至少1辆车
        loading = route['货量'] / (vehicles * capacity)
        
        return {
            "线路编码": route['线路编码'],
            "发运时间": route['发运时间'],
            "承运车辆": carrier,
            "总包裹量": route['货量'],
            "使用标准容器": "是" if use_container else "否",
            "车辆需求数": vehicles,
            "装载率": loading,
            "成本": cost,
            "装卸时间(min)": load_time
        }

# ==================== 主调度引擎 ====================
class DispatchEngine:
    def __init__(self):
        self.df, self.fleet_capacity = load_data()
        self.scheduler = PriorityScheduler(self.df, self.fleet_capacity)
    
    def run(self):
        """执行完整调度流程"""
        # 1. 执行优化调度
        df_result = self.scheduler.schedule()
        
        # 2. 保存结果
        df_result.to_excel("结果表4_优化版.xlsx", index=False)
        
        # 3. 分析结果
        stats, target_results = self.analyze_results(df_result)
        
        # 4. 可视化
        self.visualize_results(df_result, stats, target_results)
        
        return df_result, stats
    
    def analyze_results(self, df_result):
        """分析调度结果"""
        total_volume = df_result['总包裹量'].sum()
        total_cost = df_result['成本'].sum()
        avg_loading = df_result['装载率'].mean()
        time_saved = ((Config.REGULAR_VEHICLE['loading_time'] - 
                      Config.STANDARD_CONTAINER['loading_time']) * 
                     (df_result['使用标准容器'] == "是").sum())
        
        stats = {
            '总包裹量': total_volume,
            '总成本': total_cost,
            '平均装载率': avg_loading,
            '总装卸时间节省(min)': time_saved
        }
        
        return stats, df_result.head(5)
    
    def visualize_results(self, df_result, stats, target_results):
        """生成可视化图表"""
        # 1. 目标线路调度详情表格
        plt.figure(figsize=(12, 3))
        ax = plt.subplot(111)
        ax.axis('off')
        
        table_data = target_results.copy()
        table_data['发运时间'] = table_data['发运时间'].dt.strftime('%Y-%m-%d %H:%M')
        table_data['成本'] = table_data['成本'].astype(int)
        
        table = plt.table(cellText=table_data.values,
                         colLabels=table_data.columns,
                         cellLoc='center',
                         loc='center',
                         colColours=['#f3f4f6']*len(table_data.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title("目标线路调度详情", pad=20, fontsize=12)
        plt.savefig("目标线路调度详情.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. 车辆装载量分布箱线图
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_result, y='总包裹量', x='使用标准容器', 
                   width=0.5, linewidth=2.5)
        
        plt.title('车辆装载量分布（按容器使用分类）', fontsize=14)
        plt.ylabel('包裹量', fontsize=12)
        plt.xlabel('使用标准容器', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig("车辆装载量分布.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # 3. 自有车与外部车使用对比
        plt.figure(figsize=(8, 5))
        df_result['车辆类型'] = df_result['承运车辆'].apply(lambda x: '自有车' if '自有车' in x else '外部车')
        vehicle_counts = df_result['车辆类型'].value_counts()
        
        sns.barplot(x=vehicle_counts.index, y=vehicle_counts.values, 
                   alpha=0.8, errwidth=0)
        
        plt.title('自有车与外部车使用次数对比', fontsize=14)
        plt.ylabel('使用次数', fontsize=12)
        plt.xlabel('车辆类型', fontsize=12)
        plt.savefig("车辆使用对比.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # 4. 发运时间分布柱状图
        plt.figure(figsize=(12, 6))
        df_result['发运小时'] = df_result['发运时间'].dt.hour
        hourly_counts = df_result.groupby('发运小时').size()
        
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, 
                   color='skyblue', alpha=0.8)
        
        plt.title('车辆发运时间分布', fontsize=14)
        plt.ylabel('发运次数', fontsize=12)
        plt.xlabel('发运小时', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        plt.savefig("发运时间分布.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # 5. 标准容器使用效果对比
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 装载率对比
        sns.boxplot(data=df_result, y='装载率', x='使用标准容器', 
                   ax=axes[0, 0], width=0.5)
        axes[0, 0].set_title('装载率对比', fontsize=12)
        axes[0, 0].set_ylabel('装载率', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 成本对比
        cost_comparison = df_result.groupby('使用标准容器')['成本'].sum()
        sns.barplot(x=cost_comparison.index, y=cost_comparison.values, 
                   ax=axes[0, 1], alpha=0.8)
        axes[0, 1].set_title('总成本对比', fontsize=12)
        axes[0, 1].set_ylabel('总成本', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 车辆需求数对比
        vehicle_comparison = df_result.groupby('使用标准容器')['车辆需求数'].sum()
        sns.barplot(x=vehicle_comparison.index, y=vehicle_comparison.values, 
                   ax=axes[1, 0], alpha=0.8)
        axes[1, 0].set_title('车辆需求数对比', fontsize=12)
        axes[1, 0].set_ylabel('车辆需求数', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 时间节省分析
        time_save = stats['总装卸时间节省(min)']
        axes[1, 1].bar(['时间节省'], [time_save], alpha=0.8)
        axes[1, 1].set_title(f'总装卸时间节省: {time_save}分钟', fontsize=12)
        axes[1, 1].set_ylabel('节省时间（分钟）', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('标准容器使用效果综合分析', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig("容器效果分析.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # 6. 关键指标仪表盘
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2)
        
        metrics = [
            ("总运输包裹量", f"{stats['总包裹量']:,}\n件", '#2e5984'),
            ("总运输成本", f"¥{stats['总成本']:,.0f}", '#c94c4c'),
            ("平均装载率", f"{stats['平均装载率']*100:.1f}%", '#3c763d'),
            ("时间节省", f"节省 {stats['总装卸时间节省(min)']} 分钟\n({stats['总装卸时间节省(min)']/60:.1f}小时)", '#6b8e23')
        ]
        
        for i, (title, value, color) in enumerate(metrics):
            ax = fig.add_subplot(gs[i//2, i%2])
            ax.set_title(title, fontsize=14)
            ax.axis('off')
            ax.text(0.5, 0.5, value, ha='center', va='center', 
                   fontsize=24, color=color)
        
        plt.suptitle('关键绩效指标仪表盘', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig("关键指标仪表盘.png", bbox_inches='tight', dpi=300)
        plt.close()

# ==================== 主执行流程 ====================
if __name__ == "__main__":
    engine = DispatchEngine()
    df_result, stats = engine.run()
    
    print("\n目标线路调度结果:")
    print(df_result.head(5).to_string(index=False))
    
    print("\n关键统计数据:")
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    print("\n生成的可视化图表:")
    print("1. 目标线路调度详情.png - 目标线路详细调度结果")
    print("2. 车辆装载量分布.png - 按容器分类的装载量分布")
    print("3. 车辆使用对比.png - 自有车与外部车使用次数对比")
    print("4. 发运时间分布.png - 各时段发运量分布")
    print("5. 容器效果分析.png - 标准容器使用效果综合分析")
    print("6. 关键指标仪表盘.png - 核心指标仪表板")
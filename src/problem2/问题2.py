import pandas as pd
import numpy as np
import math

# 读取输入数据
df_result1 = pd.read_excel('结果表1.xlsx')     # 每条线路预测货量
df_lineinfo = pd.read_excel('附件1.xlsx')      # 线路基本信息（含发运节点、所属车队、成本）
df_fleet = pd.read_excel('附件5.xlsx')         # 每个车队拥有的自有车辆数量
# 合并预测结果与线路信息
df_merge = pd.merge(df_result1, df_lineinfo, on='线路编码', how='left')

# 按题设容量：每车最大1000件
VEHICLE_CAPACITY = 1000
df_merge['车辆需求数'] = df_merge['货量'].apply(lambda x: math.ceil(x / VEHICLE_CAPACITY))

# 预测发运日期：2024-12-16（题设）
base_date = pd.to_datetime('2024-12-16')

# 发运节点是time类型 → 合成datetime
df_merge['预计发运时间'] = df_merge['发运节点'].apply(
    lambda t: pd.Timestamp.combine(base_date, t) if pd.notna(t) else pd.NaT
)
# 构建车队车辆数状态表
fleet_capacity = dict(zip(df_fleet['车队编码'], df_fleet['自有车数量']))
fleet_use = {fleet: 0 for fleet in fleet_capacity}

dispatch_records = []

for _, row in df_merge.iterrows():
    route = row['线路编码']
    num_vehicles = row['车辆需求数']
    fleet = row['车队编码']
    dispatch_time = row['预计发运时间']

    for i in range(num_vehicles):
        if fleet_use[fleet] < fleet_capacity.get(fleet, 0):
            fleet_use[fleet] += 1
            carrier = f'{fleet}-自有车{fleet_use[fleet]}'
        else:
            carrier = '外部'
        
        dispatch_records.append({
            '线路编码': route,
            '预计发运时间': dispatch_time,
            '承运车辆': carrier
        })
import pandas as pd
import numpy as np
from itertools import combinations
from math import ceil

# === 模型参数 ===
Q = 1000             # 每辆车最大载重
OWN_COST = 100       # 自有车固定成本
MAX_COMBINE = 3      # 每辆车最多串点数
TIME_WINDOW = 30     # 可接受的拼点时间差（分钟）

# === 数据读取 ===
df_demand = pd.read_excel("结果表1.xlsx")        # 问题一结果
df_lineinfo = pd.read_excel("附件1.xlsx")        # 发运节点、车队、外部成本
df_fleet = pd.read_excel("附件5.xlsx")           # 各车队自有车辆数

# === 数据准备 ===
df = pd.merge(df_demand, df_lineinfo, on="线路编码", how="left")
df['发运时间'] = pd.to_datetime("2024-12-16") + pd.to_timedelta(df['发运节点'].astype(str))
df['优先级'] = df['货量'] / (df['发运时间'] - pd.to_datetime("2024-12-16")).dt.total_seconds()

fleet_capacity = dict(zip(df_fleet['车队编码'], df_fleet['自有车数量']))
fleet_used = {fleet: 0 for fleet in fleet_capacity}

dispatch_records = []
used_idx = set()
df = df.sort_values("优先级", ascending=False).reset_index(drop=True)

for idx, row in df.iterrows():
    if idx in used_idx:
        continue

    base_time = row['发运时间']
    base_site = row['起始场地']
    base_fleet = row['车队编码']
    base_volume = row['货量']
    base_cost = row['外部承运商成本']

    # 候选线路（同起点 + 时间差在30分钟以内）
    candidates = df[
        (df['起始场地'] == base_site) &
        (~df.index.isin(used_idx)) &
        (abs((df['发运时间'] - base_time).dt.total_seconds()) <= TIME_WINDOW * 60)
    ].head(15)  # 控制组合规模防爆炸

    best_combo = [row]
    best_volume = base_volume
    best_cost = base_cost
    best_indices = [idx]

    for r in range(2, MAX_COMBINE + 1):
        for combo in combinations(candidates.index, r):
            lines = df.loc[list(combo)]
            total_volume = lines['货量'].sum()
            if total_volume <= Q:
                cost = lines['外部承运商成本'].max()
                if total_volume > best_volume:
                    best_combo = lines.to_dict('records')
                    best_volume = total_volume
                    best_cost = cost
                    best_indices = list(combo)

    # 自有车优先
    if fleet_used[base_fleet] < fleet_capacity.get(base_fleet, 0):
        fleet_used[base_fleet] += 1
        carrier = f"{base_fleet}-自有车{fleet_used[base_fleet]}"
        cost = OWN_COST
    else:
        carrier = "外部"
        cost = best_cost

    dispatch_time = min([pd.to_datetime(r['发运时间']) for r in best_combo])

    dispatch_records.append({
        "线路编码": " + ".join([r['线路编码'] for r in best_combo]),
        "预计发运时间": dispatch_time,
        "承运车辆": carrier,
        "总包裹量": best_volume,
        "成本": cost
    })

    used_idx.update(best_indices)

# === 结果表3输出 ===
df_result3 = pd.DataFrame(dispatch_records)
df_result3.to_excel("结果表3.xlsx", index=False)

# === 多目标指标评估 ===
actual_vehicle_count = df_result3['总包裹量'].apply(lambda x: ceil(x / Q)).sum()
total_volume = df_result3['总包裹量'].sum()
true_load_rate = total_volume / (actual_vehicle_count * Q)

own_used = sum("外部" not in x for x in df_result3['承运车辆'])
own_total = sum(fleet_capacity.values())
total_cost = df_result3['成本'].sum()

print("✅ 成功输出结果表3.xlsx（优化调度结果）")
print(f"🚚 实际车辆数（按1000容量）：{actual_vehicle_count}")
print(f"✅ 自有车使用率：{own_used}/{own_total} = {own_used / own_total:.2%}")
print(f"📦 车辆平均装载率：{true_load_rate:.2%}")
print(f"💰 总运输成本：¥{total_cost}")

import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（防止中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图1：各时段车辆发运数量分布
plt.figure(figsize=(10, 5))
dispatch_hour = pd.to_datetime(df_result3['预计发运时间']).dt.hour
sns.histplot(dispatch_hour, bins=range(0, 25), kde=False, color="skyblue")
plt.title("车辆发运时间分布")
plt.xlabel("发运小时")
plt.ylabel("车辆数")
plt.tight_layout()
plt.savefig("图1_发运时间分布图.png")

# 图2：自有车 vs 外部车 使用分布
plt.figure(figsize=(6, 4))
car_types = df_result3['承运车辆'].astype(str).apply(lambda x: '自有车' if '自有车' in x else '外部')
sns.countplot(x=car_types, palette="Set2")
plt.title("自有车与外部车使用次数")
plt.xlabel("车辆类型")
plt.ylabel("使用次数")
plt.tight_layout()
plt.savefig("图2_车辆类型使用统计.png")

# 图3：各车辆装载量箱线图
plt.figure(figsize=(8, 5))
sns.boxplot(y=df_result3['总包裹量'], color="orange")
plt.title("车辆装载量分布（箱线图）")
plt.ylabel("包裹量")
plt.tight_layout()
plt.savefig("图3_车辆装载量箱线图.png")

# 图4：包裹量前20调度任务
top_dispatch = df_result3.sort_values('总包裹量', ascending=False).head(20)
plt.figure(figsize=(12, 6))
sns.barplot(x="总包裹量", y="线路编码", data=top_dispatch, palette="viridis")
plt.title("包裹量前20的调度任务")
plt.xlabel("包裹量")
plt.ylabel("线路组合")
plt.tight_layout()
plt.savefig("图4_包裹量前20调度任务.png")

print("✅ 已导出可视化图片：图1_发运时间分布图.png, 图2_车辆类型使用统计.png, 图3_车辆装载量箱线图.png, 图4_包裹量前20调度任务.png")
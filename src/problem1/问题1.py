import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime, time
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===== 第一步：读取数据 =====
df_actual = pd.read_excel(r'D:\数学建模比赛\D题\附件\附件2.xlsx')  # 历史10分钟包裹量
df_known = pd.read_excel(r'D:\数学建模比赛\D题\附件\附件3.xlsx')   # 预知日总量
df_actual['日期'] = pd.to_datetime(df_actual['日期'])
df_known['日期'] = pd.to_datetime(df_known['日期'])
df_known.rename(columns={'包裹量': '预知日总量'}, inplace=True)

# ===== 第二步：构造真实日总量 =====
df_day_total = df_actual.groupby(['线路编码', '日期'])['包裹量'].sum().reset_index()
df_day_total.rename(columns={'包裹量': '实际日总量'}, inplace=True)

# ===== 第三步：训练集 + 特征构造 =====
df = pd.merge(df_known, df_day_total, on=['线路编码', '日期'], how='inner')
df = df.sort_values(['线路编码', '日期'])
df['weekday'] = df['日期'].dt.weekday

def get_rolling_features(x):
    return pd.DataFrame({
        'roll3_mean': x['实际日总量'].shift(1).rolling(3).mean(),
        'roll3_std': x['实际日总量'].shift(1).rolling(3).std(),
        'roll7_mean': x['实际日总量'].shift(1).rolling(7).mean(),
        'roll7_std': x['实际日总量'].shift(1).rolling(7).std()
    })

df_roll = df.groupby('线路编码').apply(get_rolling_features).reset_index(drop=True)
df = pd.concat([df, df_roll], axis=1)
df.dropna(inplace=True)

# ===== 第四步：XGBoost 模型训练 =====
features = ['预知日总量', 'weekday', 'roll3_mean', 'roll3_std', 'roll7_mean', 'roll7_std']
X = df[features]
y = df['实际日总量']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)
params = {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.1}
model = xgb.train(params, dtrain, num_boost_round=200,
                  evals=[(dtrain, 'train'), (dval, 'val')],
                  early_stopping_rounds=20, verbose_eval=False)

# ===== 第五步：预测未来一天（12月16日）的日总量 =====
future_date = pd.to_datetime('2024-12-16')
df_future = df_known[df_known['日期'] == future_date].copy()
df_future['weekday'] = df_future['日期'].dt.weekday

rolling_features = df.groupby('线路编码')[['roll3_mean', 'roll3_std', 'roll7_mean', 'roll7_std']].last().reset_index()
df_future = df_future.merge(rolling_features, on='线路编码', how='left')
df_future.dropna(subset=['roll3_mean'], inplace=True)
df_future['预测日总量'] = model.predict(xgb.DMatrix(df_future[features]))
df_future['货量'] = df_future['预测日总量'].round().astype(int)

# ===== 第六步：保存结果表1（预测日总量） =====
df_result1 = df_future[['线路编码', '日期', '货量']].sort_values(['线路编码', '日期'])
df_result1.to_excel('结果表1.xlsx', index=False)

# ===== 第七步：构造历史10分钟占比结构 =====
df_actual = pd.merge(df_actual, df_day_total, on=['线路编码', '日期'])
df_actual['占比'] = df_actual['包裹量'] / df_actual['实际日总量']
dist_ratio = df_actual.groupby(['线路编码', '分钟起始'])['占比'].mean().reset_index()

# ===== 第八步：拆分预测日总量为10分钟预测 =====
df_split = pd.merge(df_future[['线路编码', '预测日总量']], dist_ratio, on='线路编码', how='left')
df_split['包裹量'] = (df_split['预测日总量'] * df_split['占比']).round().astype(int)
df_split['日期'] = future_date
df_result2 = df_split[['线路编码', '日期', '分钟起始', '包裹量']].sort_values(['线路编码', '日期', '分钟起始'])
df_result2.to_excel('结果表2.xlsx', index=False)

# ===== 第九步：提取论文要求线路预测结果 =====
target_lines = ['场地3 - 站点83 - 0600', '场地3 - 站点83 - 1400']

# 日总量结果（结果表1）
df_paper1 = df_result1[df_result1['线路编码'].isin(target_lines)]

# 每10分钟拆分结果（结果表2）
df_paper2 = df_result2[df_result2['线路编码'].isin(target_lines)]

# 显示
print("✅ 论文线路日总量预测：")
print(df_paper1)

print("\n✅ 论文线路10分钟拆分预测（前10条）：")
print(df_paper2.head(10))

# ===== 第十步：可视化展示 =====
# 1. 模型预测效果评估图
y_pred = model.predict(xgb.DMatrix(X_val))
plt.figure(figsize=(8, 5))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("实际日总量")
plt.ylabel("预测日总量")
plt.title("XGBoost模型预测效果对比")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 预测货量分布图
plt.figure(figsize=(10, 4))
plt.hist(df_result1['货量'], bins=30, color='skyblue', edgecolor='black')
plt.title("所有线路预测日总量分布")
plt.xlabel("预测货量")
plt.ylabel("频次")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. 0600和1400线路的10分钟包裹量预测图
for line, color in zip(target_lines, ['blue', 'red']):
    df_line = df_result2[df_result2['线路编码'] == line].copy()
    if not df_line.empty:
        df_line = df_line.sort_values('分钟起始')
        df_line['时间标签'] = df_line['分钟起始'].apply(lambda t: t.strftime('%H:%M') if isinstance(t, time) else str(t))
        
        plt.figure(figsize=(12, 5))
        plt.plot(df_line['时间标签'], df_line['包裹量'], marker='o', color=color)
        plt.title(f"线路 {line} 的10分钟包裹量预测")
        plt.xlabel("时间间隔")
        plt.ylabel("预测包裹量")
        plt.xticks(rotation=45, fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ 警告：线路 {line} 在预测结果中没有数据")
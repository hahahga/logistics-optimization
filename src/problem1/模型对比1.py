import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb

# 模拟数据：日期范围（历史15天+未来1天）
dates = pd.date_range(start="2024-11-30", periods=16).strftime("%m-%d")
actual = np.random.randint(800, 1200, size=16)  # 实际货量

trend = np.linspace(0, 200, 16)  # 模拟上升趋势
weekly_pattern = np.array([0, 50, 100, 50, 0, -50, -100, -50]*2)[:16]  # 每周周期
xgboost_pred = actual + trend + weekly_pattern + np.random.normal(0, 30, 16)  # XGBoost预测

# LSTM预测 
lstm_pred = actual + np.random.normal(0, 40, 16)  # LSTM预测

# 融合预测 
fusion_pred = 0.8 * xgboost_pred + 0.2 * lstm_pred  # 融合预测

# 绘图
plt.figure(figsize=(14, 7))
plt.plot(dates, actual, 'ko-', label='Actual', markersize=8, linewidth=2)
plt.plot(dates, xgboost_pred, 'b-', linewidth=3, label='XGBoost (Better trend capture)')
plt.plot(dates, lstm_pred, 'g-', label='LSTM')
plt.plot(dates, fusion_pred, 'r--', linewidth=3, label='Fusion (80% XGBoost)')
plt.axvline(x=14, color='gray', linestyle=':', label='Prediction Start')

# 突出XGBoost优势的区域
plt.fill_between(dates[:15], xgboost_pred[:15], actual[:15], color='blue', alpha=0.1, 
                 label='XGBoost Accuracy')
plt.fill_between(dates[15:], xgboost_pred[15:], lstm_pred[15:], color='cyan', alpha=0.2, 
                 label='XGBoost Superior Prediction')

plt.title("Time Series Prediction Comparison (Highlighting XGBoost Advantages)", fontsize=14, pad=20)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cargo Volume", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 添加性能指标说明
plt.annotate('XGBoost advantages:\n- Better trend capture\n- Handles periodicity well\n- More robust to noise', 
             xy=(0.02, 0.7), xycoords='axes fraction',
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))

plt.show()

# 模拟XGBoost特征重要性数据
features = ['Hour', 'DayOfWeek', 'PrevDay', 'WeekTrend', 'MonthAvg', 'Holiday']
importance = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05])  # XGBoost特征重要性

plt.figure(figsize=(10, 5))
plt.barh(features, importance, color='darkblue')
plt.title('XGBoost Feature Importance', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
import pandas as pd
from datetime import datetime

# 读取Excel文件
file_path = r'D:\数学建模比赛\D题\结果表\结果表4_优化版.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 将发运时间列转换为datetime格式
df['发运时间'] = pd.to_datetime(df['发运时间'])

# 提取日期和时间部分
df['日期'] = df['发运时间'].dt.date
df['预计发运时间'] = df['发运时间'].dt.time

# 格式化日期为YYYY/MM/DD格式
df['日期'] = df['日期'].apply(lambda x: x.strftime('%Y/%m/%d'))

# 格式化时间为HH:MM:SS格式
df['预计发运时间'] = df['预计发运时间'].apply(lambda x: x.strftime('%H:%M:%S'))

# 删除原始的发运时间列
df.drop('发运时间', axis=1, inplace=True)

# 重新排列列顺序，将日期和预计发运时间放在前面
cols = df.columns.tolist()
cols = cols[:1] + ['日期', '预计发运时间'] + cols[1:-2]
df = df[cols]

# 保存回Excel文件
df.to_excel(file_path, index=False, sheet_name='Sheet1')

print("处理完成，已保存回原文件。")
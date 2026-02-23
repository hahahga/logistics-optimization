import pandas as pd

# 读取Excel文件（假设表名为Sheet1）
input_file = r'D:\数学建模比赛\D题\结果表\结果表1.xlsx'
df = pd.read_excel(input_file, sheet_name='Sheet1')

# 检查日期列是否存在（假设列名是"日期"）
if '日期' in df.columns:
    # 将日期列转为datetime格式（确保时间部分能被识别）
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 方法1：直接取date部分（自动去除时间）
    df['日期'] = df['日期'].dt.date
    
    # 方法2：或转为字符串后截取前10字符（更保险）
    # df['日期'] = df['日期'].astype(str).str[:10]
    
    # 保存回原文件（其他所有数据不变）
    df.to_excel(input_file, index=False, sheet_name='Sheet1')
    print(f"处理完成！已清除时间部分，文件保存到: {input_file}")
else:
    print("错误：未找到名为'日期'的列，请检查列名是否正确")
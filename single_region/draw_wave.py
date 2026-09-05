import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1️⃣ 读取波长标签 (请替换 `csv_path` 为正确路径)
csv_path = "../wavenumbers.csv"  # 确保路径正确
acs_ori = pd.read_csv(csv_path, encoding='latin-1')
rows = [row for row in acs_ori]
x_label = rows[9:710]  # 701个波长标签 (string)

# 2️⃣ 自动定位最新的 shap.json (从最近一次 shap_analysis.py 输出)
candidates = sorted(
    glob.glob("mose2_*/test/shap.json"),
    key=os.path.getmtime,
    reverse=True,
)
assert candidates, "❌ 找不到 mose2_*/test/shap.json，请先跑 python shap_analysis.py"
json_path = candidates[0]
region = json_path.split(os.sep)[0].replace("mose2_", "")
print(f"📂 使用 SHAP 数据: {json_path}")
with open(json_path, "r") as f:
    vector_701 = json.load(f)  # 解析 JSON 文件

# 3️⃣ 确保数据格式正确
assert isinstance(vector_701, list), "JSON 数据必须是列表格式"
assert len(vector_701) == 701, "数据长度不是 701！请检查 JSON 文件格式。"
vector_701 = np.array(vector_701)  # 转换为 NumPy 数组

# 4️⃣ 选择每隔 50 个标签显示一次，避免横坐标过密
xticks_interval = 50
xticks_positions = np.arange(0, 701, xticks_interval)
xticks_labels = [x_label[i] for i in xticks_positions]

# 5️⃣ 画折线图（使用 salmon 颜色，线条加粗）
plt.figure(figsize=(16, 5))
plt.plot(range(701), vector_701, color='salmon', linewidth=2,alpha=0.5)  # 线条加粗，颜色为 salmon
plt.fill_between(range(701), vector_701, color='salmon', alpha=0.5)  # 下面填充颜色

# 设置横坐标刻度
plt.xticks(xticks_positions, xticks_labels, rotation=45, ha='right')

# 添加标签和标题

# 添加标签和标题
# plt.xlabel("Wavelength")
# plt.ylabel("Value")
# plt.title("Spectral Data Visualization")

# 添加网格
plt.grid(True, linestyle='--', alpha=0.5)

# 6️⃣ 保存图像
plt.tight_layout()
output_name = f"spectral_plot_salmon_{region}.png"
plt.savefig(output_name, dpi=300)
plt.close()

print(f"✅ 折线图已保存为 {output_name}")

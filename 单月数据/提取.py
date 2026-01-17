import csv
from datetime import datetime

# 输入文件路径
input_file = r"E:\铁科院项目\对比模型\单月数据\3007007.csv"
# 输出文件路径
output_file = r"E:\铁科院项目\对比模型\单月数据\3007007_x.csv"

# 打开输入文件并读取数据
with open(input_file, mode="r", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    # 获取表头
    header = next(reader)
    # 筛选四月份的数据
    april_data = [row for row in reader if row[0].startswith("2024-04")]

# 将筛选后的数据写入输出文件
with open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile)
    # 写入表头
    writer.writerow(header)
    # 写入四月份的数据
    writer.writerows(april_data)

print(f"四月份的数据已保存到 {output_file}")

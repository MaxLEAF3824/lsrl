import gzip
import torch
import glob
import os

# 假设你的文件都在这个目录下，且命名符合格式
file_pattern = "./optimization_histories/optimized_embeds_*_node*.pt.gz"
output_file = "./optimization_histories/optimized_embeds_full.pt.gz"

all_data = []
for f_name in sorted(glob.glob(file_pattern)):
    print(f"正在读取: {f_name}")
    with gzip.open(f_name, 'rb') as f:
        node_data = torch.load(f)
        all_data.extend(node_data)

print(f"合并完成！总计样本数: {len(all_data)}")
with gzip.open(output_file, 'wb') as f:
    torch.save(all_data, f)
print(f"已保存完整全量数据至: {output_file}")
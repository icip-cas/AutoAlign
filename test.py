import torch
import torch_npu
import math

# 生成随机数据，并发送到npu
q = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
k = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
v = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
scale = 1 / math.sqrt(128.0)

# 调用IFA算子
out = torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale)

# 打印输出结果
print(out)


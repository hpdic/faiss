import torch
import time

# 1. 检查设备
if not torch.cuda.is_available():
    print("❌ 没检测到 GPU！")
    exit()

device = torch.device("cuda")
print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")
print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 2. 准备数据 (搞两个巨大的矩阵，吃满 Tensor Core)
# 10000x10000 的矩阵，FP16 模式 (A100 擅长这个)
N = 10000
dtype = torch.float16 

print(f"\n🚀 开始测试矩阵乘法 ({N}x{N}, FP16)...")
a = torch.randn(N, N, device=device, dtype=dtype)
b = torch.randn(N, N, device=device, dtype=dtype)

# 3. 预热 (Warm up)
for _ in range(5):
    _ = torch.matmul(a, b)
torch.cuda.synchronize()

# 4. 正式测速
start_time = time.time()
num_iters = 100
for _ in range(num_iters):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
end_time = time.time()

avg_time = (end_time - start_time) / num_iters
tflops = (2 * N**3) / (avg_time * 1e12)

print(f"✅ 完成！平均耗时: {avg_time*1000:.2f} ms")
print(f"⚡ 估算性能: {tflops:.2f} TFLOPS")

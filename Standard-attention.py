import torch
import time
import matplotlib.pyplot as plt

Batch = 8
Heads = 16
Dim_head = 64
Seq_lens = [32*(2**x) for x in range(9)]
Runtime = [0 for _ in range(9)]
device = "cuda"
iterations = 10

seeds = [42, 43, 44]

# warmup CUDA so kernel compilation doesn't skew the first measurement (but still it does)
_w = torch.randn(1, 1, 32, Dim_head, dtype=torch.float16, device=device)
torch.matmul(_w, _w.transpose(-2, -1))
torch.cuda.synchronize()
del _w

for idx in range(len(Seq_lens)):

    tensors = []

    for s in seeds:
        torch.manual_seed(s)
        tensors.append(torch.randn(Batch, Heads, Seq_lens[idx], Dim_head, dtype=torch.float16, device=device))

    Q, K, V = tensors

    K = K.transpose(-2, -1)

    torch.cuda.synchronize()
    time_start = time.time()
    for _ in range(iterations):
        attention_scores = torch.softmax(torch.matmul(Q, K) / (Dim_head ** 0.5), dim=-1)
        O = torch.matmul(attention_scores, V)
    torch.cuda.synchronize()
    time_end = time.time()

    avg_time = (time_end - time_start) / iterations
    Runtime[idx] = avg_time


plt.plot(Seq_lens, Runtime, marker='o')
plt.xlabel("Sequence Length N")
plt.ylabel("Average Runtime (s)")
plt.title("Standard Attention Runtime vs N")
plt.grid(True)
plt.tight_layout()
plt.savefig("standard_attention_runtime.png")
plt.show()






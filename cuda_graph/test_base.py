import torch
import time

device = torch.device("cuda")
torch.manual_seed(777)

def benchmark(title, func, warmup=10, repeat=90):
    print(f"==={title}===")
    torch.cuda.empty_cache()

    # warmup
    for _ in range(warmup):
        func()

    # use cuda event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings = []

    for _ in range(repeat):
        torch.cuda.synchronize()
        start_event.record()
        func()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        timings.append(elapsed_time)

    avg_time = sum(timings)/repeat
    mem_usage = torch.cuda.max_memory_allocated() / 1024**2
    print(f"average time per Iteration: {avg_time:.6f} ms | Memory: {mem_usage:.1f} MB")

# regular
def run_baseline():
    x = torch.randn(256, 768, device=device)
    for _ in range(15):
        x = x.cos().exp().sqrt()
        x *= torch.sigmoid(x) + 0.01
        x[x<0] = 0.99


benchmark("Optimized with CUDA Graph", run_baseline)

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


#use cuda graph
static_input = torch.randn(256, 768, device=device).float()
static_out = torch.zeros_like(static_input)
tmp_in = static_input.clone()
tmp_in = tmp_in.cos().exp().sqrt()
tmp_in *= torch.sigmoid(tmp_in) + 0.001
tmp_in[tmp_in < 0] = 998
static_input.copy_(tmp_in)

# create cuda graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    updated_val = static_input.cos().exp().sqrt()
    updated_val *= torch.sigmoid(updated_val) + 0.005
    updated_val[updated_val < 0] = 997
    static_out.copy_(updated_val)
    
def graph_run():
    for _ in range(15):
        graph.replay()

benchmark("Optimized with CUDA Graph", graph_run)

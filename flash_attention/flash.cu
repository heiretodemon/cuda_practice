#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>

const int BLOCK_SIZE = 64;

__global__ void flash_attention_forward_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int seq_len,
    const int d_model,
    const float scale) {
    
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    __shared__ half Qi[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ half Kj[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Sij[BLOCK_SIZE][BLOCK_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bid = blockIdx.x;
    
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc_o[BLOCK_SIZE] = {0.0f};
    
    for (int j = 0; j < num_blocks; ++j) {
        // Load Q and K blocks
        int q_row = bid * BLOCK_SIZE + tx;
        int k_col = j * BLOCK_SIZE + ty;
        if (q_row < seq_len && ty < d_model) {
            Qi[tx][ty] = __half2float(Q[q_row * d_model + ty]) * scale;
        } else {
            Qi[tx][ty] = 0.0f;
        }
        if (k_col < seq_len && tx < d_model) {
            Kj[ty][tx] = __half2float(K[k_col * d_model + tx]);
        } else {
            Kj[ty][tx] = 0.0f;
        }
        __syncthreads();
        
        // Compute Sij = Qi * Kj^T
        float sum = 0.0f;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += __half2float(Qi[tx][k] * Kj[ty][k]);
        }
        Sij[tx][ty] = sum;
        __syncthreads();
        
        // Compute local max and update
        float local_max = -INFINITY;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            local_max = max(local_max, Sij[tx][k]);
        }
        float new_max = max(max_val, local_max);
        
        // Compute exp sum
        float exp_sum = 0.0f;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            exp_sum += exp(Sij[tx][k] - new_max);
        }
        
        // Update accumulators
        sum_exp = sum_exp * exp(max_val - new_max) + exp_sum;
        max_val = new_max;
        
        // Accumulate output
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float p = exp(Sij[tx][k] - max_val) / sum_exp;
            int v_row = j * BLOCK_SIZE + k;
            if (v_row < seq_len && ty < d_model) {
                acc_o[ty] += p * __half2float(V[v_row * d_model + ty]);
            }
        }
        __syncthreads();
    }
    
    // Write output
    int row = bid * BLOCK_SIZE + tx;
    if (row < seq_len && ty < d_model) {
        O[row * d_model + ty] = __float2half(acc_o[ty]);
    }
}

void flash_attention_forward(const half* Q, const half* K, const half* V, half* O, 
                            int seq_len, int d_model) {
    const float scale = 1.0f / sqrtf(d_model);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(num_blocks);
    flash_attention_forward_kernel<<<grid, block, 0>>>(Q, K, V, O, seq_len, d_model, scale);
}

// ...其余部分（generate_random_half, reference_attention, main）保持不变...

// 生成随机半精度数据
void generate_random_half(half* data, int size) {
    for (int i = 0; i < size; ++i) {
        float val = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
        data[i] = __float2half(val);
    }
}

// CPU参考实现（PyTorch）
torch::Tensor reference_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto scores = torch::matmul(Q, K.transpose(-1, -2));
    scores = scores / torch::sqrt(torch::tensor(Q.size(-1), torch::kFloat));
    auto attn = torch::softmax(scores, -1);
    return torch::matmul(attn, V);
}

int main() {
    const int seq_len = 64;
    const int d_model = 64;
    const int total = seq_len * d_model;
    
    // 主机内存分配
    half *h_Q = new half[total];
    half *h_K = new half[total];
    half *h_V = new half[total];
    half *h_O = new half[total];
    
    // 生成随机数据
    generate_random_half(h_Q, total);
    generate_random_half(h_K, total);
    generate_random_half(h_V, total);

    // 设备内存分配
    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, total * sizeof(half));
    cudaMalloc(&d_K, total * sizeof(half));
    cudaMalloc(&d_V, total * sizeof(half));
    cudaMalloc(&d_O, total * sizeof(half));

    // 数据拷贝到设备
    cudaMemcpy(d_Q, h_Q, total * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total * sizeof(half), cudaMemcpyHostToDevice);

    // 执行FlashAttention
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    flash_attention_forward(d_Q, d_K, d_V, d_O, seq_len, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel time: " << milliseconds << " ms\n";

    // 拷贝结果回主机
    cudaMemcpy(h_O, d_O, total * sizeof(half), cudaMemcpyDeviceToHost);

    // 转换为PyTorch Tensor进行验证
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor t_Q = torch::from_blob(h_Q, {seq_len, d_model}, options).to(torch::kCUDA);
    torch::Tensor t_K = torch::from_blob(h_K, {seq_len, d_model}, options).to(torch::kCUDA);
    torch::Tensor t_V = torch::from_blob(h_V, {seq_len, d_model}, options).to(torch::kCUDA);
    
    torch::Tensor ref_O = reference_attention(t_Q, t_K, t_V).cpu();
    
    // 转换为float比较
    float max_error = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float val = __half2float(h_O[i * d_model + j]);
            float ref_val = ref_O[i][j].item<float>();
            max_error = fmaxf(max_error, fabsf(val - ref_val));
        }
    }
    std::cout << "Max absolute error: " << max_error << std::endl;

    // 清理资源
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return (max_error < 1e-2) ? 0 : 1; // 误差阈值设为0.01
}
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vecAdd(int n, float *a, float *b, float *c){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

int main(){
    int N = 1024;
    float *a, *b, *c;
    float *devA, *devB, *devC;
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c = (float*)malloc(N * sizeof(float));
    cudaMalloc(&devA, N*sizeof(float));
    cudaMalloc(&devB, N*sizeof(float));
    cudaMalloc(&devC, N*sizeof(float));

    memset(c, 0, N*sizeof(float));
    // memset(a, 1, N*sizeof(float));
    // memset(b, 2, N*sizeof(float));
    cudaMemcpy(devA, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, N*sizeof(float), cudaMemcpyHostToDevice);

    vecAdd <<< (N+255)/256, 256 >>> (N, devA, devB, devC);
    cudaMemcpy(c, devC, N*sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i=0;i<10;i++){
    //     printf("%f, %f", a[0], a[1]);
    //     printf("%f\n", c[i]);
    // }
    return 0;
}

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void vector_add(const int* a, const int* b, int* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];   
}

int main() {

    const int N = 1024;
    const int bytes = N * sizeof(int);

    int* host_a = (int*)malloc(bytes);
    int* host_b = (int*)malloc(bytes);
    int* host_c = (int*)malloc(bytes);

    for (int i = 0; i < N; i++)
    {
        host_a[i] = i;
        host_b[i] = 10 * i;
    }    

    int* device_a;
    int* device_b;
    int* device_c;

    cudaMalloc(&device_a, bytes);
    cudaMalloc(&device_b, bytes);
    cudaMalloc(&device_c, bytes);

    cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<numBlocks, threadsPerBlock>>>(device_a, device_b, device_c, N);

    cudaDeviceSynchronize();

    cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        cout << host_a[i] << " + " << host_b[i] << " = " << host_c[i] << endl;

    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}


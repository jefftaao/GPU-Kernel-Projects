#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void color_to_grayscale_kernel(const char* R, const char* G, const char* B, char* G, int width, int height) {
    //Multidimensional Indexing:
    //need to find row and column, and use both for indexing
    //row: use y dimension
    //col: use x dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
}

void color_to_grayscale(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int length)
{
    unsigned int bytes = width * height * sizeof(unsigned char);
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**) &red_d, bytes);
    cudaMalloc((void**) &green_d, width * height * bytes);
    cudaMalloc((void**) &blue_d, width * height * bytes);
    cudaMalloc((void**) &gray_d, width * height * bytes);

    cudaMemcpy(red_d, red, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, bytes, cudaMemcpyHostToDevice);


    //Using dim3 for 2D, by default dim.z = 1
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (height + numBlocksPerThread.y - 1) / numBlocksPerThread.y);
    
    color_to_grayscale_kernel<<<numBlocks, numThreadsPerBlock>>>color_to_grayscale_kernel(red_d, green_d, blue_d, gray_d, width, height);
    cudaMemcpy(gray, gray_d, bytes, cudaMemcpyDeviceToHost);


    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);


}



int main() {

    

}


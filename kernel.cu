#include <stdio.h>
#include <cuda_runtime.h>

struct CudaMatrix
{
    float *data;
};

extern "C"
{
    int init_cuda_matrix(CudaMatrix *cuda_matrix, const float *data, int data_size)
    {
        cudaMalloc((void **)&cuda_matrix->data, data_size * sizeof(float));
        cudaMemcpy(cuda_matrix->data, data, data_size * sizeof(float), cudaMemcpyHostToDevice);
        return 0;
    }

    int free_cuda_matrix(CudaMatrix *cuda_matrix)
    {
        cudaFree(cuda_matrix->data);
        return 0;
    }
}

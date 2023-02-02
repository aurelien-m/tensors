#include <stdio.h>
#include <cuda_runtime.h>

struct CudaMatrix
{
    float *data;
    int *dims;
    int ndims;
};

__global__ void multiply_2d_matrices(float *a, float *b, float *out, int n, int m, int p)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i >= m) || (j >= p))
        return;

    float acc_sum = 0.0;
    for (int k = 0; k < n; k++)
    {
        acc_sum += a[i * n + k] * b[k * p + j];
    }
    out[i * p + j] = acc_sum;
}

extern "C"
{
    int init_cuda_matrix(CudaMatrix *cuda_matrix, const float *data, int data_size)
    {
        cudaMalloc((void **)&cuda_matrix->data, data_size * sizeof(float));
        cudaMemcpy(cuda_matrix->data, data, data_size * sizeof(float), cudaMemcpyHostToDevice);

        cuda_matrix->dims = (int *)malloc(1 * sizeof(int));
        cuda_matrix->ndims = 1;

        return 0;
    }

    int free_cuda_matrix(CudaMatrix *cuda_matrix)
    {
        cudaFree(cuda_matrix->data);
        free(cuda_matrix->dims);

        return 0;
    }
}

extern "C"
{
    int set_dims(CudaMatrix *cuda_matrix, const int *dims, int ndims)
    {
        if (cuda_matrix->dims != NULL)
            free(cuda_matrix->dims);
        
        cuda_matrix->dims = (int *)malloc(ndims * sizeof(int));
        cuda_matrix->ndims = ndims;

        for (int i = 0; i < ndims; i++)
        {
            cuda_matrix->dims[i] = dims[i];
        }

        return 0;
    }
}

extern "C"
{
    int multiply_cuda_matrix(CudaMatrix *a, CudaMatrix *b, CudaMatrix *out)
    {
        if (a->ndims != 2 || b->ndims != 2)
        {
            printf("Only 2D matrices are supported.\n");
            return 1;
        }

        int n = a->dims[0];
        int m = a->dims[1];
        int p = b->dims[1];

        // printf("Multiplying %d x %d matrix with %d x %d matrix.\n", n, m, m, p);
        float *data;
        cudaMalloc((void **)&data, n * p * sizeof(float));

        dim3 threads_per_block(32, 32);
        dim3 blocks_per_grid(1, 1);
        blocks_per_grid.x = std::ceil(static_cast<double>(p) /
                                      static_cast<double>(threads_per_block.x));
        blocks_per_grid.y = std::ceil(static_cast<double>(m) /
                                      static_cast<double>(threads_per_block.y));

        multiply_2d_matrices<<<blocks_per_grid, threads_per_block>>>(a->data, b->data, data, n, m, p);
        cudaDeviceSynchronize();

        out->data = data;
        out->dims = (int *)malloc(2 * sizeof(int));
        out->dims[0] = n;
        out->dims[1] = p;
        out->ndims = 2;

        return 0;
    }

    int print_cuda_matrix(const CudaMatrix *cuda_matrix)
    {
        if (cuda_matrix->ndims != 2)
        {
            printf("Only 2D matrices are supported.\n");
            return 1;
        }

        int n = cuda_matrix->dims[0];
        int p = cuda_matrix->dims[1];

        printf("Matrix %d x %d\n", n, p);

        float *out_data = (float *)malloc(n * p * sizeof(float));
        cudaMemcpy(out_data, cuda_matrix->data, n * p * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                printf("%f ", out_data[i * p + j]);
            }
            printf("\n");
        }
        free(out_data);

        return 0;
    }
}

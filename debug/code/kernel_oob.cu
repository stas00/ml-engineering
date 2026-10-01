#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void write_oob(int *p)
{
    int i = threadIdx.x;
    p[i] = i;   // 8 threads, 4-int buffer -> threads 4-7 run off the end
}

void run()
{
    int *d;
    cudaMalloc(&d, 4 * sizeof(int));   // tight 16-byte allocation
    write_oob<<<1, 8>>>(d);
    cudaDeviceSynchronize();
    cudaFree(d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &run, "launch the buggy kernel");
}

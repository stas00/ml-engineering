#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void race(int *out)
{
    __shared__ int s[64];
    int i = threadIdx.x;
    s[i] = i;                    // written by thread i ...
    out[i] = s[(i + 1) % 64];    // ... read by thread i-1, no barrier between
}

void run()
{
    int *d;
    cudaMalloc(&d, 64 * sizeof(int));
    race<<<1, 64>>>(d);
    cudaDeviceSynchronize();
    cudaFree(d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &run, "launch the racy kernel");
}

# GPU

XXX: to be written

## Currently available GPUs for LLM/VLM workloads

Specs:

- [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/#specifications) - huge availability but already getting outdated
- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100) - 2-3x faster than A100 (half precision), 6x faster for fp8
- [AMD MI250](https://www.amd.com/en/products/server-accelerators/instinct-mi250) ~= A100 - very few clouds have them
- AMD MI300 ~= H100 - don’t expect until late-2024 or even 2025 to be GA
- [Google TPUs](https://cloud.google.com/tpu) - lock-in, can't switch to another vendor like NVIDIA -> AMD
- [Intel Gaudi2](https://habana.ai/products/gaudi2/) ~= H100 - very difficult to find
- [GraphCore]( - very difficult to find

# AMD GPUs Performance

As I haven't had a chance to do any serious work with AMD GPUs, just sharing links for now.

- [AMD Instinct MI300X system optimization](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi300x.html)
- [AMD Instinct MI300X workload optimization](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html)

## Profilers

[omniperf](https://github.com/ROCm/rocprofiler-compute) - Advanced Profiling and Analytics for AMD Hardware - e.g. can plot a roofline performance of your AMD accelerator and many other things.

[rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) - AMD's counterpart of [NVIDIA Nsight Systems](../../../debug/pytorch.md#nvidia-nsight-systems): `rocprofv3 --sys-trace --rccl-trace -- python train.py` traces the HIP and HSA API calls, kernels, memory copies and RCCL (AMD's NCCL) collectives into an SQLite database, which `rocpd2pftrace` converts for viewing in the [Perfetto UI](https://ui.perfetto.dev/).

[ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/) - the successor to Omnitrace, whose `rocprof-sys-*` tools also trace the CPU side, with call-stack sampling and binary instrumentation.

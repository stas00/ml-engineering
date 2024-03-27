#!/usr/bin/env python

"""

This script will help you find the intermediate value of the hidden layer of the MLP when SwiGLU is
used.

It performs a brute force search for the best number closest to 8/3*h that would give the highest
TFLOPS for a matmal of [b*s, h]×[h, 8/3*h]

Despite SwiGLU MLP using 3 matrices, this script searches only one matmul, since the performance is
the same for each matmul.

In the situation where tensor parallelism is used with tp>1 it'd be even faster to search for m1 =
m/tp - so 1/8th with tp=8

To adapt for your situation please modify the search parameters below.

This benchmark was written for the paper The Case for Co-Designing Model Architectures with
Hardware: https://arxiv.org/abs/2401.14489

"""

import torch
from tqdm import trange

### Modify the Search Parameters Begin ###

# this is the hidden_size of the model
d_hidden = 4096

# Now either let the 8/3 ratio give the starting dimension size or choose you own - the 8/3 is
# only a suggestion to compensate for the 3rd additional matrix
d_ff_base = int(8/3*d_hidden)
#d_ff_base = 11008

# batch size - make it larger for small matrices
batch_size = 2**2

# add more profiler iterations for small matrices
num_iterations = 100

# searching range: d_ff_base-distance < d_ff_base < d_ff_base+distance
distance = 100

### Modify the Search Parameters End ###

def benchmark_bmm(b, m, n, k, num_iterations=100, num_matmuls=1):
    A = torch.randn((b, m, n)).half().to("cuda:0")
    B = torch.randn((b, n, k)).half().to("cuda:0")
    C = torch.empty((b, m, k)).half().to("cuda:0")
    num_warmup_iterations = 50

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_event.record()
        with torch.no_grad():
            for i in range(num_matmuls):
                torch.bmm(A, B, out=C)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / (1000 * num_iterations)
    flops_per_sec = (2 * b * m * n * k * num_matmuls) / (elapsed_time * 10**12)
    #print(f"Elapsed time for {num_matmuls} times {b}x{m}x{n}x{k} : {elapsed_time:.3f}")
    #print(f"Throughput (in TFLOP/s) for {b}x{m}x{n}x{k}: {flops_per_sec:.3f}")
    #print("-" * 80)
    return flops_per_sec


print(f"Wanted the closest to {d_ff_base} d_ff value that leads to the highest TFLOPS (d_hidden={d_hidden})\n")
print(f"Searching {int(distance/2)} steps in the range of {d_ff_base-distance} .. {d_ff_base+distance}")
results = {}
for d in trange(-distance, distance, 4):
    d_ff = d_ff_base + d
    # find closest div 4 number, pointless to search odd numbers
    d_ff -= d_ff % 4
    #print(d_ff)
    results[d_ff] = benchmark_bmm(batch_size, m=d_hidden, n=d_ff, k=d_hidden, num_iterations=num_iterations, num_matmuls=1)

starting_tflops_per_sec = benchmark_bmm(batch_size, m=d_hidden, n=d_ff_base, k=d_hidden, num_iterations=num_iterations, num_matmuls=1)
print("Results: baseline, followed by near-by best performing d_ff results:\n")
print(" d_ff  tflops mlp_params")
print("-" * 25)
print(f"{d_ff_base} {starting_tflops_per_sec:7.2f} {3*d_ff_base*d_hidden}")
print("-" * 25)
cut_off = 5  # how many results do you want to see
for d_ff in list(reversed(sorted(results, key=lambda x: results[x])))[:cut_off]:
    print(f"{d_ff} {results[d_ff]:7.2f} {3*d_ff*d_hidden}")

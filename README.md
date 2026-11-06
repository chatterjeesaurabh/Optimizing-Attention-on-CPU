# CS6886 Assignment 5: Optimising Attention on a CPU

**Saurabh Chatterjee (NS26Z334)**

Single-head attention, `O = softmax(Q K^T / sqrt(d)) V`, made fast step by step in C on a dual-socket
Intel Xeon Gold 6338 (Ice Lake-SP, 2 x 32 cores, AVX-512). The answers to Q1-Q5 are in `report.pdf`.

## Contents

| Path | Contents |
|---|---|
| `src/attn_naive.c` | Given naive implementation (baseline) |
| `src/attn_loops.c` | Q3a: loops reordered so the compiler can vectorise them |
| `src/attn_tiled.c` | Q3b: cache and register tiling, with K packed into panels |
| `src/attn_fused.c` | Q3c: row-wise fusion; the N x N matrix is never stored |
| `src/attn_best.c` | Q3d: FlashAttention-style AVX-512 kernel with a vectorised exp |
| `src/attn_omp.c` | Q4: `attn_best` parallelised with OpenMP |
| `input/` | The given inputs |
| `machine_spec.txt` | Machine specification and derived peak, bandwidth and ridge points |
| `results.csv` | Time, GFLOP/s and arithmetic intensity of every variant and thread count |
| `plots/` | Roofline and one time plot per input |
| `report.pdf` | Answers to Q1-Q5 |

## Results (N = 4096, d = 64)

| Variant | Time (ms) | GFLOP/s | Speedup vs naive |
|---|---:|---:|---:|
| naive | 2677 | 1.6 | 1x |
| loops | 381 | 11.3 | 7x |
| tiled | 174 | 24.6 | 15x |
| fused | 173 | 24.8 | 15x |
| best (1 thread) | 44.0 | 97.6 | 61x |
| omp (64 threads) | 1.64 | 2622 | 1634x |

## Build and run

The code was built with clang 22.1.8 using the Makefile's flags (`-O3 -march=native`, plus `-fopenmp` for
`omp`). It does not use BLAS, maths libraries or `-ffast-math`.

```bash
make naive                              # build one variant and time it on every input
make all                                # every variant
OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=close make omp
make check                              # compare every variant's output with naive's (needs make naive first)
make clean
```

`attn_best.c` and `attn_omp.c` need a CPU with AVX-512 for the fast path. Without AVX-512, or when `d` is not a
multiple of 16, they use a portable scalar fallback. `make omp` needs an OpenMP runtime (`libomp`).

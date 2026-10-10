# C Wheel-30 Sieve — msubra_c_wheel30

Contributed by davepl

![Algorithm](https://img.shields.io/badge/Algorithm-wheel-green)
![Faithfulness](https://img.shields.io/badge/Faithful-yes-green)
![Parallelism](https://img.shields.io/badge/Parallel-yes-green)
![Bit count](https://img.shields.io/badge/Bits-1-green)

## Description

Sieve of Eratosthenes in C using a Wheel-30 representation: only numbers
coprime to 30 are stored (residues 1, 7, 11, 13, 17, 19, 23, 29), giving
8 bits per byte and ~33 KB of sieve storage for N=1,000,000.

Key optimizations:
- **Wheel-30 bit packing**: 1 byte per 30 numbers, ~47% smaller than odd-only
- **Pre-sieve**: composites of 7, 11, and 13 are pre-computed once at startup
  (lcm period = 1001 bytes) and tiled into each new sieve instance, skipping
  three marking passes per iteration
- **WIDE_TILE vectorization**: for primes up to 251, multiples are marked by
  building a p-byte tile, expanding it 16× into a wide tile, then ORing with
  Clang auto-vectorized inner loop (NEON on arm64, AVX2 on x86-64)
- **Multi-threaded**: each thread independently allocates and sieves; pass
  counts are summed at the end

Requires Clang (uses `_Pragma("clang loop vectorize(enable)")`). GCC will
compile and produce correct results but without SIMD vectorization of the
marking loop.

## Run instructions

### Docker

```bash
docker build -t primes_c .
docker run --rm primes_c
```

### Native (Linux / macOS)

```bash
clang -march=native -mtune=native -pthread -O3 -ffast-math \
      -fno-math-errno -fno-trapping-math -flto -std=c11 \
      -fvectorize -fslp-vectorize \
      PrimeC_array.c -o primes_c_array -lm -lpthread
./primes_c_array
```

Or via the helper script:

```bash
./run.sh c
```

## Output

```text
Primes Benchmark (c) 2025 Dave's Garage - https://github.com/davepl/primes
--------------------------------------------------------------------------
Computing primes to 1000000 on 1 thread for 5 seconds.
Passes: 58804, Threads: 1, Time: 5.0003, Average: 8.50333e-05, Limit: 1000000, Counts: 78498/78498, Valid: Pass

msubra_c_wheel30;58804;5.0003;1;algorithm=wheel,faithful=yes,bits=1
```

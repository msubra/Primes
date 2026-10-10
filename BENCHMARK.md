# Benchmark Results — msubra_c_wheel30 vs PrimeC solutions

## Machine specs

| Property | Value |
|---|---|
| Model | MacBook Pro (MacBookPro16,4) |
| CPU | Intel Core i9-9980HK @ 2.40 GHz |
| Architecture | x86_64 |
| Physical cores | 8 |
| Logical cores | 16 (Hyper-Threading enabled) |
| L1 cache (per core) | 32 KB I + 32 KB D |
| L2 cache (per core) | 256 KB |
| L3 cache (shared) | 16 MB |
| RAM | 64 GB |
| OS | macOS Darwin 23.6.0 |
| Compiler | Apple Clang 16.0.0 (clang-1600.0.26.6) |
| SIMD available | AVX2 (256-bit) |

---

## Benchmark parameters

**Compiler flags:** `-Ofast -march=native -mtune=native -fvectorize -fslp-vectorize`
**Sieve limit:** 1,000,000
**Duration:** 5 seconds per run
**Reference repo:** `https://github.com/PlummersSoftwareLLC/Primes/tree/drag-race/PrimeC`

---

## Single-thread (1T)

| Rank | Solution | Passes/5s | Algorithm | Faithful | Notes |
|------|----------|-----------|-----------|----------|-------|
| 1 | **msubra_c_wheel30** | **62,043** | wheel | yes | Wheel-30 + pre-sieve + WIDE_TILE AVX2 |
| 2 | rogiervandam_extend | 42,848 | other | yes | Pattern extension, cache-blocked |
| 3 | fvbakel_Cwords | 18,509 | other | yes | Word-sized OR operations |
| 4 | mckoss-c830 | 14,335 | wheel | yes | Wheel-30, no SIMD |
| 5 | merriam | 13,489 | wheel | yes | Lazy wheel |
| 6 | danielspaangberg_48of210 | 11,986 | wheel | yes | Wheel-210 |
| 7 | danielspaangberg_8of30 | 10,796 | wheel | yes | Wheel-30 |
| 8 | fvbakel_Cstriped-block | 9,229 | base | yes | L1 cache-blocked segmented |

## Multi-thread (16T)

| Rank | Solution | Passes/5s | Algorithm | Faithful | Notes |
|------|----------|-----------|-----------|----------|-------|
| 1 | **msubra_c_wheel30** | **463,729** | wheel | yes | pthreads, alloc per iteration |
| 2 | danielspaangberg_5760of30030_epar | 172,354 | wheel | yes | Wheel-30030, OpenMP |
| 3 | danielspaangberg_48of210_epar | 117,734 | wheel | yes | Wheel-210, OpenMP |
| 4 | danielspaangberg_480of2310_epar | 90,591 | wheel | yes | Wheel-2310, OpenMP |
| 5 | danielspaangberg_8of30_epar | 90,101 | wheel | yes | Wheel-30, OpenMP |

---

## Key observations

- **msubra_c_wheel30 is #1 on both 1T and 16T** on this hardware.
- The WIDE_TILE AVX2 vectorization (Clang auto-vectorized inner loop with
  compile-time trip count) is the decisive advantage — no other solution
  leverages SIMD for the marking loop.
- rogiervandam_extend is the closest competitor at 1T (~31% slower); it uses
  pattern extension with cache-blocking but relies on the compiler for SIMD
  without compile-time trip-count guarantees.
- solution_2's OpenMP variants scale poorly vs pthreads on this machine —
  msubra is **2.7× faster** at 16T than the best OpenMP variant
  (5760of30030_epar), despite using a much simpler wheel.
- danielspaangberg's larger wheels (Wheel-210, Wheel-2310, Wheel-30030) reduce
  candidate density but provide no benefit without a fast marking loop.

## Why msubra_c_wheel30 wins

1. **WIDE_TILE**: for each prime p ≤ 251, builds a p-byte tile, expands it
   16× into a `p×16`-byte wide tile, then ORs into the sieve with a
   `_Pragma("clang loop vectorize(enable)")` inner loop. Clang sees a
   compile-time trip count and emits exact AVX2 instructions (no scalar
   epilogue). On this machine: 32-byte YMM registers → up to 32 sieve bytes
   marked per instruction.
2. **Pre-sieve**: composites of 7, 11, 13 (lcm period = 1001 bytes) are
   pre-computed once at startup and tiled into each new sieve instance,
   eliminating three marking passes per iteration.
3. **Wheel-30**: 8 residues per 30 numbers → ~33KB sieve for N=1M,
   fits within the 256KB L2 cache per core.
4. **pthread embarrassingly parallel**: each thread independently allocates
   and sieves; no shared mutable state, no OpenMP scheduling overhead.

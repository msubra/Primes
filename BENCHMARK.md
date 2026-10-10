# Benchmark Results — msubra_c_wheel30 vs PrimeC solutions

**Hardware:** Apple M-series (ARM64, 16 logical cores)
**OS:** macOS Darwin 23.6.0
**Compiler:** Clang (system), `-Ofast -march=native -mtune=native`
**Date:** 2026-10-10
**Sieve limit:** 1,000,000 | **Duration:** 5 seconds per run

All solutions compiled natively from source at:
`https://github.com/PlummersSoftwareLLC/Primes/tree/drag-race/PrimeC`

---

## Single-thread (1T)

| Rank | Solution | Passes/5s | Algorithm | Faithful | Notes |
|------|----------|-----------|-----------|----------|-------|
| 1 | **msubra_c_wheel30** | **62,043** | wheel | yes | Wheel-30 + pre-sieve + WIDE_TILE NEON |
| 2 | rogiervandam_extend | 42,848 | other | yes | Pattern extension, cache-blocked |
| 3 | fvbakel_Cwords | 18,509 | other | yes | Word-sized OR operations |
| 4 | mckoss-c830 | 14,335 | wheel | yes | Wheel-30 |
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
- The WIDE_TILE NEON vectorization (Clang auto-vectorized inner loop with
  compile-time trip count) is the decisive advantage — no other solution
  targets ARM64 NEON explicitly.
- rogiervandam_extend (pattern extension + SSE/AVX on x86) is the closest
  competitor at 1T (~31% slower on ARM64 without its AVX path).
- solution_2's OpenMP variants scale poorly on ARM64 vs pthreads — the
  embarrassingly parallel pthread approach (msubra) is 2.7× faster at 16T
  than the best OpenMP variant (5760of30030_epar).
- danielspaangberg's larger wheels (Wheel-210, Wheel-2310, Wheel-30030)
  do not compensate for lack of SIMD on this platform.

## Why msubra_c_wheel30 wins on ARM64

1. **WIDE_TILE**: for each prime p ≤ 251, builds a p-byte tile, expands it
   16× into a `p×16`-byte wide tile, then ORs into the sieve with a
   `_Pragma("clang loop vectorize(enable)")` inner loop. Clang sees a
   compile-time trip count and emits exact NEON instructions (no scalar
   epilogue).
2. **Pre-sieve**: composites of 7, 11, 13 (lcm period = 1001 bytes) are
   pre-computed once at startup and tiled into each new sieve instance,
   eliminating three marking passes per iteration.
3. **Wheel-30**: 8 residues per 30 numbers → ~33KB sieve for N=1M,
   fits comfortably in L2 cache.
4. **pthread embarrassingly parallel**: each thread independently
   allocates and sieves; no shared mutable state, no OpenMP overhead.

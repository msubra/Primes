#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Usage: ./run.sh [cpp|c|both]   (default: both)
# ---------------------------------------------------------------------------
MODE="${1:-both}"

IS_DARWIN=false
[[ "$(uname -s)" == "Darwin" ]] && IS_DARWIN=true

# Common vectorization flags for Darwin
DARWIN_EXTRA=()
$IS_DARWIN && DARWIN_EXTRA=(-fvectorize -fslp-vectorize)

build_and_run_cpp() {
    local src="PrimeCPP_array.cpp"
    local bin="primes_array"
    local cxx="${CXX:-clang++}"
    local flags=(-march=native -mtune=native -pthread -O3 -ffast-math
                 -fno-math-errno -fno-trapping-math -flto -std=c++17
                 "${DARWIN_EXTRA[@]}")

    echo "=== C++ ==="
    echo "Compiling $src -> $bin"
    "$cxx" "${flags[@]}" "$src" -o "$bin"

    echo "Running $bin"
    set +e; "./$bin"; local s=$?; set -e
    [[ $s -ne 0 ]] && echo "Note: exited with status $s (prime count mod 256)."
    echo
}

build_and_run_c() {
    local src="PrimeC_array.c"
    local bin="primes_c_array"
    local cc="${CC:-clang}"
    local flags=(-march=native -mtune=native -pthread -O3 -ffast-math
                 -fno-math-errno -fno-trapping-math -flto -std=c11
                 "${DARWIN_EXTRA[@]}")

    echo "=== C ==="
    echo "Compiling $src -> $bin"
    "$cc" "${flags[@]}" "$src" -o "$bin" -lm -lpthread

    echo "Running $bin"
    set +e; "./$bin"; local s=$?; set -e
    [[ $s -ne 0 ]] && echo "Note: exited with status $s (prime count mod 256)."
    echo
}

case "$MODE" in
    cpp)  build_and_run_cpp ;;
    c)    build_and_run_c   ;;
    both) build_and_run_cpp; build_and_run_c ;;
    *)
        echo "Usage: $0 [cpp|c|both]"
        exit 1
        ;;
esac

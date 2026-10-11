IMAGE       := primes_c
BENCH_IMAGE := primes_bench
CC       := clang
CXX      := clang++
CFLAGS   := -march=native -mtune=native -pthread -O3 -ffast-math \
             -fno-math-errno -fno-trapping-math -flto -std=c11 \
             -fvectorize -fslp-vectorize
CXXFLAGS := -march=native -mtune=native -pthread -O3 -ffast-math \
             -fno-math-errno -fno-trapping-math -flto -std=c++17 \
             -fvectorize -fslp-vectorize
LLVM_PROFDATA := $(shell xcrun -f llvm-profdata 2>/dev/null || echo llvm-profdata)
PGO_DIR  := pgo_data

.PHONY: build run clean build-native verify bench bench-build pgo-extend

build:
	docker build -t $(IMAGE) .

run: build
	-docker run --rm $(IMAGE)

clean:
	docker rmi -f $(IMAGE)
	rm -rf bin

bin:
	mkdir -p bin

build-native: bin/primes_c_array bin/primes_array bin/primes_c_extend bin/primes_c_w210

bin/primes_c_array: PrimeC_array.c | bin
	$(CC) $(CFLAGS) $< -o $@ -lm -lpthread

bin/primes_array: PrimeCPP_array.cpp | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/primes_c_extend: PrimeC_extend.c | bin
	$(CC) $(CFLAGS) $< -o $@ -lm -lpthread

bin/primes_c_w210: PrimeC_w210.c | bin
	$(CC) $(CFLAGS) $< -o $@ -lm -lpthread

verify: build-native
	./bin/primes_c_array -1 -p 2>/tmp/c_raw.txt;   true
	./bin/primes_array   -1 -p 2>/tmp/cpp_raw.txt; true
	grep '^2, ' /tmp/c_raw.txt   | tr ', ' '\n' | grep -E '^[0-9]+$$' > /tmp/c_primes.txt
	grep '^2, ' /tmp/cpp_raw.txt | tr ', ' '\n' | grep -E '^[0-9]+$$' > /tmp/cpp_primes.txt
	@echo "C   primes: $$(wc -l < /tmp/c_primes.txt)"
	@echo "C++ primes: $$(wc -l < /tmp/cpp_primes.txt)"
	@diff /tmp/c_primes.txt /tmp/cpp_primes.txt && echo "IDENTICAL" || echo "DIFFER"

bench-build:
	docker build -f Dockerfile.bench -t $(BENCH_IMAGE) .

bench: bench-build
	docker run --rm $(BENCH_IMAGE)

# PGO build for primes_c_extend (+2.5% on this sieve; others show no benefit).
# Instruments, profiles for 10s, merges, then rebuilds with profile data.
pgo-extend: PrimeC_extend.c | bin $(PGO_DIR)
	$(CC) $(CFLAGS) -fprofile-instr-generate=$(PGO_DIR)/extend.profraw \
	    $< -o bin/primes_c_extend_pgo_instr -lm -lpthread
	LLVM_PROFILE_FILE=$(PGO_DIR)/extend.profraw \
	    ./bin/primes_c_extend_pgo_instr -t 1 -s 10 -q 2>/dev/null || true
	$(LLVM_PROFDATA) merge $(PGO_DIR)/extend.profraw \
	    -o $(PGO_DIR)/extend.profdata
	$(CC) $(CFLAGS) -fprofile-instr-use=$(PGO_DIR)/extend.profdata \
	    $< -o bin/primes_c_extend -lm -lpthread
	rm -f bin/primes_c_extend_pgo_instr

$(PGO_DIR):
	mkdir -p $(PGO_DIR)

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

.PHONY: build run clean build-native verify bench bench-build

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

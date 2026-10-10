IMAGE    := primes_c
CC       := clang
CXX      := clang++
CFLAGS   := -march=native -mtune=native -pthread -O3 -ffast-math \
             -fno-math-errno -fno-trapping-math -flto -std=c11 \
             -fvectorize -fslp-vectorize
CXXFLAGS := -march=native -mtune=native -pthread -O3 -ffast-math \
             -fno-math-errno -fno-trapping-math -flto -std=c++17 \
             -fvectorize -fslp-vectorize

.PHONY: build run clean build-native verify

build:
	docker build -t $(IMAGE) .

run: build
	-docker run --rm $(IMAGE)

clean:
	docker rmi -f $(IMAGE)
	rm -f primes_c_array primes_array

build-native: primes_c_array primes_array

primes_c_array: PrimeC_array.c
	$(CC) $(CFLAGS) $< -o $@ -lm -lpthread

primes_array: PrimeCPP_array.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

verify: build-native
	./primes_c_array -1 -p 2>/tmp/c_raw.txt;   true
	./primes_array   -1 -p 2>/tmp/cpp_raw.txt; true
	grep '^2, ' /tmp/c_raw.txt   | tr ', ' '\n' | grep -E '^[0-9]+$$' > /tmp/c_primes.txt
	grep '^2, ' /tmp/cpp_raw.txt | tr ', ' '\n' | grep -E '^[0-9]+$$' > /tmp/cpp_primes.txt
	@echo "C   primes: $$(wc -l < /tmp/c_primes.txt)"
	@echo "C++ primes: $$(wc -l < /tmp/cpp_primes.txt)"
	@diff /tmp/c_primes.txt /tmp/cpp_primes.txt && echo "IDENTICAL" || echo "DIFFER"

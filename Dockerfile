# ---- build stage ----
FROM ubuntu:22.04 AS build

RUN apt-get update -qq \
 && apt-get install -y bash clang lld

WORKDIR /opt/app
COPY *.c .

ARG TARGETARCH
RUN set -eux; \
    CFLAGS="-march=native -mtune=native -pthread -O3 -ffast-math -fno-math-errno -fno-trapping-math -flto -std=c11"; \
    if [ "${TARGETARCH:-amd64}" = "arm64" ]; then \
      CFLAGS="-mcpu=native -pthread -O3 -ffast-math -fno-math-errno -fno-trapping-math -flto -std=c11"; \
    fi; \
    clang ${CFLAGS} -fuse-ld=lld PrimeC_array.c -o primes_c_array.exe -lm -lpthread

# ---- runtime stage ----
FROM ubuntu:22.04

WORKDIR /opt/app
COPY --from=build /opt/app/primes_c_array.exe /opt/app/
COPY benchmark.sh .
RUN chmod +x benchmark.sh

ENTRYPOINT ["./benchmark.sh"]
CMD ["dummy", "-l", "1000000"]

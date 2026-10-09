@ECHO OFF

ECHO Building and running the C array approach...
ECHO:
gcc -Ofast PrimeC_array.c -std=c11 -o Primes_c_array.exe -lm -lpthread
.\Primes_c_array.exe
ECHO:

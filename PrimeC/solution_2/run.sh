#!/bin/sh
# Parse flags
RUN_ALL=0
for arg in "$@"; do
    if [ "$arg" = "--all" ]; then
        RUN_ALL=1
    fi
done

# Serial versions
for x in sieve_1of2 sieve_8of30 sieve_8of30_only_write_read_bits sieve_48of210 sieve_48of210_only_write_read_bits sieve_480of2310_only_write_read_bits sieve_5760of30030_only_write_read_bits; do
    ./$x
done
# Thread parallel versions, computing each sieve quicker.
#
# Chooses a number corresponding to all processing units in your system,
# but max 4 since it does not scale extremely well
NUM_PROCS=`nproc`
if [ "$NUM_PROCS" -gt "4" ]; then
    NUM_PROCS=4
fi
for x in sieve_1of2_par sieve_8of30_par sieve_48of210_par sieve_480of2310_par sieve_5760of30030_par; do
    OMP_NUM_THREADS=$NUM_PROCS ./$x
done
# Embarrassingly parallel versions, computing multiple sieves at the same time.
# Chooses a number corresponding to all processing units in your system
NUM_PROCS=`nproc`
for x in sieve_1of2_epar sieve_8of30_epar sieve_48of210_epar sieve_480of2310_epar sieve_5760of30030_epar; do
    OMP_NUM_THREADS=$NUM_PROCS ./$x
done
# msubra's 8-of-30 with presieve and wide-tile marking
./sieve_8of30_msubra_extend -t 1
./sieve_8of30_msubra_extend -t $NUM_PROCS

# --all: also run solutions 1, 3, 4, and 5
if [ "$RUN_ALL" = "1" ]; then
    SOLUTION_ROOT="$(dirname "$0")/.."

    # solution_1 — mckoss
    "$SOLUTION_ROOT/solution_1/sieve"

    # solution_3 — fvbakel (segmented words + striped-block)
    "$SOLUTION_ROOT/solution_3/primes_words"
    "$SOLUTION_ROOT/solution_3/primes_striped-block"

    # solution_4 — merriam
    "$SOLUTION_ROOT/solution_4/sieve"

    # solution_5 — rogiervandam extend (serial + embarrassingly parallel)
    S5="$SOLUTION_ROOT/solution_5"
    if [ ! -f "$S5/bin/sieve_extend" ]; then
        echo "solution_5 not compiled — run: cd $S5 && ./sieve clang compile"
    else
        "$S5/bin/sieve_extend"
        "$S5/bin/sieve_extend_epar" --time 5 --threads 4
        "$S5/bin/sieve_extend_epar" --time 5 --threads "$NUM_PROCS"
    fi
fi

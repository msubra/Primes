#!/usr/bin/env python3
"""
bench.py — run all prime sieve C/C++ solutions and display a ranked table.

Output lines from each binary are expected in the standard drag-race format:
    label;passes;time_seconds;threads;algorithm=X,faithful=Y,bits=Z
"""
import subprocess
import re
import os
import sys

NPROC = os.cpu_count() or 1

# Each tuple: (section, binary, args_list, env_override)
# section  — "1T" or f"{NPROC}T" — controls which table the result lands in
SOLUTIONS = [
    # ── Single-thread ──────────────────────────────────────────────────
    ("1T", "./bin/mckoss",               ["--size", "1000000", "--secs", "5"], {}),
    ("1T", "./bin/daniel_8of30",         [],                                   {}),
    ("1T", "./bin/daniel_48of210",       [],                                   {}),
    ("1T", "./bin/fvbakel_words",        [],                                   {}),
    ("1T", "./bin/fvbakel_striped",      [],                                   {}),
    ("1T", "./bin/merriam",              [],                                   {}),
    ("1T", "./bin/rogiervandam",         [],                                   {}),
    ("1T", "./bin/primes_cpp",           ["-t", "1", "-s", "5"],               {}),
    ("1T", "./bin/primes_c_array",       ["-t", "1", "-s", "5"],               {}),
    ("1T", "./bin/primes_c_extend",      ["-t", "1", "-s", "5"],               {}),
    ("1T", "./bin/primes_c_w210",        ["-t", "1", "-s", "5"],               {}),
    # ── Multi-thread ───────────────────────────────────────────────────
    ("MT", "./bin/daniel_8of30_epar",       [], {"OMP_NUM_THREADS": str(NPROC)}),
    ("MT", "./bin/daniel_48of210_epar",     [], {"OMP_NUM_THREADS": str(NPROC)}),
    ("MT", "./bin/daniel_480of2310_epar",   [], {"OMP_NUM_THREADS": str(NPROC)}),
    ("MT", "./bin/daniel_5760of30030_epar", [], {"OMP_NUM_THREADS": str(NPROC)}),
    ("MT", "./bin/primes_cpp",              ["-t", str(NPROC), "-s", "5"],     {}),
    ("MT", "./bin/primes_c_array",          ["-t", str(NPROC), "-s", "5"],     {}),
    ("MT", "./bin/primes_c_extend",         ["-t", str(NPROC), "-s", "5"],     {}),
    ("MT", "./bin/primes_c_w210",           ["-t", str(NPROC), "-s", "5"],     {}),
]

# Regex matching the standard drag-race output line
RESULT_RE = re.compile(r'^([^\s;]+);(\d+);([\d.]+);(\d+);(.*)$')


def run_solution(section, binary, args, env_override):
    env = {**os.environ, **env_override}
    cmd = [binary] + args
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=35,
        )
        output = proc.stdout.decode(errors='replace')
        for line in reversed(output.splitlines()):   # last matching line wins
            m = RESULT_RE.match(line.strip())
            if m:
                label, passes, time_s, threads, tags = m.groups()
                passes  = int(passes)
                time_s  = float(time_s)
                threads = int(threads)
                pps     = passes / time_s if time_s > 0 else 0
                algo    = "?"
                faithful = "?"
                for tag in tags.split(','):
                    if tag.startswith('algorithm='):
                        algo = tag.split('=', 1)[1]
                    elif tag.startswith('faithful='):
                        faithful = tag.split('=', 1)[1]
                return {
                    'section':  section,
                    'label':    label,
                    'passes':   passes,
                    'time':     time_s,
                    'threads':  threads,
                    'pps':      pps,
                    'algorithm': algo,
                    'faithful':  faithful,
                }
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {binary}", file=sys.stderr)
    except FileNotFoundError:
        print(f"  MISSING: {binary}", file=sys.stderr)
    return None


def print_table(title, results, show_threads=False):
    if not results:
        print(f"\n{title}\n  (no results)", flush=True)
        return

    sol_w  = max(max(len(r['label']) for r in results), 30)
    col_defs = [
        ("Rank",       4,  'r'),
        ("Solution",   sol_w, 'l'),
        ("Passes/5s",  10, 'r'),
        ("Passes/sec", 12, 'r'),
    ]
    if show_threads:
        col_defs.append(("Threads", 7, 'r'))
    col_defs += [
        ("Algorithm",  9, 'l'),
        ("Faithful",   8, 'l'),
    ]

    def sep_line(left, mid, right):
        return left + mid.join('─' * (w + 2) for _, w, _ in col_defs) + right

    def row(values):
        cells = []
        for (_, w, align), v in zip(col_defs, values):
            s = str(v)
            if align == 'r':
                cells.append(f" {s.rjust(w)} ")
            else:
                cells.append(f" {s.ljust(w)} ")
        return "│" + "│".join(cells) + "│"

    print(f"\n{title}", flush=True)
    print(sep_line('┌', '┬', '┐'))
    print(row([c[0] for c in col_defs]))
    print(sep_line('├', '┼', '┤'))

    for rank, r in enumerate(results, 1):
        vals = [rank, r['label'], f"{r['passes']:,}", f"{r['pps']:,.1f}"]
        if show_threads:
            vals.append(r['threads'])
        vals += [r['algorithm'], r['faithful']]
        print(row(vals))

    print(sep_line('└', '┴', '┘'))


def main():
    total = len(SOLUTIONS)
    print(f"Benchmark — {NPROC} logical CPUs detected")
    print(f"Running {total} configurations (~5s each, ~{total * 5 // 60}m{total * 5 % 60}s total)...\n")

    results_1t = []
    results_mt = []

    for i, (section, binary, args, env_override) in enumerate(SOLUTIONS, 1):
        desc = os.path.basename(binary)
        if args:
            desc += ' ' + ' '.join(args)
        if env_override:
            desc += '  [' + ' '.join(f'{k}={v}' for k, v in env_override.items()) + ']'
        print(f"  [{i:2d}/{total}] {desc}", end=' ', flush=True)
        r = run_solution(section, binary, args, env_override)
        if r:
            print(f"→ {r['passes']:,} passes  ({r['pps']:,.0f}/s)", flush=True)
            if section == '1T':
                results_1t.append(r)
            else:
                results_mt.append(r)
        else:
            print("→ skipped", flush=True)

    results_1t.sort(key=lambda r: r['pps'], reverse=True)
    results_mt.sort(key=lambda r: r['pps'], reverse=True)

    print_table("── Single-Thread (1T) " + "─" * 50, results_1t, show_threads=False)
    print_table(f"── Multi-Thread ({NPROC}T) " + "─" * 50, results_mt, show_threads=True)
    print()


if __name__ == '__main__':
    main()

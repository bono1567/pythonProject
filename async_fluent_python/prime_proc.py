#!/usr/bin/env python3

"""
procs.py: shows that multiprocessing on a multicore machine
can be faster than sequential code for CPU-intensive work.
"""

# tag::PRIMES_PROC_TOP[]
import sys
import timeit
from time import perf_counter
from typing import NamedTuple
from multiprocessing import Process, SimpleQueue, cpu_count
from multiprocessing import queues

from primes import is_prime, NUMBERS

class PrimeResult(NamedTuple):
    n: int
    prime: bool
    elapsed: float

JobQueue = queues.SimpleQueue[int]
ResultQueue = queues.SimpleQueue[PrimeResult]

def check(n: int) -> PrimeResult:
    t0 = perf_counter()
    res = is_prime(n)
    return PrimeResult(n, res, perf_counter() - t0)

def worker(jobs: JobQueue, results: ResultQueue) -> None:
    while n := jobs.get():
        results.put(check(n))
    results.put(PrimeResult(0, False, 0.0))

def start_jobs(
    procs: int, jobs: JobQueue, results: ResultQueue
) -> None:
    for n in NUMBERS:
        jobs.put(n)
    for _ in range(procs):
        proc = Process(target=worker, args=(jobs, results))
        proc.start()
        jobs.put(0)

def report(procs: int, results: ResultQueue) -> int:
    checked = 0
    procs_done = 0
    while procs_done < procs:
        n, prime, elapsed = results.get()
        if n == 0:
            procs_done += 1
        else:
            checked += 1
            label = 'P' if prime else ' '
            print(f'{n:16}  {label} {elapsed:9.6f}s')
    return checked

def main() -> None:
    if len(sys.argv) < 2:
        procs = cpu_count()
    else:
        procs = int(sys.argv[1])

    print(f'Checking {len(NUMBERS)} numbers with {procs} processes:')
    t0 = perf_counter()
    jobs: JobQueue = SimpleQueue()
    results: ResultQueue = SimpleQueue()
    start_jobs(procs, jobs, results)
    checked = report(procs, results)
    elapsed = perf_counter() - t0
    print(f'{checked} checks in {elapsed:.2f}s') 

if __name__ == '__main__':
    print(cpu_count())
    execution_time = timeit.timeit(
    'main()', 
    setup='from __main__ import main',
    number=1
    )
    print(f"Average execution time: {execution_time:.6f} seconds")
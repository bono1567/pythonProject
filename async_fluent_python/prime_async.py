import os
import sys
import asyncio
from time import perf_counter
from typing import NamedTuple

from primes import is_prime, NUMBERS

class PrimeResult(NamedTuple):
    n: int
    prime: bool
    elapsed: float

async def check(n: int) -> PrimeResult:
    t0 = perf_counter()
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, is_prime, n) # Use this for executing normal functions in an event loop.
    return PrimeResult(n, res, perf_counter() - t0)

async def worker(jobs: asyncio.Queue[int], results: asyncio.Queue[PrimeResult]) -> None:
    while True:
        n = await jobs.get()
        if n == 0:
            break
        result = await check(n)
        await results.put(result)
    await results.put(PrimeResult(0, False, 0.0))

async def start_jobs(workers: int, jobs: asyncio.Queue[int], results: asyncio.Queue[PrimeResult]) -> None:
    for n in NUMBERS:
        await jobs.put(n)
    
    worker_tasks = []
    for _ in range(workers):
        task = asyncio.create_task(worker(jobs, results))
        worker_tasks.append(task)
        await jobs.put(0)
    await asyncio.gather(*worker_tasks)

async def report(workers: int, results: asyncio.Queue[PrimeResult]) -> int:
    checked = 0
    workers_done = 0
    while workers_done < workers:
        n, prime, elapsed = await results.get()
        if n == 0:
            workers_done += 1
        else:
            checked += 1
            label = 'P' if prime else ' '
            print(f'{n:16}  {label} {elapsed:9.6f}s')
    return checked

async def main_async() -> None:
    if len(sys.argv) < 2:
        workers = os.cpu_count() or 1
    else:
        workers = int(sys.argv[1])

    print(f'Checking {len(NUMBERS)} numbers with {workers} async workers:')
    t0 = perf_counter()
    
    # Create asyncio queues
    jobs: asyncio.Queue[int] = asyncio.Queue()
    results: asyncio.Queue[PrimeResult] = asyncio.Queue()
    
    # Start the reporter FIRST so it's ready to consume results
    report_task = asyncio.create_task(report(workers, results))
    
    # Give the reporter a moment to start up
    # await asyncio.sleep(0.001)
    
    # Then start the jobs
    jobs_task = asyncio.create_task(start_jobs(workers, jobs, results))
    
    # Wait for both to complete
    await jobs_task
    checked = await report_task
    
    elapsed = perf_counter() - t0
    print(f'{checked} checks in {elapsed:.2f}s')

def main() -> None:
    asyncio.run(main_async())

if __name__ == '__main__':
    main()
import asyncio
import itertools
import math


async def spin(msg: str) -> None:
    for char in itertools.cycle(r'\|/-'):
        status = f'\r{char} {msg}'
        print(status, end='', flush=True)
        try:
            await asyncio.sleep(.1)
        except asyncio.CancelledError:
            break
    blanks = ' ' * len(status)
    print(f'\r{blanks}\r', end='')


async def is_prime(n: int) -> bool:
    try:
        if n < 2:
            return False
        if n == 2:
            return True
        
        root = math.isqrt(n)
        for i in range(3, root + 1, 2):
            if n % i == 0:
                return False
        return True
    except:
        print("error in prime check")
        return False


async def supervisor(n: int) -> bool:
    spinner = asyncio.create_task(spin("Async Thinking"))
    print(f'Spinner Object: {spinner}')
    result = await is_prime(n)
    spinner.cancel()
    return result


if __name__ == "__main__":
    n = int(input("Check prime for: "))
    result = asyncio.run(supervisor(10000))
    print(f"\nResult: {n} is {'prime' if result else 'not prime'}")
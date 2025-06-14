import math
import itertools
from threading import Thread, Event

def is_prime(n: int) -> bool:
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

def spin(msg: str, done: Event) -> None:
    for char in itertools.cycle(r'\|/-'):
        status = f'\r{char} {msg}'
        print(status, end='', flush=True)
        if done.wait(.1):
            break
    blanks = ' ' * len(status)
    print(f'\r{blanks}\r', end='')


def supervisor(n: int) -> int:
    done = Event()
    spinner = Thread(target=spin, args=('Thinking', done))
    print(f'Spnner Object: {spinner}')
    spinner.start()
    result = is_prime(n)
    done.set()
    spinner.join()
    return result


def main() -> None:
    result = supervisor(int(input("Check prime for: ")))
    print(f'Answer: {result}')

if __name__ == '__main__':
    main()
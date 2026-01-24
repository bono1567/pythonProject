from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

condemed = 0

def disciple():
    global condemed
    condemed += 1

disciples = 12
with ProcessPoolExecutor() as counsel:
    for _ in repeat(None, disciples):
        counsel.submit(disciple)

print(condemed)
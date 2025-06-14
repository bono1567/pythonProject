from concurrent.futures import ProcessPoolExecutor, Future, as_completed
import random
import os
import timeit
from datetime import datetime

def caculate_pv_of_bond(face_value: float, ytm: float, coupon_rate: float, time_to_maturity: float,
                        bi_yearly: bool = False, last_coupon_date: datetime = None, clean: bool = True):
    """
    Calculate the PV of a given bond given its details
    
    """

    if not clean and last_coupon_date:
        raise NotImplementedError("For the dirty price please provide the 'last_coupon_date' value.")

    if bi_yearly:
        return caculate_pv_of_bond(face_value, ytm/2, coupon_rate/2, time_to_maturity*2, last_coupon_date, clean, bi_yearly=False)
    
    coupon_payment = face_value * coupon_rate
    present_value = coupon_payment * (1 - (1 + ytm)**(-time_to_maturity)) / ytm + \
                   face_value * (1 + ytm)**(-time_to_maturity)
    if not clean:
        interest_accrued = coupon_payment * (datetime.now() - last_coupon_date).days / 365
        present_value += interest_accrued
        return present_value, interest_accrued
    return present_value, -1


def get_collection_of_bond_pv_threaded_pool(n=1000, no_thread=True):
    
    list_of_bonds = [
        {
            "face_value": random.choice([100, 1000, 1_000_000]),
            "ytm": random.uniform(0.06, 0.2),
            "coupon_rate": random.uniform(0.01, 0.05),
            "time_to_maturity": 10,
            "clean": True
        } for _ in range(n)
    ]

    if no_thread:
        [caculate_pv_of_bond(**bond) for bond in list_of_bonds]
        return
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        to_do: list[Future] = []
        for bond in list_of_bonds:
            future = executor.submit(caculate_pv_of_bond, **bond)
            to_do.append(future)
        
        for count, future in enumerate(as_completed(to_do), 1):
            future.result()


def benchmark_bonds(n: int = 1000, repeat: int = 1) -> None:
    """Benchmark bond calculation with and without threading"""
    print(f"Benchmarking with {n} bonds...")
    
    # Time synchronous version
    sync_time = timeit.repeat(
        lambda: get_collection_of_bond_pv_threaded_pool(n, no_thread=True),
        number=1,
        repeat=repeat
    )
    
    # Time threaded version
    threaded_time = timeit.repeat(
        lambda: get_collection_of_bond_pv_threaded_pool(n, no_thread=False),
        number=1,
        repeat=repeat
    )
    
    print(f"\nResults for {n} bonds (best of {repeat} runs):")
    print(f"Synchronous: {min(sync_time):.4f} seconds")
    print(f"Threaded:    {min(threaded_time):.4f} seconds")
    print(f"Speedup:     {min(sync_time)/min(threaded_time):.2f}x")

if __name__ == "__main__":
    benchmark_bonds(n=1, repeat=1)a
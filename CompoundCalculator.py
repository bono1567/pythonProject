import random

INITIAL_AMOUNT = 100000
PER_STOCK_INTEREST = 0.05
PER_MONTH_SALARY = 15000
N_OF_YEARS = 1
PER_MONTH_STOCK_SOLD = 4
TOTAL_STOCKS_HELD = 5
LOSS_CHANCE = 1/10


def amount_per_month(current_amount, time_elapsed, loss_accumulated):
    current_amount += PER_MONTH_SALARY
    for _ in range(PER_MONTH_STOCK_SOLD):
        if random.choices([1, 2], [LOSS_CHANCE, 1 - LOSS_CHANCE])[0] == 2:
            current_amount += (current_amount / TOTAL_STOCKS_HELD) * PER_STOCK_INTEREST
            print("GAIN. Current amount: {}. Time: {} years {} months.".format(current_amount, int(time_elapsed / 12),
                                                                               time_elapsed % 12))
        else:
            current_amount -= (current_amount / TOTAL_STOCKS_HELD) * PER_STOCK_INTEREST
            loss_accumulated += (current_amount / TOTAL_STOCKS_HELD) * PER_STOCK_INTEREST
            print("LOSS. Current amount: {}. Time: {} years {} months.".format(current_amount, int(time_elapsed / 12),
                                                                               time_elapsed % 12))
    return current_amount, loss_accumulated


def amount_per_year(current_amount, months_passed, loss_accumulated):
    if months_passed % 12 == 0:
        current_amount, loss_accumulated = amount_per_month(current_amount, months_passed, loss_accumulated)
        print("YEARLY EARNING FOR YEAR {}: {}".format(int(months_passed / 12), current_amount))
        return current_amount, loss_accumulated
    else:
        result = amount_per_month(current_amount, months_passed, loss_accumulated)
        return amount_per_year(result[0], months_passed + 1, result[1])


def amount_per_session(current_amount, loss_accumulated=0):
    for i in range(N_OF_YEARS):
        current_amount, loss_accumulated = amount_per_year(current_amount, i * 12 + 1, loss_accumulated)
    print("TOTAL AMOUNT: {}".format(current_amount))
    print("TOTAL AMOUNT INVESTED: {}".format(INITIAL_AMOUNT + PER_MONTH_SALARY * N_OF_YEARS * 12))
    print("TOTAL LOSS INCURRED IN ALL TRADES: {}".format(loss_accumulated))
    profit = (current_amount - (INITIAL_AMOUNT + PER_MONTH_SALARY * N_OF_YEARS * 12))
    print("NET PROFIT: {}".format(profit))
    print("NET PROFIT PER CENT: {}%".format((profit/(INITIAL_AMOUNT + PER_MONTH_SALARY * N_OF_YEARS * 12))*100))
    return current_amount


if __name__ == '__main__':
    amount_per_session(INITIAL_AMOUNT)

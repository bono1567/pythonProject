def parse(A, i, j, counter):
    if i == 6 and j == 6:
        counter += 1
        return counter
    if i + 1 < 7:
        counter = parse(A, i + 1, j, counter)
    if j + 1 < 7:
        counter = parse(A, i, j + 1, counter)
    return counter


A = [[0 for _ in range(6)] for _ in range(7)]
c = parse(A, 0, 0, 0)
print(c)

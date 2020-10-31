import numpy as np

# xtol=exit condition, xnew-xold is less than xtol
maxiter = 1000

n = 20
lambd = 3
h = 10
g = 0.1


def raw(left, right, up, down):
    t = ((h ** 2 * g) / lambd + left + right + up + down) / 4

    return t


def seq(t0):
    for _ in range(maxiter):
        t1 = np.zeros((n, n))
        for row in range(1, n - 1):
            for col in range(1, n - 1):
                print('Row: {row}, Column: {col}'.format(col=col, row=row))
                t1[row, col] = raw(t0[row, col - 1], t0[row, col + 1], t0[row + 1, col], t0[row - 1, col])
                print('{} -> {}'.format(t0[row, col], t1[row, col]))
        t0 = t1
    return t0


t_0 = np.zeros((n, n))

print(t_0)

t = seq(t_0)

print(t)

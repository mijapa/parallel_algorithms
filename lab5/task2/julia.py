# mpiexec -n 1 -usize 17 python julia.py

import time
import os

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

x0, x1, w = -2.0, +2.0, 640 * 2
y0, y1, h = -1.5, +1.5, 480 * 2
dx = (x1 - x0) / w
dy = (y1 - y0) / h

c = complex(0, 0.65)

# filename = "times.csv"
filename = "times_fft.csv"


def fft():
    from pandas import np
    start = time.time()
    np.fft.fft(np.exp(2j * np.pi * np.arange(10000000) / 8))
    end = time.time()
    t = end - start
    print(f'fft time: {t} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


def julia(x, y):
    z = complex(x, y)
    n = 255
    while abs(z) < 3 and n > 1:
        z = z ** 2 + c
        n -= 1
    return n


def julia_line(k):
    """
    1. rozszerz funkcję julia_line(k), tak aby wypisywała ona czas obliczeń dla linii, rank procesu oraz numer linii.
    """
    # start = time.time()
    if k == 0:
        fft()
    line = bytearray(w)
    y = y1 - k * dy
    for j in range(w):
        x = x0 + j * dx
        line[j] = julia(x, y)
    # end = time.time()
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # t = end - start
    # print(f'czas: {t}, rank: {rank}, numer linii: {k}')

    # with open(filename, "a") as myfile:
    #     myfile.write(f'{t},{rank},{k}\n')
    return line


if __name__ == '__main__':
    # if not os.path.exists(filename):
    #     with open(filename, "a") as myfile:
    #         myfile.write('czas,rank,linia\n')

    start = time.time()
    with MPIPoolExecutor() as executor:
        image = executor.map(julia_line, range(h))
        with open('julia.pgm', 'wb') as f:
            f.write(b'P5 %d %d %d\n' % (w, h, 255))
            for line in image:
                f.write(line)
    end = time.time()
    time = end - start
    print(f'Czas wykonania całego programu (bez pomiarów wewnętrznych): {time}')

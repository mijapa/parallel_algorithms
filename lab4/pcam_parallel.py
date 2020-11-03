#!/usr/bin/env python
import math
import time

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = 0

if rank == 0:
    start = time.time()

if math.sqrt(size) % 1 != 0:
    print('World size should be power of integer')

maxiter = 1000

size_sqrt = int(math.sqrt(size))
m = size_sqrt
i = 1000
n = i**2

sqrt_pts_per_task = int(i / m)

if n / m % 1 != 0:
    print('World size should be correlated with task size')

lambd = 3
h = 10
g = 0.1
border_value = 0
start_value = 0


# left, right, up, down
def raw(left, right, up, down):
    t_ = ((h ** 2 * g) / lambd + left + right + up + down) / 4

    return t_


def initialize_t(value):
    t_ = []
    for _ in range(sqrt_pts_per_task):
        t_.append(value)
    return t_


def isent_recv(deso):
    t_ = initialize_t(start_value)
    comm.isend(t_, dest=deso)
    comm.recv(source=deso)


dim = sqrt_pts_per_task + 2
t0 = np.zeros((dim, dim))

for _ in range(maxiter):

    # upper border
    if 0 < rank < m - 1:
        t = np.zeros((sqrt_pts_per_task, sqrt_pts_per_task))
        isent_recv(rank + m)

    # lower border
    elif m * (m - 1) < rank < m * m - 1:
        t = np.zeros((sqrt_pts_per_task, sqrt_pts_per_task))
        isent_recv(rank - m)

    # left border
    elif rank % m == 0 and rank != 0 and rank != m * (m - 1):
        t = np.zeros((sqrt_pts_per_task, sqrt_pts_per_task))
        isent_recv(rank + 1)

    # right border
    elif rank % m == m - 1 and rank != m - 1 and rank != m * m - 1:
        t = np.zeros((sqrt_pts_per_task, sqrt_pts_per_task))
        isent_recv(rank - 1)

    # corners
    elif rank == 0 or rank == m - 1 or rank == m * (m - 1) or rank == m * m - 1:
        t = np.ones((sqrt_pts_per_task, sqrt_pts_per_task))

    # middle
    else:
        tl = []
        tr = []
        tu = []
        td = []
        for i in range(1, dim - 1):
            tl.append(t0[i][1])
            tr.append(t0[i][dim - 2])
            tu.append(t0[1][i])
            td.append(t0[dim - 2][i])

        comm.isend(tl, dest=rank - 1)
        comm.isend(tr, dest=rank + 1)
        comm.isend(tu, dest=rank - m)
        comm.isend(td, dest=rank + m)

        tl = comm.recv(source=rank - 1)
        tr = comm.recv(source=rank + 1)
        tu = comm.recv(source=rank - m)
        td = comm.recv(source=rank + m)

        for i in range(1, dim - 1):
            t0[i][0] = tl.pop()
            t0[i][dim - 1] = tr.pop()
            t0[0][i] = tu.pop()
            t0[dim - 1][i] = td.pop()

        t1 = np.zeros((dim, dim))

        t = []
        for row in range(1, dim - 1):
            trow = []
            for col in range(1, dim - 1):
                r = raw(t0[row, col - 1], t0[row, col + 1], t0[row + 1, col], t0[row - 1, col])
                t1[row, col] = r
                trow.append(r)
            t.append(trow)
        t = np.array(t)
        t0 = t1

list = comm.gather(t, root=0)

# for good printing purpose
comm.Barrier()
#


if rank == 0:
    end = time.time()
    exec_time = end - start;
    print(end - start)
    with open("times.txt", "a") as myfile:
        myfile.write('rozmiar, {}\n'.format(exec_time))


    # import pydevd_pycharm

    # pydevd_pycharm.settrace('localhost', port=36623, stdoutToServer=True, stderrToServer=True)
    print("Gather results from processes")
    # print(list)

    print('Finally\n')


    rows = list[0]
    for i in range(1, size_sqrt):
        rows = np.hstack((rows, list[i]))
    all = rows
    for j in range(2, size_sqrt+1):
        rows = list[size_sqrt*(j-1)]
        for i in range(size_sqrt*(j-1)+1, size_sqrt*(j)):
            rows = np.hstack((rows, list[i]))
        all = np.concatenate((all, rows))

    allb = all[sqrt_pts_per_task:-sqrt_pts_per_task, sqrt_pts_per_task:-sqrt_pts_per_task]
    # print(allb)


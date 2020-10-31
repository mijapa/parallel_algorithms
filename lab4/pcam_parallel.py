import math
import sys

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if math.sqrt(size) % 1 != 0:
    print('World size should be power of integer')


maxiter = 100

size_sqrt = int(math.sqrt(size))
n = 10

sqrt_pts_per_task = int(size_sqrt/n)

if size_sqrt/n % 1 != 0:
    print('World size should be correlated with task size')


lambd = 3
h = 10
g = 0.1
border_value = 0
start_value = 0

# left, right, up, down
neighbours = [rank - 1, rank + 1, rank - n, rank + n]


def raw(left, right, up, down):
    t_ = ((h ** 2 * g) / lambd + left + right + up + down) / 4

    return t_


def isend_to_neighbours(t_):
    for dest in neighbours:
        comm.isend(t_, dest=dest)


def receive_from_neighbours():
    data_ = []
    for source in neighbours:
        data_.append(comm.recv(source=source))
    return data_


t = start_value

for _ in range(maxiter):

    # upper border
    if 0 < rank < n - 1:
        t = border_value
        comm.isend(border_value, dest=rank+n)
        comm.recv(source=rank+n)

    # lower border
    elif n * (n - 1) < rank < n * n - 1:
        t = border_value
        comm.isend(border_value, dest=rank-n)
        comm.recv(source=rank-n)

    # left border
    elif rank % n == 0 and rank != 0 and rank != n * (n - 1):
        t = border_value
        comm.isend(border_value, dest=rank + 1)
        comm.recv(source=rank + 1)

    # right border
    elif rank % n == n - 1 and rank != n - 1 and rank != n * n - 1:
        t = border_value
        comm.isend(border_value, dest=rank-1)
        comm.recv(source=rank-1)

    # corners
    elif rank == 0 or rank == n - 1 or rank == n * (n - 1) or rank == n * n - 1:
        t = 10

    # middle
    else:
        isend_to_neighbours(t)
        data = receive_from_neighbours()
        t = raw(data.pop(), data.pop(), data.pop(), data.pop())


list = comm.gather(t, root=0)

# for good printing purpose
sys.stdout.flush()
comm.Barrier()
#
if rank == 0:
    import pydevd_pycharm

    pydevd_pycharm.settrace('localhost', port=42349, stdoutToServer=True, stderrToServer=True)
    print("Gather results from processes")
    print(list)
    array = np.asarray(list).reshape((n, n))
    print(array)

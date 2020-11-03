#!/usr/bin/env python
import math
import os
import time
import numpy as np
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = 0

if rank == 0:
    start = time.time()

    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace('localhost', port=40625, stdoutToServer=True, stderrToServer=True)

if rank == 0:
    if len(sys.argv) != 4:
        print("There should be three arguments: task size, iterations and cores")
        exit(1)

maxiter = int(sys.argv[1])
z_n = int(sys.argv[2])
p = int(sys.argv[3])

elements_per_task = int(z_n / size)
n = elements_per_task * size

lambd = 3
h = 10
g = 0.1

border_value = 0
start_value = 0


# left, right, up, down
def raw(left, right, up, down):
    t_ = ((h ** 2 * g) / lambd + left + right + up + down) / 4
    return t_


def calculate(t0):
    t1 = np.zeros(t0.shape)
    for row in range(1, t0.shape[0] - 1):
        for col in range(1, t0.shape[1] - 1):
            t1[row, col] = raw(t0[row, col - 1], t0[row, col + 1], t0[row + 1, col], t0[row - 1, col])
    return t1


sq = int(math.sqrt(n)/size)
stripe_v = int(elements_per_task / sq)
stripe_h = int(elements_per_task / stripe_v)

t = np.zeros((stripe_h+2, stripe_v+2))
# initial_table = np.arange(stripe_h * stripe_v).reshape((stripe_h, stripe_v))
# t[1:stripe_h+1, 1:stripe_v+1] = initial_table

for _ in range(maxiter):

    if rank + 1 < size:
        penultimate_strip = t[stripe_h, 1:stripe_v+1]
        comm.isend(penultimate_strip, dest=rank + 1)

        recived = comm.recv(source=rank + 1)
        t[stripe_h + 1, 1:stripe_v+1] = recived

    if rank - 1 >= 0:
        second_strip = t[1, 1:stripe_v+1]
        comm.isend(second_strip, dest=rank - 1)

        recived = comm.recv(source=rank - 1)
        t[0, 1:stripe_v+1] = recived

    t = calculate(t)

t = t[1:stripe_h+1, 1:stripe_v+1]
array = comm.gather(t, root=0)

comm.Barrier()

if rank == 0:
    end = time.time()
    exec_time = end - start

    filename = "times_local.csv"
    if not os.path.exists(filename):
        with open(filename, "a") as myfile:
            myfile.write('z_n,n,i,p,t\n')
    with open(filename, "a") as myfile:
        myfile.write('{z_n},{n},{i},{p},{t}\n'.format(z_n=z_n, n=n, i=maxiter, p=p, t=exec_time))

    array = np.concatenate([i for i in array])
    print("Done")

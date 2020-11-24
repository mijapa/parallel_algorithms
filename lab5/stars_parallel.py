#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='Calculate star positions')
parser.add_argument('iterations', type=int, help='number of iterations of position calculations')
parser.add_argument('n_stars', type=int, help='problem size - number of stars')
args = parser.parse_args()

import math
import os
import random
import time
import numpy as np
from mpi4py import MPI

from aa import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

maxiter = args.iterations
n_stars = args.n_stars
n = z_n = n_stars
p = size

stars_per_node = int(n_stars / p)

start = 0

if rank == 0:
    start = time.time()

    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace('localhost', port=40625, stdoutToServer=True, stderrToServer=True)

# stars set
stars = np.zeros((0, 3), dtype=[('mass', np.double),
                                ('r', [('x', np.double), ('y', np.double), ('z', np.double)]),
                                ('v', [('vx', np.double), ('vy', np.double), ('vz', np.double)])])

for _ in range(stars_per_node):
    newrow = [(random.randrange(0, 100),
               (random.randrange(0, 100), random.randrange(0, 100), random.randrange(0, 100)),
               (random.randrange(0, 100), random.randrange(0, 100), random.randrange(0, 100)))]
    stars = np.vstack([stars, newrow])
print(stars)

# stars buffer
stars_buf = stars.copy()

# accumulator
accu = np.zeros((stars_per_node), dtype=[('ax', np.double), ('ay', np.double), ('az', np.double)])
print(accu)


def calculate_r(star_a, star_b):
    ri_rj = math.sqrt((star_a[1][0] - star_b[1][0]) ** 2 +
                      (star_a[1][1] - star_b[1][1]) ** 2 +
                      (star_a[1][2] - star_b[1][2]) ** 2)
    return ri_rj


def calculate(old_ac, star_a, star_b):
    # |ri-rj|
    ri_rj = calculate_r(star_a, star_b)
    # Mj/(|rj-ri|^3)*(xj-xi)
    ax = star_b[0] / ri_rj ** 3 * (star_b[1][0] - star_a[1][0])
    ay = star_b[0] / ri_rj ** 3 * (star_b[1][1] - star_a[1][1])
    az = star_b[0] / ri_rj ** 3 * (star_b[1][2] - star_a[1][2])
    new_ac = (ax, ay, az)
    return new_ac


def calculate_new_position(stars, a):
    return 0

coord = stars.copy()
for _ in range(maxiter):
    # calculate interactions of own stars
    for i in range(stars_per_node):
        for j in range(stars_per_node):
            if i == j:
                continue
            accu[i] = calculate(accu[i], stars[i], stars_buf[j])

    # calculate interactions witch other stars
    for p in range(size-1):
        # send to right neighbour
        dest = rank + 1
        if dest == size:
            dest = 0
        comm.isend(stars_buf, dest=dest)

        # receive from left neighbour
        source = rank - 1
        if source < 0:
            source = size - 1
        stars_buf = comm.recv(source=source)

        # calculate interactions
        # add interactions to accumulator
        for i in range(stars_per_node):
            for j in range(stars_per_node):
                accu[i] = calculate(accu[i], stars[i], stars_buf[j])

    array = comm.gather(accu, root=0)
    # if rank == 0:
    #     coord.append(calculate_new_position(array))

if rank == 0:
    print("coord: {}".format(coord))
    print("array: {}".format(array))

    points, = ax.plot(x, y, z, '*')
    ani = animation.FuncAnimation(fig, update_points, frames=10, fargs=(x, y, z, points))

    end = time.time()
    exec_time = end - start

    filename = "times.csv"
    if not os.path.exists(filename):
        with open(filename, "a") as myfile:
            myfile.write('z_n,n,i,p,t\n')
    with open(filename, "a") as myfile:
        myfile.write('{z_n},{n},{i},{p},{t}\n'.format(z_n=z_n, n=n, i=maxiter, p=p, t=exec_time))

    # array = np.concatenate([i for i in array])
    print("Done")

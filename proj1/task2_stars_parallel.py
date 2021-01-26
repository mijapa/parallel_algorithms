#!/usr/bin/env python
# mpirun -np 4 python task2_stars_parallel.py
import argparse
import operator

parser = argparse.ArgumentParser(description='Calculate star positions')
parser.add_argument('iterations', type=int, help='number of iterations of position calculations')
parser.add_argument('n_stars', type=int, help='problem size - number of stars')
# Simulate the args to be expected...
# argv = ["1", "4"]
# args = parser.parse_args(argv)
args = parser.parse_args()


import math
import os
import random
import time
import numpy as np
from mpi4py import MPI


# from aa import *


def calculate_ri_rj(star_a, star_b):
    # print(f'{rank} star_a {star_a}', flush=True)
    # print(f'{rank} star_b {star_b}', flush=True)
    ri_rj = math.sqrt((star_a[1][0] - star_b[1][0]) ** 2 +
                      (star_a[1][1] - star_b[1][1]) ** 2 +
                      (star_a[1][2] - star_b[1][2]) ** 2)
    # print(f'ri_rj {ri_rj}', flush=True)
    return ri_rj


def calculate_interactions(old_ac, star_a, star_b):
    import operator
    # |ri-rj|
    ri_rj = calculate_ri_rj(star_a, star_b)
    # G * Mj/(|rj-ri|^3)*(xj-xi)
    ax = G *star_b[0] / ri_rj ** 3 * (star_b[1][0] - star_a[1][0])
    ay = G *star_b[0] / ri_rj ** 3 * (star_b[1][1] - star_a[1][1])
    az = G *star_b[0] / ri_rj ** 3 * (star_b[1][2] - star_a[1][2])
    new_ac = tuple(map(operator.add, old_ac, (ax, ay, az)))
    return new_ac


def calculate_new_position(stars, a):
    for i in range(len(stars)):
        # Euler-Cromer method variant
        # x1 = x0 + v0*dt
        # v1 = v0 + a0*dt
        # print(stars[i])
        # print(a[i])
        stars[i][1] = (stars[i][1][0] + stars[i][2][0] * dt,
                       stars[i][1][1] + stars[i][2][1] * dt,
                       stars[i][1][2] + stars[i][2][2] * dt)
        stars[i][2] = (stars[i][2][0] + a[i][0] * dt,
                       stars[i][2][1] + a[i][1] * dt,
                       stars[i][2][2] + a[i][2] * dt)
    return stars


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"word size: {size}")

maxiter = args.iterations
n_stars = args.n_stars
n = z_n = n_stars
p = size
dt = 1
# todo zmień stałą grawitacji na coś sensownego
G = 1

stars_per_node = int(n_stars / p)

if stars_per_node == 0:
    print('too few stars!!!')
    exit(1)


def generate_all_stars():


    stars = []
    for _ in range(size):
        node_stars = np.zeros((0, 3), dtype=[('mass', np.double),
                                             ('r', [('x', np.double), ('y', np.double), ('z', np.double)]),
                                             ('v', [('vx', np.double), ('vy', np.double), ('vz', np.double)])])
        for _ in range(stars_per_node):
            newrow = [(random.randrange(0, 1000),
                       (random.randrange(0, 1000), random.randrange(0, 1000), random.randrange(0, 1000)),
                       (random.randrange(0, 1000), random.randrange(0, 1000), random.randrange(0, 1000)))]
            node_stars = np.vstack([node_stars, newrow])

        stars.append(node_stars)

    # przykładowe 4 gwiazdy
    # stars = [[[100000, (1, 1, 1), (1, 1, 1)]],
    #       [[1, (2, 2, 2), (2, 1000, 2)]],
    #       [[100000, (3, 3, 3), (3, 3, 3)]],
    #       [[1, (4, 4, 4), (4, 1000, 4)]]]
    return stars


if rank == 0:
    start = time.time()

    all_stars = generate_all_stars()
    # print(all_stars, flush=True)
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace('localhost', port=40625, stdoutToServer=True, stderrToServer=True)
else:
    all_stars = 0

stars = np.empty(1, dtype=object)

stars = comm.scatter(all_stars, root=0)

'''
Każdy z procesów zawiera równoliczny zbiór gwiazd N/p, bufor na gwiazdy przechodnie, bufor na akumulator przechodni oraz akumulator swoich interakcji. 
'''

# stars set

# print(f"{rank} stars: {stars}", flush=True)

'''
Początkowo w buforze przechodnim znajdują się własne gwiazdy danego procesu. 
'''
# stars buffer
stars_buf = stars.copy()

# accumulator
accu = np.zeros((stars_per_node), dtype=[('ax', np.double), ('ay', np.double), ('az', np.double)])
# print(f"{rank} accu: {accu}", flush=True)

# accu buffer
transitional_accu = accu.copy()



for _ in range(maxiter):
    # calculate interactions of own stars
    for i in range(stars_per_node):
        for j in range(stars_per_node):
            if i == j:
                continue
            accu[i] = calculate_interactions(accu[i], stars[i], stars[j])
            # print(f'{rank} accu[{i}] {accu[i]}', flush=True)
    # print(f"{rank} accu: {accu}", flush=True)
    '''
    każda gwiazda ma swój akumulator do którego zbierane są przyspieszenia wynikające z oddziaływania z innymi gwiazdami
    Obliczenie przyspieszenia dla każdej z gwiazd wymaga danych od wszystkich innych gwiazd (ilość obliczeń O(N^2)) 
    '''
    ##### PARALLEL

    # calculate interactions witch other stars
    '''
    Następnie powtarza floor(p/2) razy następujące czynności: 
    '''
    for p in range(math.floor(size/2)):
        # send stars and accu to left neighbour (on the right side)
        '''
        1. przesyła do lewego sąsiada przechodnią porcję (N/p) gwiazd oraz skojarzony z nimi akumulator przechodni. 
        '''
        dest = rank + 1
        if dest == size:
            dest = 0
        comm.isend((stars_buf, transitional_accu), dest=dest)
        # print(f'{rank} after isend stars_buf {stars_buf} to {dest}', flush=True)

        # receive from left neighbour
        '''
        2. odbiera porcję (N/p) gwiazd oraz skojarzony z nimi akumulator przechodni od prawego sąsiada do swoich buforów 
        '''
        source = rank - 1
        if source < 0:
            source = size - 1
        (stars_buf, transitional_accu) = comm.recv(source=source)
        # print(f'{rank} after recv stars_buf {stars_buf} from {source}', flush=True)

        # calculate interactions
        # add interactions to accumulator
        '''
        3. oblicza interakcje swoich gwiazd i gwiazd otrzymanych
        4. dodaje obliczona interakcje do swojego akumulatora
        5. dodaje obliczone interakcje do bufora przechodniego  
        '''
        for i in range(stars_per_node):
            for j in range(stars_per_node):
                accu[i] = calculate_interactions(accu[i], stars[i], stars_buf[j])
                transitional_accu[i] = calculate_interactions(transitional_accu[i], stars[i], stars_buf[j])
    ##### END PARALLEL

    '''
    Po ostatnim kroku wysyła akumulator przechodni do właściciela gwiazd ostatniej przechodniej porcji. 
    '''
    last_portion_owner = (rank + math.floor(size/2)) % size
    comm.isend(transitional_accu, dest=last_portion_owner)
    # print(f'{rank} after isend transitional_accu {stars_buf} to {last_portion_owner}', flush=True)

    trans_accu_source = (rank + math.floor(size/2)) % size
    # print(f'{rank} before recv transistional_accu from {trans_accu_source}', flush=True)
    transitional_accu = comm.recv(source=trans_accu_source)
    # print(f'{rank} after recv transitional_accu {transitional_accu} from {trans_accu_source}', flush=True)

    accu = np.append(accu, transitional_accu)

    # print(f"{rank} accu: {accu[0]}", flush=True)
    # print(f"{rank} transitional_accu: {transitional_accu[0]}", flush=True)

    # accu1 = accu
    # accu = []
    # for i in range(3):
    #     accu.append(accu1[0][i]+transitional_accu[0][i])
    # # accu = tuple(map(operator.add, accu, transitional_accu))
    # accu = [accu]

    # calculate new positions for your own stars
    # stars = calculate_new_position(stars, accu)

    array = comm.gather(stars, root=0)
    accu = comm.gather(accu, root=0)
    # if rank == 0:
    #     print("accu in time: {}".format(array))
        # coord = np.vstack([coord, calculate_new_position(coord, array)])

if rank == 0:

    end = time.time()
    exec_time = end - start

    filename = "times2.csv"
    # create file and make header if not exist
    if not os.path.exists(filename):
        with open(filename, "w") as myfile:
            myfile.write('z_n,n,i,p,t\n')
    with open(filename, "a") as myfile:
    # with open(filename, "w") as myfile:
        myfile.write('{z_n},{n},{i},{p},{t}\n'.format(z_n=z_n, n=n, i=maxiter, p=size, t=exec_time))

    # array = np.concatenate([i for i in array])
    # print("array: {}".format(array), flush=True)
    accu = np.concatenate([i for i in accu])
    print("accu: {}".format(accu), flush=True)

    print("Done")

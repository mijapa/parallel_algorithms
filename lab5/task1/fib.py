#!/usr/bin/env python

import sys

import numpy
from mpi4py import MPI

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

n = numpy.array(0, dtype='i')
comm.Bcast([n, MPI.INT], root=0)
# print(f"hello fib after recv {n}", flush=True)

if n < 2:
    '''wyślij n do rodzica'''
    # print(f'<2 {n}', flush=True)
    comm.Reduce([n, MPI.INT], None,
                op=MPI.SUM, root=0)
else:
    '''utworz nowy proces liczacy Fib(n-1)'''
    comm1 = MPI.COMM_SELF.Spawn(sys.executable,
                                args=['fib.py'],
                                maxprocs=1)
    N_1 = numpy.array(n - 1, 'i')
    comm1.Bcast([N_1, MPI.INT], root=MPI.ROOT)

    '''utworz nowy proces liczacy Fib(n-2)'''
    comm2 = MPI.COMM_SELF.Spawn(sys.executable,
                                args=['fib.py'],
                                maxprocs=1)
    N_2 = numpy.array(n - 2, 'i')
    comm2.Bcast([N_2, MPI.INT], root=MPI.ROOT)

    '''odbierz wynik od procesu  liczącego  Fib(n-1)'''
    x = numpy.array(0, 'i')
    # print(f'x {x}', flush=True)
    comm1.Reduce(None, [x, MPI.INT],
                 op=MPI.SUM, root=MPI.ROOT)

    '''odbierz wynik od procesu  liczącego  Fib(n-2)'''
    y = numpy.array(0, 'i')
    # print(f'y {y}', flush=True)
    comm2.Reduce(None, [y, MPI.INT],
                 op=MPI.SUM, root=MPI.ROOT)

    '''fibn = x + y;'''
    fib = numpy.array(x + y, 'i')
    # print(f'fib sum {fib}', flush=True)

    '''wyslij fibn do rodzica'''
    comm.Reduce([fib, MPI.INT], None,
                op=MPI.SUM, root=0)
comm.Disconnect()

#!/usr/bin/env python
# mpirun -np 1 python task1.py 5
import argparse
import sys

import numpy
from mpi4py import MPI
'''
Dynamiczne zarządzanie procesami - funkcja  MPI.COMM_SELF.Spawn

W MPI (od standardu 2.0) mamy wsparcie dla dynamicznego zarządzania procesami https://mpi4py.readthedocs.io/en/stable/tutorial.html#dynamic-process-management

Korzystając z funkcji MPI.COMM_SELF.Spawn oraz przykładu z linku powyżej napisz program w MPI liczący ciąg Fibonacciego wg pseudocodu:

obliczanie Fib(n) {
   if (n < 2)
        wyslij n do rodzica
    else {
        utworz nowy proces liczacy Fib(n-1)
        utworz nowy proces liczacy Fib(n-2)
        x=odbierz wynik od procesu  liczącego  Fib(n-1)
        y=odbierz wynik od procesu  liczącego  Fib(n-2)
        fibn = x + y;
        wyslij fibn do rodzica
     }
}

Zaalokuj rozsądną liczbę rdzeni na Zeusie i przetestuj jak dla tej liczby rdzeni zmienia się czas wykonania w zależności od liczby n. Narysuj wykresy
Punktacja:
działający program 1 pkt
wykresy 1 pkt
'''
'''
chmod +x ./task1.py
mpirun -np 1 ./task1.py 2
'''

parser = argparse.ArgumentParser(description='Calculate fibonacci number')
parser.add_argument('position', type=int, help='Fibonacci sequence position to calculate')
args = parser.parse_args()
n = args.position


comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['fib.py'],
                           maxprocs=1)


N = numpy.array(n, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)


fib = numpy.array(0, 'i')
comm.Reduce(None, [fib, MPI.INT],
            op=MPI.SUM, root=MPI.ROOT)
print(f'total fib {fib}', flush=True)

comm.Disconnect()

import sys

import numpy
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

randNum = numpy.zeros(1)

# range A=[2..N]
N = 100
# range B=[2,floor(sqrt(N))]
S = int(numpy.sqrt(N))

if rank == 0 & size > N - S:
    print('SOME PROCESSES WILL DO NOTHIG!!!', flush=True)

# 1. find prime numbers in range B=[2,S]
print('Finding prime numbers in range B=[2,{}]'.format(S))
numbers = []
for i in range(2, S):
    numbers.append(i)

# print(numbers)

for i in numbers:
    for j in range(2, int(S / 2)):
        if i * j in numbers:
            numbers.remove(i * j)

primes = numbers
print(primes)

# 2. calculate subrange for certain process c=[floor(sqrt(n))+1,n]
# domain decomposition
print('Calculating subranges')
range_lenght = (N - S) / size
subrange_low = int(S + 1 + range_lenght * rank)
subrange_high = int(S + 1 + range_lenght * (rank + 1))

# 3. calculate prime numbers in subrange
print('Finding prime numbers in subrange [{low}, {high}]'.format(low=subrange_low, high=subrange_high))

subrange_primes = []
reminder = 0
for i in range(subrange_low, int(subrange_high)):
    for j in primes:
        reminder = i % j
        if reminder == 0:
            break
    if reminder != 0:
        subrange_primes.append(i)

print(subrange_primes)
print("Primes in subrange: {}".format(len(subrange_primes)))

# 4. gathering results
lists = comm.gather(subrange_primes, root=0)

# for good printing purpose
sys.stdout.flush()
comm.Barrier()
#

if rank == 0:
    for li in lists:
        primes = primes + li
    print('Gathered primes:')
    print(primes)
    print("Total primes: {}".format(len(primes)))

import sys
import numpy as np
from mpi4py import MPI
sys.path.append('../../')
from mpi_utilities import *

rank, size, comm = initialize_mpi(debug=False)
root=0

comm.Barrier()

# this array lives on each processor
data = np.zeros(5)
for i in range(comm.rank, len(data), comm.size):
    # set data in each array that is different for each processor
    data[i] = i

print("Proc", rank, "data= ", data)
    
comm.Barrier()

# the 'totals' array will hold the sum of each 'data' array
if rank==0:
    # only processor 0 will actually get the data
    totals = np.zeros_like(data)
else:
    totals = None

# use MPI to get the totals
totals = mpi_reduce_np_array_in_place(data, comm, root=0,
                                      oper=MPI.SUM, debug=False,
                                      pre="Proc "+str(rank))
if rank == 0:
    print("Proc", rank, "reduced data= ", totals)
comm.Barrier()


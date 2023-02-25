import sys
import numpy as np
from mpi4py import MPI
sys.path.append('../')
from mpi_utilities import *

rank, size, comm = initialize_mpi(debug=False)
root=0

comm.Barrier()

data = {}
for i in range(rank, size):
    data[str(rank)] = rank*100
print("Proc", rank, "data=", data)

comm.Barrier()
    
mydict = mpi_reduce_dict_to_root(data, comm, root=0)

if(rank==0):
    print("Proc", rank, "data=", mydict)

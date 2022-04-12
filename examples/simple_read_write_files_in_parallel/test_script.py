import sys
sys.path.append('../../')
from mpi_utilities import *
from time import time

def read_write_file(input_files, proc_num):
    for file_ in input_files:
        with open(file_,'w') as f:
            f.write("proc num:"+str(proc_num))
            f.write("this file is:"+str(file_))
            tick = time()
            f.write("time is:"+str(tick))
            f.close()
    return

root=0
rank, size, comm = initialize_mpi()

file_dir = './files/'
generic_file_name = 'file'
file_names, start, end = gather_files(generic_file_name, file_dir,
                                      suffix='.txt', start=None, end=None, debug=False)

args = [rank]
kwargs={}
chunk_size=1

comm.Barrier()
perform_task_in_parallel(read_write_file, args, kwargs,
                         file_names, chunk_size, rank, size, comm,
                         root=0, debug=False)

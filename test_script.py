from mpi_utilities import *
from time import time

def read_write_file(input_file, proc_num):
    with open(input_file,'w') as f:
        f.write("proc num:", proc_num)
        f.write("this file is:", input_file)
        f.write("time is:", time())
        f.close()
    return

file_names, start, end = gather_files(generic_file_name, file_dir, suffix='',
                 start=None, end=None, debug=False)
rank, size, comm = initialize_mpi()
args = [rank]
kwargs={}
chunk_size=1

perform_task_in_parallel(read_write_file, args, kwargs,
                         file_names, chunk_size, rank, size, comm,
                         root=0, debug=False)

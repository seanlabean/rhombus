import numpy as np
import sys
import inspect
from glob import glob
from mpi4py import MPI
# Note MPI_Init is called on mpi4py import.
# MPI_Finalize is called on script end.

def gather_files(generic_file_name, file_dir, suffix='',
                 start=None, end=None, debug=False):

    """
    This function globs up all files that
    match the glob pattern of generic_file_name*
    that are in file_dir

    Keyword arguments:
    generic_file_name -- File name to search for,
                         like flash_hdf5_plt_cnt_
                         would search *flash_hdf5_plt_cnt*
                         or plt would search *plt*.
    file_dir  -- Directory to glob inside.
    suffix    -- Any string that comes after the numbering of the file.
    start     -- String number that matches the starting file number,
                 i.e. flash_hdf5_plt_cnt_0104 would mean start='104'.
    end       -- String number that matches the ending file number.
    debug     -- Switch on debugging print statements.

    Returns:
    files     -- Sorted globbed list of file names
                 truncated from start to end.
    """

    joined_file_name = file_dir+'*'+generic_file_name+'*'
    files = glob(joined_file_name) # Get the files.
    files.sort() # Sort them!
    if (debug):
        print(generic_file_name)
        print(file_dir)
        print(joined_file_name)
        print('start=', start)
        print('end=', end)
        print(files)
    
    if (start==None):
        start = 0
    else: # Find the starting file index.
        start = str(start).zfill(4)
        start = glob(joined_file_name+start+suffix)
        start = np.where(np.array(files)==start[0])[0]
        start = start[0].astype(int)

    if (end==None): 
        end = len(files)
    else: # Find the ending file index.
        end = str(end).zfill(4)
        end = glob(joined_file_name+end+suffix)
        end = np.where(np.array(files)==end[0])[0]
        end = end[0].astype(int)+1
        
    files = files[start:end]

    return files, start, end

def initialize_mpi(debug=False):
    """
    Only initialize MPI and do
    nothing else.

    Keyword arguments:
    debug -- Switch on some print statements.

    Returns:
    rank -- Processor rank.
    size -- Number of total processors.
    comm -- The communicator for MPI.
    """
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if (debug): print("Size =", size)
    if (debug): print("Rank =", rank)

    return rank, size, comm

def send_data(data_chunk, destination, comm, tag=0):
    """
    Sends data from current proc to
    destination proc.
    
    Arguments:
    data_chunk  -- Data to be sent.
    destination -- Receiving processor id
    comm        -- MPI.COMM_WORLD object
    tag         -- Data label (used as current 
                   index in these routines)
    
    """
    comm.send(data_chunk, dest=destination, tag=tag)

    return

def wait_for_message(recv_buff, status, comm, debug=False):
    """
    Establish a buffer to
    receive waiting for a request
    for the next instruction.
    
    Arguments:
    recv_buff -- Empty receive array.
    status    -- MPI.status() object.
    comm      -- MPI.COMM_WORLD object.
    debug     -- Switch on debugging print 
                 statements.
    
    Returns:
    recv_buff -- MPI receive buffer.
    source    -- expected source of incoming 
                 message (set to ANY proc).
    tag       -- expected tag of message 
                 (set to ANY tag).
    
    """
    if (debug): print("Root in wait_for_message")
    comm.Recv(recv_buff, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    source = status.Get_source()
    tag    = status.Get_tag()
    if (debug):
        print("source=", source)
        print("tag=", tag)

    return recv_buff, source, tag

#def ask_for_work(msg, destination, comm, tag=0):
    # Send message to root proc
    # that labels this worker proc
    # as needing something to do.
#    comm.Send([msg, 1, MPI.INT], destination, tag=tag)

#    return

def get_chunk(all_data, current_index, chunk_size):
    """
    Gather subset of total work to be
    sent to a worker processor.

    Arguments:
    all_data      -- Total input data to 
                     parallelized function.
    current_index -- Current index of 
                     input data.
    chunk_size    -- Amount of data to be 
                     delived to each proc.

    Returns:
    chunk         -- Subset of input data 
                     to be worked on.
    new_index     -- first index of the next
                     chunk-to-be.
    
    
    """
    ci = current_index
    ni = min(chunk_size, len(all_data[ci:])) # next index
    chunk = all_data[ci:ci+ni]
    new_index = ci+ni
    return chunk, new_index

def needs_current_index_arg(function):
    """
    Gets keyword argument (kwarg)
    current_index from the function 
    we want to parallelize.

    Arguments:
    function -- Parallelized function.
    
    Returns:
    current_index function arg.
    """
    # See https://stackoverflow.com/questions/196960/
    #    can-you-list-the-keyword-arguments-a-function-receives 
    args, varargs, varkw, defaults = inspect.getargspec(function)
    return ('current_index' in args)

def perform_task_in_parallel(function, args, kwargs, all_data,
                             chunk_size, rank, size, comm,
                             root=0, debug=False):

    """
    Perform a function on all_data
    in parallel. This breaks the data up
    into chunks on the root processor, who
    then hands out the chunks to the other
    processors as soon as they are free to
    do the work.
    This task based parallel programming
    ensures that all processors are always
    busy, unlike data parallelization.
    Note the work function takes args and kwargs
    in the form of 
    function(local_data, *args, **kwargs)
    where local_data is the chunk sent by
    the root, args is a list of arguments
    to be unrolled in passing, and **kwargs
    is a dictionary of {'kw':'arg'} pairs
    that is unrolled to kw=arg in the
    function call.

    Arguments:
    function   -- Function to be parallelized.
    args       -- List of arguments in function.
    kwargs     -- Dictionary of kwargs in function.
    all_data   -- Data that is to be distributed.
    chunk_size -- Amount of data to be sent to
                  each processor.
    rank       -- Processor identification.
    size       -- Number of processors in use.
    comm       -- MPI.COMM_WORLD object.
    root       -- root (supervisor) proc id.
    debug      -- turn print statements.
    """

    pre = "Proc", rank

    if (debug): print(pre, "Entering perform_task_in_parallel.")
    
    # Keep a status object around for async comms.
    status = MPI.Status()

    send_me_work = np.array([999], dtype=int)
    everyone_all_done = np.array([100], dtype=int)
    recv_buff  = np.zeros(1, dtype=int)
    work_tag = 999
    done_tag = 100
    
    idle    = 200
    working = 999
    done    = 100
    
    # Using integer array to track which
    # procs are working, idle, or done.
    # Worker processors start idle.
    procs_status = np.zeros(size-1)
    procs_status[:] = idle

    not_done = True
    current_index = 0

    counter = 0
    frac_done = 0.

    comm.Barrier()

    if (rank == root):
        number_of_work_units = len(all_data)
        last_data_index = len(all_data)-1
        procs_status[:] = working

    if (rank == root): # Instructions for supervisor processor.
        # Check to see if any procs are still active.
        while (np.array(procs_status != done).any()):
            # Progress tracker from supervisor perspective.
            frac_done = 100.*float(counter)*float(chunk_size)/float(number_of_work_units)
            print("Progress at {:.0f} %".format(frac_done))
            
            # Wait for someone to say they want some work.
            if (debug): print(pre, "Waiting for message asking for work.")
            recv_buff, source, tag = wait_for_message(recv_buff, status,
                                                      comm, debug=debug)
            if (debug): print(pre, "recieved", recv_buff, "from", source)
            
            if (not_done):
                # Get a chunk of data.
                if (debug): print(pre, "Getting data chunk.")
                data_chunk, new_index = \
                    get_chunk(all_data, current_index, chunk_size)
                counter += 1
            # Who did we recieve from, and are they ready for work?
            if (recv_buff == send_me_work and not_done):
                # Tell this processor we are about to send some
                # work.
                comm.Send([send_me_work, 1, MPI.INT],
                           dest=source, tag=current_index)
                # Send the work.
                if (debug): print(pre, "Sending", source, "work =", data_chunk)
                send_data(data_chunk, source, comm, tag=current_index)
                procs_status[source-1] = working
            else:
                # Tell this proc we're done.
                if (debug): print(pre, "Telling", source, "we're done.")
                comm.Send([everyone_all_done,  MPI.INT], dest=source, tag=done_tag)
                procs_status[source-1] = done
                if (debug): print(pre, "proc_status array =", procs_status)
            
            current_index = new_index
            if (current_index > last_data_index):
                not_done = False

            if (debug): print("Counter = ", counter)

    else: # Instructions for worker processor.

        while not_done:

            # Ask the root for more work.
            if (debug): print(pre, "Asking root for work.")
            comm.Send([send_me_work, 1, MPI.INT], dest=root, tag=0)
            # Wait for a reply, then check the reply size
            # to determine if this is work or just a message
            # tell us we are all done.
            if (debug):
                print(pre, "Getting message about if we are done.")
                # For serious debugging, uncomment these lines.
                comm.Probe(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG, status=status)
                print(pre, "source = ", status.Get_source())
                print(pre, "tag    = ", status.Get_tag())
                print(pre, "count  = ", status.Get_elements(MPI.INT))
                print(pre, "size   = ", status.Get_count())
            comm.Recv([recv_buff, MPI.INT], source=root,
                       tag=MPI.ANY_TAG, status=status)
            tag   = status.Get_tag()
            if (debug): print(pre, "Message tag =", tag)
            if (recv_buff != done_tag):
                # Its a data object, just use a regular recieve.
                if (debug): print(pre, "Getting data from root.")
                local_data = comm.recv(source=root, tag=tag, status=status)
                if (debug): print(pre, "Processing data from root.")
                if (needs_current_index_arg(function)):
                    kwargs['current_index']=tag
                if (debug): print("kwargs=", kwargs)
                function(local_data, *args, **kwargs)
            else:
                # Otherwise its a simple one count array message. Recieve that.
                if (debug): print(pre, "Got non-data message from root.")
                # Not used now that we explicitly communicate a message
                # about work status before we send the work chunk.
                #comm.Recv(recv_buff, source=root, tag=MPI.ANY_TAG, status=status)
                if (debug): print(pre, "Root sent", recv_buff)
                if (recv_buff == everyone_all_done):
                    not_done = False
                    if (debug):
                        print(pre, "Hears from root we're all done.")

    
    print(pre, "Exiting from perform_task_in_parallel")
    return

# The following can be thought of as MPI post-processing methods.
# They work to gather information spread across multiple processors
# and 'reduce' the data down to the root node.
#
# Imagine if we had a mosaic of sky picutres in which several procs
# count the number of galaxies in the pics given to them. If we want
# the total number of galaxies in the entire mosaic, we'd let the procs
# do their job then reduce their individual counts to a single value
# (via MPI.SUM method).

def mpi_reduce_np_array_in_place(array, comm,
                                 root=0, oper=MPI.SUM,
                                 debug=False, pre=None):

    """
    Mimics the MPI_reduce operation but with
    a numpy array.

    Arguments:
    array -- Array to be reduced in place.
    comm  -- MPI.COMM_WORLD object.
    root  -- Processor to reduce to. 
             Default is 0 (supervisor).
    oper  -- The reduction operation. 
             Default is MPI.SUM

    Returns:
    array -- Reduced array.
    """

    recv_array = np.zeros_like(array)
    if (debug): print(pre, 'before reduce, array=',array)
    comm.Reduce(array, recv_array, root=root, op=oper)
    if (debug): print(pre, 'after reduce, array=', array)
    if (debug): print(pre, 'after reduce, recv_array=', recv_array)
    array = recv_array.copy()
    del(recv_array)
    return array

def mpi_reduce_dict_to_root(mydict, comm, root=0):
    
    """ 
    For a given dictionary that was created on
    each processor separately during an MPI
    task, this function gathers all the
    dictionaries on the root process
    under the same name. 

    This mimics the MPI_reduce operation.
        
    Arguments:
    mydict -- Dictionary you want to reduce.
    comm   -- MPI.COMM_WORLD object.
    root   -- Location to reduce mydict to. 
              Default is 0 (supervisor).
        
    Returns:
     mydict -- the concatenated dictionary from
               all processes.
    """
    
    rank = comm.Get_rank()
    all_dicts = comm.gather(mydict, root=root)
    
    if (rank == root):
        for item in all_dicts[0:root]: # all but the root
            mydict.update(item)
        for item in all_dicts[root+1:]: # all but the root
            mydict.update(item)
    
    return mydict

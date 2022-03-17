# PythonOpenMPI

A generalizable python-mpi utility for task-based parallel programming.

This implementation of task-based parallel programming consists of one root processor, and any number of worker processors. The root breaks a portion of a job into bite sized chunks (like a single file) which are then sent to the workers. While the workers... well... work, the root sits and waits. When a working finishes with its allotted chunk, it pings the root node and asks for another chunk, which the root node then provides. Therefore no worker is ever left without something to do.

This is fundamentally different and more efficient than data-bases paralllel processing in which an **entire** job is split into n equally sized chunks (where n is the number of processors) and sent to the worker processors. In this method, when a worker is done processing, it does not need to ask the root node for any more work (since everything has already been distributed to the workers). Therefore, although the task is being completed in parallel, there is a chance that workers will be left idle while they wait for other workers to finish.


This repository is meant to aid other students at Drexel University.

Contributors:

- Sean C. Lewis (owner)

- Evan Arena
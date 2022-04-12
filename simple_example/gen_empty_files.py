"""
 This script generates 10 empty files, to be written to in the example module
"""

import sys,os

# Create subdirectory for example files
path_to_files = 'files/'
if not os.path.exists(path_to_files):
    os.mkdir(path_to_files)
    
# Number of files
N = 10

# Create files
for i in range(N):
    f_i = open(path_to_files+'file_'+str(i)+'.txt', 'w')
    f_i.write('')
    f_i.close()

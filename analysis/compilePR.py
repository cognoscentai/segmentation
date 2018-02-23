from PixelEM import * 
import pandas as pd
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
from sample_worker_seeds import sample_specs
sample_lst = sample_specs.keys()
print "Compiling the output from .json to one single csv file for each algo (should take ~1min)"
import sys
algorithms = [sys.argv[1]]
#algorithms = ["GTLSA", "isoGTLSA", "GT", "isoGT", "basic","MV","isobasic"]

for algo in algorithms:
    compile_PR(mode=algo, ground_truth=False)
    #compile_PR(mode=algo, ground_truth=True)

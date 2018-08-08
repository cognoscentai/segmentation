from PixelEM import * 
import pandas as pd
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
from sample_worker_seeds import sample_specs
sample_lst = sample_specs.keys()
rand_object_lst = [47, 18,10, 43, 44, 31, 32, 6, 20, 27] 
for sample in ['5workers_rand8','10workers_rand0','30workers_rand0', '20workers_rand1', '15workers_rand2','25workers_rand1']:#tqdm(sample_specs.keys()):
    for objid in rand_object_lst:
        print sample+":"+str(objid)
        binarySearchDeriveGTinGroundTruthExperiments(sample, objid, "basic",exclude_isovote=False,rerun_existing=True,compareWith="gt")
        binarySearchDeriveGTinGroundTruthExperiments(sample, objid, "GT",exclude_isovote=False,rerun_existing=True,compareWith="gt")
        binarySearchDeriveGTinGroundTruthExperiments(sample, objid, "GTLSA", exclude_isovote=False,rerun_existing=True,compareWith="gt")
        binarySearchDeriveGTinGroundTruthExperiments(sample, objid, "GT",exclude_isovote=True,rerun_existing=True,compareWith="gt")
        binarySearchDeriveGTinGroundTruthExperiments(sample, objid, "GTLSA", exclude_isovote=True,rerun_existing=True,compareWith="gt")

        binarySearchDeriveGTinGroundTruthExperiments(sample, objid, "basic",exclude_isovote=False,rerun_existing=True,compareWith="MV")
        binarySearchDeriveGTinGroundTruthExperiments(sample, objid, "GT",exclude_isovote=False,rerun_existing=True,compareWith="MV")
        binarySearchDeriveGTinGroundTruthExperiments(sample, objid, "GTLSA", exclude_isovote=False,rerun_existing=True,compareWith="MV")
        binarySearchDeriveGTinGroundTruthExperiments(sample, objid, "GT",exclude_isovote=True,rerun_existing=True,compareWith="MV")
        binarySearchDeriveGTinGroundTruthExperiments(sample, objid, "GTLSA", exclude_isovote=True,rerun_existing=True,compareWith="MV")

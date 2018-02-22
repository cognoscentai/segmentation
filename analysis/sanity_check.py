from sample_worker_seeds import sample_specs
import pandas as pd 
from utils import *
sample_lst = sample_specs.keys()
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40,41,42, 43, 44, 45, 46, 47]

from sample_worker_seeds import sample_specs
sample_lst = sample_specs.keys()
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40,41,42, 43, 44, 45, 46, 47]

def check_sample_cluster():
    df = pd.read_csv("spectral_clustering_all_hard_obj.csv")
    print "Checking that all clustered samples can reconstruct original worker list exactly"
    for sample in sample_lst:
        for objid in object_lst:
            cluster_ids = df[(df["objid"] == objid)].cluster.unique()
            if len(cluster_ids)!=0:
                outdir = tile_and_mask_dir(sample, objid, -1)
                noClust_worker_ids = json.load(open(outdir+"/worker_ids.json"))
                cluster_worker_ids = []
                Nworkers = int(sample.split("workers")[0])
                try:
                    assert len(noClust_worker_ids)==Nworkers
                except(AssertionError):
                    print "length of original cluster not equal to number of workers: "+ sample + ':' + str(objid) 
                for clust_id in list(cluster_ids):
                    worker_ids = np.array(df[(df["objid"] == objid) & (df["cluster"] == clust_id)].wid)
                    #if len(worker_ids) > 1:
                    outdir = tile_and_mask_dir(sample, objid, clust_id)    
                    worker_ids = json.load(open(outdir+"/worker_ids.json"))
                    cluster_worker_ids.extend(worker_ids)
                try:
                    assert set(noClust_worker_ids)==set(cluster_worker_ids)
                except(AssertionError):
                    print "sum of clustered worker list over all clusters not equal to original worker list"
                    print sample + ':' + str(objid) 
                    print noClust_worker_ids
                    print cluster_worker_ids
    print "Success!"
if __name__=="__main__":
    check_sample_cluster()

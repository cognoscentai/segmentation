from PixelEM import *

object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
from sample_worker_seeds import sample_specs
sample_lst = sample_specs.keys()
# sample_size = len(sample_lst)   # if all worker sets
sample_size = 5

# print "1. if directory does not exist, create pixel_em/"
# import os.path
# if not os.path.exists("pixel_em"):
#     os.makedirs("pixel_em")

# print "2. Creating all worker and GT pixel masks (2-3 min)"
# for objid in object_lst:
#     create_all_gt_and_worker_masks(objid)
# Changes start here for the preprocessed cluster 
# sample_lst = sample_lst[:sample_size]
# print "3.Creating megamask (aggregated mask over all workers in that sample) for all sample-objects [mega_mask.pkl, voted_workers_mask.pkl]"
# print "This might take a while (~2hrs)"
# for sample in sample_lst:
#     for objid in object_lst:
#         print sample + ":" + str(objid)
#         create_mega_mask(objid, PLOT=False, sample_name=sample)

df = pd.read_csv("spectral_clustering_all_hard_obj.csv")
'''
print "3.Creating megamask (aggregated mask over all workers in that sample) for all sample-objects [mega_mask.pkl, voted_workers_mask.pkl]"
print "This might take a while (~2hrs)"
for sample in sample_lst:
    for objid in object_lst:
        cluster_ids = df[(df["objid"]==objid)].cluster.unique()
        for cluster_id in cluster_ids:
            worker_ids = np.array(df[(df["objid"]==objid)&(df["cluster"]==cluster_id)].wid)
            print sample + ":" + str(objid) +"; clust:" + str(cluster_id)
            create_mega_mask(objid, worker_ids=worker_ids,cluster_id = cluster_id,PLOT=False, sample_name=sample)
'''
'''
print "4.Creating MV mask (should take 5 min)"
mv_prj_vals=[]
for sample in sample_lst:
     for objid in object_lst:
	 cluster_ids = df[(df["objid"]==objid)].cluster.unique()
         for cluster_id in cluster_ids:
             worker_ids = np.array(df[(df["objid"]==objid)&(df["cluster"]==cluster_id)].wid)
             if len(worker_ids)!=1:
		 print sample + ":" + str(objid)+";clust"+str(cluster_id)
         	 p,r,j = compute_PRJ_MV(sample, objid,cluster_id)
	         mv_prj_vals.append([sample,objid,cluster_id,p,r,j])
mv_df = pd.DataFrame(mv_prj_vals,columns=["sample","objid","clust","MV_precision","MV_recall","MV_jaccard"])
mv_df.to_csv("pixel_em/withClust_MV_PRJ_table.csv")
'''

# from areaMask import *
# print "5.Creating area mask for all sample-objects"
# print "This will also take a while (~5hrs)"
# for sample in tqdm(sample_lst):
#         for objid in object_lst:
#             if os.path.exists("pixel_em/{}/obj{}/tiles.pkl".format(sample, objid)):
#                 print sample+":"+str(objid)+" already exist"
#         else:
#             print sample + ":" + str(objid)
#             create_PixTiles(sample, objid, check_edges=True)
'''
from areaMask import *
print "5.Creating area mask for all sample-objects"
print "This will also take a while (~5hrs)"
for sample in tqdm(sample_lst):
    for objid in object_lst:
        cluster_ids = df[(df["objid"]==objid)].cluster.unique()
        for cluster_id in cluster_ids:
            worker_ids = np.array(df[(df["objid"]==objid)&(df["cluster"]==cluster_id)].wid)
	    if len(worker_ids)!=1:
                print sample + ":" + str(objid)+"clust"+str(cluster_id)
                create_PixTiles(sample, objid, cluster_id,check_edges=True)
'''
###########################################################
# DEBUG PIXTILE OUTPUT (VISUALLY INSPECT)
'''
def tiles2AreaMask(sample, objid):
    tiles = pkl.load(open("pixel_em/{}/obj{}/tiles.pkl".format(sample, objid)))
    mega_mask = pkl.load(open("pixel_em/{}/obj{}/mega_mask.pkl".format(sample, objid)))
    tarea = [len(t) for t in tiles]
    mask = np.zeros_like(mega_mask)
    for tidx in range(len(tiles)):
        for i in list(tiles[tidx]):
                mask[i] = tarea[tidx]
        return mask

test_sample = sample_lst[0]
print 'Testing tiles2AreaMask for', test_sample
mask = tiles2AreaMask(test_sample, 1)
plt.figure()
plt.imshow(mask)
plt.title("Tile index map")
plt.colorbar()
plt.savefig('testing_tiles2AreaMask.png')
plt.close()
'''
###########################################################

# Check number of object that have completed their full run by :
# ls pixel_em/*/obj*/EM_prj_iter2_thresh-4.json |wc -l
# ls pixel_em/*/obj*/GT_EM_prj_iter2_thresh-4.json |wc -l
# ls pixel_em/*/obj*/GTLSA_EM_prj_iter1_thresh-4.json |wc -l
# Use  submitPixelEM.sh to submit all the jobs in parallel for different samples independently

###########################################################
'''
sample = sys.argv[1]
#for sample in tqdm(sample_specs.keys()):

for objid in object_lst:
    print sample+":"+str(objid)
    # if True:
    #   thresh = -4
    for thresh in [-4, -2, 0, 2, 4]:
        do_EM_for(sample, objid, thresh=thresh, rerun_existing=False, compute_PR_every_iter=True, exclude_isovote=False, num_iterations=3)
        do_GT_EM_for(sample, objid, thresh=thresh, rerun_existing=False, exclude_isovote=True, compute_PR_every_iter=True, num_iterations=3)
        do_GT_EM_for(sample, objid, thresh=thresh, rerun_existing=False, exclude_isovote=False, compute_PR_every_iter=True, num_iterations=3)
        do_GTLSA_EM_for(sample, objid, thresh=thresh, rerun_existing=False, compute_PR_every_iter=True, exclude_isovote=True, num_iterations=3)
        do_GTLSA_EM_for(sample, objid, thresh=thresh, rerun_existing=False, compute_PR_every_iter=True, exclude_isovote=False, num_iterations=3)
'''
###########################################################
'''
print "With Cluster version" 
print "Running Ground Truth Experiment to generate pInT and pNotInT"
clust_df = pd.read_csv("spectral_clustering_all_hard_obj.csv")
noClust_obj =[obj for obj in object_lst if obj not in clust_df.objid.unique() ]
sample = sys.argv[1]
print sample
#for sample in tqdm(sample_specs.keys()):
for objid in object_lst:
# first do all the unclustered objects 
#for objid in noClust_obj: 
    print sample +":"+ str(objid)
    GroundTruth_doM_once(sample,objid,cluster_id = "", algo="GTLSA", exclude_isovote=False, rerun_existing=False)
    GroundTruth_doM_once(sample,objid,cluster_id = "", algo="GTLSA", exclude_isovote=True, rerun_existing=False)
'''
'''
# then do all the clustered objects
#for objid in list(clust_df.objid.unique()):
for objid in object_lst:
    cluster_ids = df[(df["objid"]==objid)].cluster.unique()
    for cluster_id in cluster_ids:
        worker_ids = np.array(df[(df["objid"]==objid)&(df["cluster"]==cluster_id)].wid)
        if len(worker_ids)!=1:
            print sample + ":" + str(objid)+"clust"+str(cluster_id)
	    #GroundTruth_doM_once(sample,objid,cluster_id = cluster_id,algo="basic", exclude_isovote=False, rerun_existing=False)
    	    #GroundTruth_doM_once(sample,objid,cluster_id = cluster_id, algo="GT", exclude_isovote=False, rerun_existing=False)
    	    #GroundTruth_doM_once(sample,objid,cluster_id = cluster_id, algo="GTLSA", exclude_isovote=False, rerun_existing=False)
    	    #GroundTruth_doM_once(sample,objid,cluster_id = cluster_id, algo="GT", exclude_isovote=True, rerun_existing=False)
    	    #GroundTruth_doM_once(sample,objid,cluster_id = cluster_id, algo="GTLSA", exclude_isovote=True, rerun_existing=False)
	    GroundTruth_doM_once(sample,objid,cluster_id = cluster_id, algo="GTLSA", exclude_isovote=False, rerun_existing=False)
	    GroundTruth_doM_once(sample,objid,cluster_id = cluster_id, algo="GTLSA", exclude_isovote=True, rerun_existing=False)
'''
###########################################################
# Using different thresholds to get GT of different thresholds
clust_df = pd.read_csv("spectral_clustering_all_hard_obj.csv")
noClust_obj =[obj for obj in object_lst if obj not in clust_df.objid.unique() ]
thresh_lst = [-4, -2, 0, 2, 4]
for sample in tqdm(sample_specs.keys()):
    for objid in object_lst:
    #for objid in noClust_obj:
        print sample+":"+str(objid)
        deriveGTinGroundTruthExperiments(sample, objid, "basic",thresh_lst, exclude_isovote=False,rerun_existing=True)
        deriveGTinGroundTruthExperiments(sample, objid, "GT",thresh_lst, exclude_isovote=False,rerun_existing=True)
        deriveGTinGroundTruthExperiments(sample, objid, "GTLSA",thresh_lst, exclude_isovote=False,rerun_existing=True)
        deriveGTinGroundTruthExperiments(sample, objid, "GT",thresh_lst, exclude_isovote=True,rerun_existing=True)
        deriveGTinGroundTruthExperiments(sample, objid, "GTLSA", thresh_lst, exclude_isovote=True,rerun_existing=True)
'''
# Using different thresholds to get GT of different thresholds
thresh_lst = [-4, -2, 0, 2, 4]
for sample in tqdm(sample_specs.keys()):
    for objid in object_lst:
	cluster_ids = df[(df["objid"]==objid)].cluster.unique()
	for cluster_id in cluster_ids:
            worker_ids = np.array(df[(df["objid"]==objid)&(df["cluster"]==cluster_id)].wid)
            if len(worker_ids)!=1:
		print sample + ";" + str(objid)+"; clust"+str(cluster_id)
	        deriveGTinGroundTruthExperiments(sample, objid, "basic",thresh_lst,cluster_id = cluster_id, exclude_isovote=False,rerun_existing=True)
        	deriveGTinGroundTruthExperiments(sample, objid, "GT",thresh_lst,cluster_id = cluster_id, exclude_isovote=False,rerun_existing=True)
	        deriveGTinGroundTruthExperiments(sample, objid, "GTLSA",thresh_lst,cluster_id = cluster_id, exclude_isovote=False,rerun_existing=True)
       	 	deriveGTinGroundTruthExperiments(sample, objid, "GT",thresh_lst,cluster_id = cluster_id, exclude_isovote=True,rerun_existing=True)
        	deriveGTinGroundTruthExperiments(sample, objid, "GTLSA", thresh_lst,cluster_id = cluster_id, exclude_isovote=True,rerun_existing=True)
'''
'''
# Compiled PRJ written to config::HOME_DIR/analysis/pixel_em/<algoname>_full_PRJ_table.csv
print "Compiling the output from .json to one single csv file for each algo (should take ~1min)"
algorithms = ["GTLSA", "isoGTLSA", "GT", "isoGT", "basic"]
for algo in algorithms:
    # compile_PR(mode=algo, ground_truth=False)
    compile_PR(mode=algo, ground_truth=True)
'''

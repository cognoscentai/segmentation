import pickle as pkl
from PixelEM import *
def greedy(sample,objid,algo,est_type=""):
    tiles = pkl.load(open("../analysis/pixel_em/{}/obj{}/tiles.pkl".format(sample,objid)))
    gt = get_gt_mask(objid)
    if est_type=="ground_truth":
        #using ground truth ia for testing purposes
        gt_idxs = set(zip(*np.where(gt)))
    elif est_type=="worker_fraction":
        worker_mask = pkl.load(open("../analysis/pixel_em/{}/obj{}/voted_workers_mask.pkl".format(sample,objid,algo)))	
	Nworkers = float(sample.split("workers")[0])
    else:
        log_probability_in_mask=pkl.load(open("../analysis/pixel_em/{}/obj{}/{}_p_in_mask_ground_truth.pkl".format(sample,objid,algo)))
        log_probability_not_in_mask =pkl.load(open("../analysis/pixel_em/{}/obj{}/{}_p_not_in_ground_truth.pkl".format(sample,objid,algo)))

    candidate_tiles_lst = []
    metric_lst = []
    ia_lst = []
    picked_tiles = []
    GT_area  = 0. #sum of intersection areas of all tiles
    total_intersection_area = 0. 
    total_outside_area = 0. 

    # compute I/O metric for all tiles
    #for tile in tiles[1:]:#ignore the large outside tile
    for tile in tiles[1:]:
        if est_type=="ground_truth":
            intersection_area = float(len(gt_idxs.intersection(set(tile)))) # exact intersection areas
        else:
	    if est_type =="worker_fraction":
		norm_pInT = len(worker_mask[list(tile)[0]])/Nworkers
            else:
	        pInT = np.exp(log_probability_in_mask[list(tile)[0]]) # all pixels in same tile should have the same pInT
   	        pNotInT = np.exp(log_probability_not_in_mask[list(tile)[0]])
                if pInT+pNotInT!=0:
                    norm_pInT = pInT/(pNotInT+pInT) #normalized pInT
                else: #weird bug for object 18 isoGT case
                    norm_pInT = 1.
            assert norm_pInT<=1 and norm_pInT>=0
            intersection_area = float(len(tile) * norm_pInT) #estimated intersection area
        outside_area = float(len(tile) - intersection_area)
        GT_area+= intersection_area
        if outside_area!=0:
            metric = intersection_area/outside_area
            metric_lst.append(metric)
            candidate_tiles_lst.append(tile)
            ia_lst.append(intersection_area)
        else:# if outside area =0, then tile completely encapsulated by GT, it must be included in picked tiles
            #print "here"
            picked_tiles.append(tile)
            total_intersection_area += intersection_area
            total_outside_area += outside_area

    assert len(metric_lst)==len(candidate_tiles_lst)==len(ia_lst)
    srt_decr_idx = np.argsort(metric_lst)[::-1] # sorting from largest to smallest metric_lst
    #print np.array(metric_lst)[srt_decr_idx]
    jaccard_lst = []
#     if total_area!=0:
#         print "here"
#         prev_jac = ia_cum / total_area
#     else:
#         prev_jac = -10000.
    prev_jacc = total_intersection_area/(total_outside_area+GT_area)
#     prev_jac = -10000.
#    print ia_cum,total_area
#    print prev_jac
    for tidx  in srt_decr_idx:
        tile = candidate_tiles_lst[tidx]
        ia = ia_lst[tidx]
        #if ia!=0:
        temp_new_out_area = total_outside_area + len(tile) - ia
        temp_new_in_area = total_intersection_area + ia 
        jaccard = (temp_new_in_area) / (temp_new_out_area + GT_area)
        if jaccard >= prev_jacc:
            prev_jacc = jaccard
            total_outside_area = temp_new_out_area
            total_intersection_area = temp_new_in_area
            picked_tiles.append(tile)
        else: # stop when jaccard starts decreasing after the addition of a tile
            break
            #continue #for debugging purposes to see how jaccard_lst evolves, technically should break here

    #populate final img with tiles in picked tiles
    gt_est_mask = np.zeros_like(gt)
    for t in picked_tiles:
        for tidx in t:
            gt_est_mask[tidx]=1
    [p, r, j] = faster_compute_prj(gt_est_mask, gt)
    return p,r,j

object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
from sample_worker_seeds import sample_specs
sample_lst = sample_specs.keys()

import pandas as pd 

'''
df_data = []
for sample in tqdm(sample_specs.keys()):
    for objid in object_lst:
        p,r,j = greedy(sample,objid,"","ground_truth")
        df_data.append([sample,objid,"ground truth",p,r,j])
        print sample,objid,p,r,j
df = pd.DataFrame(df_data,columns=['sample','objid','algo','p','r','j'])
df.to_csv("greedy_result_ground_truth.csv",index=None)
'''
df_data = []
for sample in tqdm(sample_specs.keys()):
    for objid in object_lst:
        p,r,j = greedy(sample,objid,"","worker_fraction")
        df_data.append([sample,objid,"worker fraction",p,r,j])
        print sample,objid,p,r,j
df = pd.DataFrame(df_data,columns=['sample','objid','algo','p','r','j'])
df.to_csv("greedy_result_worker_fraction.csv",index=None)
'''


df_data = []
#for sample in tqdm(sample_specs.keys()[::-1]):
import sys
idx = int(sys.argv[1])
sample = sample_specs.keys()[idx]
for objid in object_lst:
    for algo in ['basic','GT','isoGT','GTLSA','isoGTLSA']:
	p,r,j = greedy(sample,objid,algo)
	df_data.append([sample,objid,algo,p,r,j])
	print sample,objid,algo,p,r,j
df = pd.DataFrame(df_data,columns=['sample','objid','algo','p','r','j'])
df.to_csv("greedy_result_{}.csv".format(idx))	
'''

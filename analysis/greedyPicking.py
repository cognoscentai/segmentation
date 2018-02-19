import pickle as pkl
import json
from utils import PIXEL_EM_DIR, get_gt_mask, get_all_worker_tiles, get_mega_mask, \
    num_workers, faster_compute_prj,clusters
import os.path
import numpy as np
from collections import defaultdict
import tqdm
# from PixelEM import *


def process_all_worker_tiles(sample, objid, algo, cluster_id="", DEBUG=False):
    '''
    returns:
        1) num_votes[tid] = number of votes given to tid
        2) tile_area[tid] = area of tile tid
        3) tile_int_area[tid] = est_int_area for tid under given algo
            algo can be 'worker_fraction', 'ground_truth', 'GTLSA' etc.
    '''

    if cluster_id != "" and cluster_id != "-1":
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample, objid)

    if algo not in ['ground_truth', 'worker_fraction']:
        # for ground_truth, worker_fraction don't need to do anything, already computed by default
        log_probability_in_mask = pkl.load(open("{}{}_p_in_mask_ground_truth.pkl".format(outdir, algo)))
        log_probability_not_in_mask = pkl.load(open("{}{}_p_not_in_ground_truth.pkl".format(outdir, algo)))

    pixels_in_tile = get_all_worker_tiles(sample, objid, cluster_id)
    gt_mask = get_gt_mask(objid)
    worker_mega_mask = get_mega_mask(sample, objid,cluster_id)
    nworkers = num_workers(sample, objid, cluster_id)
    # mv_mask = get_MV_mask(sample, objid)
    if DEBUG:
        print 'Num tiles: ', len(pixels_in_tile)

    num_votes = defaultdict(int)
    tile_area = defaultdict(int)
    tile_int_area = defaultdict(int)

    pix_frequency = defaultdict(int)

    num_pixs_total = 0
    for tid in range(len(pixels_in_tile)):
        pixs = list(pixels_in_tile[tid])
        # num_votes[tid] = worker_mega_mask[next(iter(pixs))]
        num_votes[tid] = worker_mega_mask[pixs[0]]
        tile_area[tid] = len(pixs)
        if algo == 'ground_truth':
            for pix in pixs:
                tile_int_area[tid] += int(gt_mask[pix])
        elif algo == 'worker_fraction':
            tile_int_area[tid] = (float(num_votes[tid]) / float(nworkers)) * tile_area[tid]
        else:
            # for ground_truth, worker_fraction don't need to do anything, already computed by default
            pInT = np.exp(log_probability_in_mask[pixs[0]])  # all pixels in same tile should have the same pInT
            pNotInT = np.exp(log_probability_not_in_mask[pixs[0]])
            if pInT + pNotInT != 0:
                norm_pInT = pInT / (pNotInT+pInT)  # normalized pInT
            else:  # weird bug for object 18 isoGT case
                norm_pInT = 1.
            assert norm_pInT <= 1 and norm_pInT >= 0
            tile_int_area[tid] = norm_pInT * tile_area[tid]

    if DEBUG:
        print 'Max number of tiles any pixel belongs to: ', max(pix_frequency.values())
        print 'Height x width of gt_mask = {}, num pixs accounted for: {}'.format(
            len(gt_mask)*len(gt_mask[0]), num_pixs_total)

    return num_votes, tile_area, tile_int_area


def run_greedy_jaccard(tile_area, tile_int_area, DEBUG=False):
    '''
    input:
        tile_area[tid], tile_int_area[tid]
    output:
        1) sorted_order_tids = [tid1, tid2, ...]  # decreasing order of I/O
        2) tile_added = {tid: bool}  # tile included or not
        3) est_jacc_list = [estj1, estj2, ...]  # estimated jaccard after each tile
    '''

    GT_area = sum(tile_int_area.values())
    completely_contained_tid_list = [tid for tid in tile_area.keys() if tile_area[tid] == tile_int_area[tid]]
    tids_not_completely_contained = [tid for tid in tile_area.keys() if tile_area[tid] != tile_int_area[tid]]
    sorted_order_tids = completely_contained_tid_list + sorted(
        tids_not_completely_contained,
        key=(lambda x: (float((tile_int_area[x])) / float(tile_area[x] - tile_int_area[x]))),
        reverse=True
    )
    tile_added = defaultdict(bool)
    est_jacc_list = []

    curr_total_int_area = 0.0
    curr_total_out_area = 0.0
    curr_jacc = 0.0
    for tid in sorted_order_tids:
        IA = tile_int_area[tid]
        OA = tile_area[tid] - IA
        new_jacc_if_added = float(curr_total_int_area + IA) / float(curr_total_out_area + OA + GT_area)
        if DEBUG:
            print 'IA: {}, OA: {}'.format(IA, OA)
            print curr_jacc, new_jacc_if_added
        if new_jacc_if_added >= curr_jacc:
            # add tile if it improves jaccard
            tile_added[tid] = True
            curr_total_int_area += IA
            curr_total_out_area += OA
            curr_jacc = new_jacc_if_added
        est_jacc_list.append(curr_jacc)

    return sorted_order_tids, tile_added, est_jacc_list


def greedy(sample, objid, algo, cluster_id="", output="prj", rerun_existing=False):
    if cluster_id != "" and  cluster_id != "-1":
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample, objid)
    outfile  = '{}{}_greedy_prj.json'.format(outdir, algo)
    if (not rerun_existing) and os.path.exists(outfile):
        print outfile + " already exist, read from file"
        p, r, j = json.load(open(outfile))
        return p, r, j

    num_votes, tile_area, tile_int_area = process_all_worker_tiles(sample, objid, algo, cluster_id, DEBUG=False)
    sorted_order_tids, tile_added, est_jacc_list = run_greedy_jaccard(tile_area, tile_int_area, DEBUG=False)

    pixels_in_tile = get_all_worker_tiles(sample, objid, cluster_id)
    gt = get_gt_mask(objid)
    gt_est_mask = np.zeros_like(gt)
    for tid in sorted_order_tids:
        if tile_added[tid]:
            for pix in pixels_in_tile[tid]:
                gt_est_mask[pix] = 1

    if output == "mask":
        return gt_est_mask
    elif output == "prj":
        [p, r, j] = faster_compute_prj(gt_est_mask, gt)
        with open(outfile, 'w') as fp:
            fp.write(json.dumps([p, r, j]))
        if j <= 0.5:  # in the case where we are aggregating a semantic error cluster
            pkl.dump(gt_est_mask, open('{}{}_gt_est_mask_greedy.pkl'.format(outdir, algo), 'w'))
        return p, r, j


if __name__ == '__main__':
    object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
    # object_lst = [1]
    from sample_worker_seeds import sample_specs
    import time
    import sys
    import pandas as pd
    DEBUG = False 
    '''
    df_data = []
    for sample in sample_specs.keys():
        for objid in object_lst:
	    if DEBUG: start = time.time()
            p, r, j = greedy(sample, objid, "ground_truth")
	    if DEBUG:
		end =  time.time()
		print "Time Elapsed:",end-start
            df_data.append([sample, objid, "ground truth", p, r, j])
            print 'ground_truth', sample, objid, p, r, j
    df = pd.DataFrame(df_data, columns=['sample', 'objid', 'algo', 'p', 'r', 'j'])
    df.to_csv("greedy_result_ground_truth.csv", index=None)
    '''
    obj_clusters = clusters()
    df_data = []
    idx = int(sys.argv[1])
    sample = sample_specs.keys()[idx]
    #for sample in sample_specs.keys():
    if True:
        for objid in object_lst:
	    if str(objid) in obj_clusters[sample]:
                clusts = ["-1"] + [obj_clusters[sample][str(objid)]]
            else:
                clusts = ["-1"]
	    for clust in clusts: 
            	p, r, j = greedy(sample, objid, "worker_fraction",cluster_id=clust,rerun_existing=True)
            	df_data.append([sample, objid, "worker fraction", p, r, j])
            	print 'worker_fraction', sample, objid, clust, p, r, j
    df = pd.DataFrame(df_data, columns=['sample', 'objid', 'algo', 'p', 'r', 'j'])
    df.to_csv("greedy_result_worker_fraction.csv", index=None)
   
    # Takes about 2.5~3hrs to run
    obj_clusters = clusters()
    df_data = []
    idx = int(sys.argv[1])
    sample = sample_specs.keys()[idx]    
    for objid in object_lst:
        if str(objid) in obj_clusters[sample]:
            clusts = ["-1"] + [obj_clusters[sample][str(objid)]]
        else:
            clusts = ["-1"]
        for clust in clusts:
	    for algo in ["worker_fraction",'basic', 'GT', 'isoGT', 'GTLSA', 'isoGTLSA']:
            p, r, j = greedy(sample, objid, algo,cluster_id=clust,rerun_existing=False)
            df_data.append([sample, objid, algo, clust, p, r, j])
            print algo, sample, objid, clust, p, r, j
    df = pd.DataFrame(df_data, columns=['sample', 'objid', 'algo','clust', 'p', 'r', 'j'])
    df.to_csv("greedy_result_{}.csv".format(idx), index=None) 

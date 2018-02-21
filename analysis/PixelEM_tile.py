SHAPELY_OFF = True

import matplotlib
import numpy as np
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
# if SHAPELY_OFF:
#     from analysis_toolbox import *

from collections import defaultdict
import pickle
import json
import time
import os
from utils import get_gt_mask, get_worker_mask, get_mega_mask, workers_in_sample, get_MV_mask, \
    get_all_worker_mega_masks_for_sample, faster_compute_prj, compute_PRJ_MV, TFPNR, \
    get_all_worker_tiles, tile_and_mask_dir, get_voted_workers_mask, \
    PIXEL_EM_DIR, num_workers, tiles_to_mask, prj_tile_against_tile, tile_em_output_dir, \
    get_tile_to_area_map, get_tile_to_workers_map, get_worker_to_tiles_map, get_MV_tiles


def create_tile_area_map(sample, objid, clust_id='-1'):
    # creates tarea[tid] = area of tile dictionary
    tiles = get_all_worker_tiles(sample, objid, clust_id)
    tarea = dict()
    for tid, tile in enumerate(tiles):
        tarea[tid] = len(tile)
    outdir = tile_and_mask_dir(sample, objid, clust_id)
    with open('{}/tile_area.pkl'.format(outdir), 'w') as fp:
        fp.write(pickle.dumps(tarea))


def create_MV_tiles(sample, objid, clust_id='-1'):
    # creates set() of tiles in MV
    tiles = get_all_worker_tiles(sample, objid, clust_id)
    MV_mask = get_MV_mask(sample, objid, clust_id)
    MV_tiles = set()
    for tid, tile in enumerate(tiles):
        pix_in_tile = list(tile)[0]
        if MV_mask[pix_in_tile]:
            MV_tiles.add(tid)
    outdir = tile_and_mask_dir(sample, objid, clust_id)
    with open('{}/MV_tiles.pkl'.format(outdir), 'w') as fp:
        fp.write(pickle.dumps(MV_tiles))


def create_tile_to_worker_list_map_and_inverse(sample, objid, clust_id='-1', DEBUG=True):
    # creates tworkers[tid] = [list of workers voting yes for tid] dictionary
    # creates wtiles[wid] = [list of tiles voted yes by wid] dictionary
    tiles = get_all_worker_tiles(sample, objid, clust_id)
    voted_workers_mask = get_voted_workers_mask(sample, objid, clust_id)
    if DEBUG:
        mega_mask = get_mega_mask(sample, objid, clust_id)  # only for sanity check
    tworkers = defaultdict(list)
    wtiles = defaultdict(list)
    for tid, tile in enumerate(tiles):
        pix_in_tile = list(tile)[0]
        workers_on_tile = voted_workers_mask[pix_in_tile]
        if workers_on_tile == 0:
            # TODO: fix voted_workers_mask in PixelEM to have [] instead of 0 for pixs with no votes
            workers_on_tile = []
        if DEBUG:
            num_worker_on_tile = mega_mask[pix_in_tile]
            assert len(workers_on_tile) == num_worker_on_tile
        for wid in workers_on_tile:
            tworkers[tid].append(wid)
            wtiles[wid].append(tid)
    outdir = tile_and_mask_dir(sample, objid, clust_id)
    with open('{}/tile_to_workers.pkl'.format(outdir), 'w') as fp:
        fp.write(pickle.dumps(tworkers))
    with open('{}/worker_to_tiles.pkl'.format(outdir), 'w') as fp:
        fp.write(pickle.dumps(wtiles))


def sanity_checks(sample, objid, clust_id='-1'):
    # check that all the tile info is consistent with all the pixel info
    print 'Running pixel vs tile sanity checks for {}:{}:{}'.format(sample, objid, clust_id)

    # pixel info
    tiles = get_all_worker_tiles(sample, objid, clust_id)
    voted_workers_mask = get_voted_workers_mask(sample, objid, clust_id)
    mega_mask = get_mega_mask(sample, objid, clust_id)
    MV_mask = get_MV_mask(sample, objid, clust_id)

    # tile info
    tarea = get_tile_to_area_map(sample, objid, clust_id)
    tworkers = get_tile_to_workers_map(sample, objid, clust_id)
    wtiles = get_worker_to_tiles_map(sample, objid, clust_id)
    mv_tiles = get_MV_tiles(sample, objid, clust_id)

    for tid, tile in enumerate(tiles):
        assert len(tile) == tarea[tid]
        pix_in_tile = list(tile)[0]
        voted_workers = voted_workers_mask[pix_in_tile]
        if MV_mask[pix_in_tile]:
            assert tid in mv_tiles
        else:
            assert tid not in mv_tiles
        if voted_workers == 0:
            # TODO: fix voted_workers_mask in PixelEM to have [] instead of 0 for pixs with no votes
            voted_workers = []
        assert set(tworkers[tid]) == set(voted_workers)
        assert len(tworkers[tid]) == mega_mask[pix_in_tile]
        for wid in voted_workers:
            assert tid in wtiles[wid]


def GTLSAworker_prob_correct(gt_tiles, curr_worker_tiles, tarea, tworkers, Nworkers, exclude_isovote=False):
    '''
    gt_tiles = set() of tiles in ground truth`
    curr_worker_tiles = set() of tiles voted for by worker whose prob is being calculated
    tarea = {tid: area of tile}
    tworkers: {tid: [workers voted 1 for tile]}
    '''

    # TODO: why was the following commented section in the code?
    # for t in tiles:
    #     numerator = 0
    #     for tidx in t:
    #         numerator += gt_mask[tidx]
    #     if len(tidx) != 0:
    #         gt_percentage = numerator / float(len(t))
    #     if gt_percentage > 0.6:
    #         gt_tiles.append(t)
    #     else:
    #         ngt_tiles.append(t)

    ngt_tiles = set(tarea.keys()) - gt_tiles
    gt_areas = []
    for t in gt_tiles:
        gt_areas.append(tarea[t])
    ngt_areas = []
    for t in ngt_tiles:
        ngt_areas.append(tarea[t])
    if gt_areas != [] and ngt_areas != []:
        area_thresh_gt = (min(gt_areas)+max(gt_areas))/2.
        area_thresh_ngt = (min(ngt_areas)+max(ngt_areas))/2.
    else:
        # TODO: investigate further
        print "Case where one of gt or ngt area list is empty, probably due to low number of datapoints (from one of the smaller , possibly mistaken, clusters)"
        gt_areas.extend(ngt_areas)
        area_thresh_gt = np.mean(gt_areas)
        area_thresh_ngt = np.mean(gt_areas)

    large_gt_Ncorrect, large_gt_total, large_ngt_Ncorrect, large_ngt_total = 0, 0, 0, 0
    small_gt_Ncorrect, small_gt_total, small_ngt_Ncorrect, small_ngt_total = 0, 0, 0, 0

    for tid in tarea:
        gt = (tid in gt_tiles)
        w = (tid in curr_worker_tiles)
        m = len(tworkers[tid])  # num workers voted for tile
        a = tarea[tid]
        if exclude_isovote:
            not_agreement = False
            if m != 0 and m != Nworkers:
                not_agreement = True
        else:
            not_agreement = True
        if not_agreement:
            if gt is True and (a >= area_thresh_gt):
                large_gt_total += a
                if w is True:
                    large_gt_Ncorrect += a
            if gt is False and (a >= area_thresh_ngt):
                large_ngt_total += a
                if w is False:
                    large_ngt_Ncorrect += a
            if gt is True and (a < area_thresh_gt):
                small_gt_total += a
                if w is True:
                    small_gt_Ncorrect += a
            if gt is False and (a < area_thresh_ngt):
                small_ngt_total += a
                if w is False:
                    small_ngt_Ncorrect += a
    qp1 = float(large_gt_Ncorrect)/float(large_gt_total) if large_gt_total != 0 else 0.6
    qn1 = float(large_ngt_Ncorrect)/float(large_ngt_total) if large_ngt_total != 0 else 0.6
    qp2 = float(small_gt_Ncorrect)/float(small_gt_total) if small_gt_total != 0 else 0.6
    qn2 = float(small_ngt_Ncorrect)/float(small_ngt_total) if small_ngt_total != 0 else 0.6
    # print "qp1, qn1, qp2, qn2:", qp1, qn1, qp2, qn2
    return qp1, qn1, qp2, qn2, area_thresh_gt, area_thresh_ngt


def GTLSAmask_log_probabilities(wtiles, qp1, qn1, qp2, qn2, tarea, area_thresh_gt, area_thresh_ngt):
    '''
    input wtiles: {wid: [list of tiles voted yes for]}
    output log_probability_in[tid] = log_prob  of each pixel in tile (all pixels identical)
    output log_probability_not_in[tid] = log_prob  of each pixel not in tile (all pixels identical)
    '''
    worker_ids = qp1.keys()
    log_probability_in = defaultdict(float)
    log_probability_not_in = defaultdict(float)

    for tid in tarea:
        for wid in worker_ids:
            qp1i = qp1[wid]
            qn1i = qn1[wid]
            qp2i = qp2[wid]
            qn2i = qn2[wid]
            ljk = (tid in wtiles[wid])
            large_gt = (tarea[tid] >= area_thresh_gt)  # would the tile qualify as large if in GT
            large_ngt = (tarea[tid] >= area_thresh_ngt)  # would the tile qualify as large if not in GT
            if ljk is True:
                if large_gt:
                    # update pInT masks
                    log_probability_in[tid] += np.log(qp1i)
                else:
                    log_probability_in[tid] += np.log(qp2i)
                if large_ngt:
                    # update pNotInT masks
                    log_probability_not_in[tid] += np.log(1.0 - qn1i)
                else:
                    log_probability_not_in[tid] += np.log(1.0 - qn2i)
            else:
                if large_gt:
                    # update pInT masks
                    log_probability_in[tid] += np.log(1.0 - qp1i)
                else:
                    log_probability_in[tid] += np.log(1.0 - qp2i)
                if large_ngt:
                    # update pNotInT masks
                    log_probability_not_in[tid] += np.log(qn1i)
                else:
                    log_probability_not_in[tid] += np.log(qn2i)
    return log_probability_in, log_probability_not_in


def estimate_gt_from(log_probability_in, log_probability_not_in, thresh=0):
    '''
    input log_probability_in[tid] = log_prob  of each pixel in tile (all pixels identical)
    input log_probability_not_in[tid] = log_prob  of each pixel not in tile (all pixels identical)
    output: set() of tiles with log_prob_in >= thresh + log_prob_not_in
    '''
    gt_est_tiles = set()
    for tid in log_probability_in.keys():
        if log_probability_in[tid] >= thresh + log_probability_not_in[tid]:
            gt_est_tiles.add(tid)

    return gt_est_tiles


# def compute_A_thres(condition, tarea):
#     '''
#     intput condition lambda that takes tid as input
#     # Compute the new area threshold based on the median area of high confidence pixels
#     '''
#     high_confidence_pixel_area = []
#     for tid in tarea.keys():
#         if condition(tid):
#             high_confidence_pixel_area.append(tarea[tid])

#     # TODO: why was the following commented section in the code?
#     # passing_xs, passing_ys = np.where(condition)  # pInT >= pNotInT)
#     # for i in range(len(passing_xs)):
#     #     high_confidence_pixel_area.append(area_mask[passing_xs[i]][passing_ys[i]])

#     A_thres = np.median(high_confidence_pixel_area)
#     return A_thres


def do_GTLSA_EM_for(
    sample_name, objid, cluster_id="", rerun_existing=False, exclude_isovote=False,
    dump_output_at_every_iter=False, compute_PR_every_iter=False, PLOT=False, DEBUG=False
):
    if exclude_isovote:
        mode = 'iso'
    else:
        mode = ''
    if DEBUG:
        print "Doing GTLSA mode=", mode
        start = time.time()

    outdir = tile_em_output_dir(sample_name, objid, cluster_id)

    print "Doing GTLSA mode=", mode
    if not rerun_existing:
        if os.path.isfile('{}{}GTLSA_EM_prj_best_thresh.json'.format(outdir, mode)):
            print "Already ran GTLSA, Skipped"
            return

    tarea = get_tile_to_area_map(sample_name, objid, cluster_id)
    tworkers = get_tile_to_workers_map(sample_name, objid, cluster_id)
    wtiles = get_worker_to_tiles_map(sample_name, objid, cluster_id)
    Nworkers = num_workers(sample_name, objid, cluster_id)

    # only used to convert tiles to pixels when prj needs to be computed
    tiles = get_all_worker_tiles(sample_name, objid, cluster_id)
    gt_mask = get_gt_mask(objid)

    # initialize MV tiles
    MV_tiles = get_MV_tiles(sample_name, objid, cluster_id)
    gt_est_tiles = MV_tiles.copy()
    # In the first step we use 50% MV for initializing T*, A thres is therefore the median area pixel based on votes and noVotes
    prev_gt_est = gt_est_tiles.copy()
    jaccard_against_prev_gt_est = 0
    it = 0
    # for it in range(num_iterations):
    max_iter = 6
    while (jaccard_against_prev_gt_est < 0.999 or it <= 1):
        if (it >= max_iter):
            break
        if DEBUG:
            print "iteration:", it
        it += 1
        qp1 = dict()
        qn1 = dict()
        qp2 = dict()
        qn2 = dict()
        if DEBUG:
            t0 = time.time()
        for wid in wtiles.keys():
            qp1[wid], qn1[wid], qp2[wid], qn2[wid], area_thresh_gt, area_thresh_ngt = GTLSAworker_prob_correct(
                gt_est_tiles, wtiles[wid], tarea, tworkers, Nworkers, exclude_isovote=exclude_isovote)
        if DEBUG:
            t1 = time.time()
            print "Time for worker prob calculation:", t1-t0
        # Compute pInMask and pNotInMask
        log_probability_in, log_probability_not_in = GTLSAmask_log_probabilities(
            wtiles, qp1, qn1, qp2, qn2, tarea, area_thresh_gt, area_thresh_ngt)
        if DEBUG:
            t2 = time.time()
            print "Time for mask log prob calculation:", t2-t1
        # gt_est_mask = estimate_gt_from(log_probability_in_mask, log_probability_not_in_mask,thresh=thresh)
        p, r, j, thresh, gt_est_tiles = binarySearchDeriveBestThresh(
            sample_name, objid, cluster_id, log_probability_in, log_probability_not_in, MV_tiles,
            exclude_isovote=exclude_isovote, rerun_existing=rerun_existing, DEBUG=DEBUG)
        # Compute PR mask based on the EM estimate mask from every iteration
        if compute_PR_every_iter:
            gt_est_mask = tiles_to_mask(gt_est_tiles, tiles, gt_mask)
            [p, r, j] = faster_compute_prj(gt_est_mask, gt_mask)
            with open('{}{}GTLSA_EM_prj_iter{}_thresh{}.json'.format(outdir, mode, it, thresh), 'w') as fp:
                fp.write(json.dumps([p, r, j]))

            if DEBUG:
                print qp1, qn1, qp2, qn2
                print "-->"+str([p, r, j])

        # compute jaccard between previous and current gt estimation mask
        [p_against_prev, r_against_prev, jaccard_against_prev_gt_est] = prj_tile_against_tile(
            gt_est_tiles, prev_gt_est, tarea)
        if DEBUG:
            print "jaccard_against_prev_gt_est:", jaccard_against_prev_gt_est
        prev_gt_est = gt_est_tiles.copy()

    gt_est_mask = tiles_to_mask(gt_est_tiles, tiles, gt_mask)
    [p, r, j] = faster_compute_prj(gt_est_mask, gt_mask)
    print 'Final prj:', [p, r, j]
    with open('{}{}GTLSA_EM_prj_best_thresh.json'.format(outdir, mode), 'w') as fp:
        fp.write(json.dumps([p, r, j]))
    pickle.dump(gt_est_tiles, open('{}{}GTLSA_gt_est_tiles_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(log_probability_in, open('{}{}GTLSA_p_in_tiles_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(log_probability_not_in, open('{}{}GTLSA_p_not_in_tiles_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(qp1, open('{}{}GTLSA_qp1_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(qn1, open('{}{}GTLSA_qn1_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(qp2, open('{}{}GTLSA_qp2_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(qn2, open('{}{}GTLSA_qn2_best_thresh.pkl'.format(outdir, mode), 'w'))
    if PLOT:
        plt.figure()
        plt.imshow(gt_est_mask, interpolation="none")  # ,cmap="rainbow")
        plt.colorbar()
        plt.savefig('{}{}GTLSA_EM_mask_thresh{}.png'.format(outdir, mode, thresh))
    if DEBUG:
        end = time.time()
        print "Time:{}".format(end-start)
        return end-start


def estimate_gt_compute_PRJ_against_MV(
    sample, objid, cluster_id, log_probability_in, log_probability_not_in,
    tiles_to_search_against, thresh, exclude_isovote=False
):
    gt_est_tiles = estimate_gt_from(log_probability_in, log_probability_not_in, thresh=thresh)
    tarea = get_tile_to_area_map(sample, objid, cluster_id)
    if exclude_isovote:
        nWorkers = num_workers(sample, objid, cluster_id)
        tworkers = get_tile_to_workers_map(sample, objid, cluster_id)
        for tid in tworkers:
            num_votes = len(tworkers[tid])
            if num_votes == nWorkers:
                gt_est_tiles.add(tid)
            elif num_votes == 0 and tid in gt_est_tiles:
                gt_est_tiles.remove(tid)

    # PRJ values against MV
    [p, r, j] = prj_tile_against_tile(gt_est_tiles, tiles_to_search_against, tarea)
    return [p, r, j], gt_est_tiles


def binarySearchDeriveBestThresh(
    sample_name, objid, cluster_id, log_probability_in, log_probability_not_in, tiles_to_search_against,
    exclude_isovote=False, rerun_existing=False, plot_crossover=False, DEBUG=False
):
    # binary search for p == r point
    # p and r computed against tiles_to_search_against
    # if arbitrary reference (for eg., gt) that does not respect tile boundaries, need to change to mask
    thresh_min = -200.0
    thresh_max = 200.0
    thresh = (thresh_min+thresh_max)/2.0
    p, r = 0, -1
    iterations = 0
    epsilon = 0.125
    outdir = tile_em_output_dir(sample_name, objid, cluster_id)
    if plot_crossover:
        track = []  # for plotting

    if DEBUG:
        # used to compute actual prj
        gt_mask = get_gt_mask(objid)
        tiles = get_all_worker_tiles(sample_name, objid, cluster_id)

    while (iterations <= 100 or p == -1):  # continue iterations below max iterations or if p=-1
        if (p == r) or (thresh_min + epsilon >= thresh_max):
            # stop if p==r or if epsilon (range in x) gets below a certain threshold
            break
        [p, r, j], gt_est_tiles = estimate_gt_compute_PRJ_against_MV(
            sample_name, objid, cluster_id, log_probability_in, log_probability_not_in, tiles_to_search_against, thresh, exclude_isovote=exclude_isovote)
        if p > r:
            thresh_max = thresh
        else:
            thresh_min = thresh

        if DEBUG:
            print "----Trying threshold:", thresh, "-----"
            print p, r, j, thresh_max, thresh_min
            gt_est_mask = tiles_to_mask(gt_est_tiles, tiles, gt_mask)
            gt_p, gt_r, gt_j = faster_compute_prj(gt_est_mask, gt_mask)
            print "actual prj against GT", gt_p, gt_r, gt_j
        if plot_crossover:
            # track.append([thresh, gt_p, gt_r, gt_j])
            track.append([thresh, p, r, j])
            # plt.figure()
            # plt.title("Iter #"+str(iterations))
            # plt.imshow(gt_est_mask)
            # plt.colorbar()

        thresh = (thresh_min+thresh_max)/2.0
        iterations += 1

    if DEBUG and plot_crossover:
        # TODO: currently overwritten by different algos and iterations
        track = np.array(track)
        plt.figure()
        plt.title('prj_crossover')
        idx = np.argsort(track[:, 0])
        ths = track[:, 0][idx]
        ps = track[:, 1][idx]
        rs = track[:, 2][idx]
        plt.plot(ths, ps, label="p")
        plt.plot(ths, rs, label="r")
        plt.plot(track[-1][0], track[-1][1], '^')
        # print track
        # print track[-1][0], track[-1][1]
        plt.savefig('{}tmp_crossover.png'.format(outdir))
        plt.close()

    return p, r, j, thresh, gt_est_tiles

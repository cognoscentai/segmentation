DEBUG = False
SHAPELY_OFF = True

import matplotlib
import numpy as np
from config import *
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import sys
sys.path.append("..")
if SHAPELY_OFF:
    from analysis_toolbox import *

import pickle
import json
import time
import os
from sample_worker_seeds import sample_specs
from utils import get_gt_mask, get_worker_mask, get_mega_mask, workers_in_sample, get_MV_mask, \
    get_all_worker_mega_masks_for_sample, faster_compute_prj, compute_PRJ_MV, TFPNR


def create_all_gt_and_worker_masks(objid, PLOT=False, PRINT=False, EXCLUDE_BBG=True):
    img_info, object_tbl, bb_info, hit_info = load_info()
    # Ji_tbl (bb_info) is the set of all workers that annotated object i
    bb_objects = bb_info[bb_info["object_id"] == objid]
    # Create a masked image for the object
    # where each of the worker BB is considered a mask and overlaid on top of each other
    img_name = img_info[img_info.id == int(object_tbl[object_tbl.id == objid]["image_id"])]["filename"].iloc[0]
    fname = ORIGINAL_IMG_DIR + img_name + ".png"
    img = mpimg.imread(fname)
    width, height = get_size(fname)

    outdir = '{}obj{}/'.format(PIXEL_EM_DIR, objid)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    obj_x_locs = [process_raw_locs([x, y])[0] for x, y in zip(bb_objects["x_locs"], bb_objects["y_locs"])]
    obj_y_locs = [process_raw_locs([x, y])[1] for x, y in zip(bb_objects["x_locs"], bb_objects["y_locs"])]
    worker_ids = list(bb_objects["worker_id"])

    # for x_locs, y_locs in zip(obj_x_locs, obj_y_locs):
    for i in range(len(obj_x_locs)):
        x_locs = obj_x_locs[i]
        y_locs = obj_y_locs[i]
        wid = worker_ids[i]

        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(zip(x_locs, y_locs), outline=1, fill=1)
        mask = np.array(img) == 1
        # plt.imshow(mask)
        with open('{}mask{}.pkl'.format(outdir, wid), 'w') as fp:
            fp.write(pickle.dumps(mask))

        if PLOT:
            plt.figure()
            plt.imshow(mask, interpolation="none")  # ,cmap="rainbow")
            plt.colorbar()
            plt.show()

    my_BB = pd.read_csv('{}/my_ground_truth.csv'.format(DATA_DIR))
    bb_match = my_BB[my_BB.object_id == objid]
    x_locs, y_locs = process_raw_locs([bb_match['x_locs'].iloc[0], bb_match['y_locs'].iloc[0]])
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(zip(x_locs, y_locs), outline=1, fill=1)
    mask = np.array(img) == 1
    with open('{}gt.pkl'.format(outdir), 'w') as fp:
        fp.write(pickle.dumps(mask))


def create_mega_mask(objid, cluster_id="", PLOT=False, sample_name='5workers_rand0', PRINT=False, EXCLUDE_BBG=True):
    img_info, object_tbl, bb_info, hit_info = load_info()
    # bb_info is the set of all workers that annotated object i
    bb_objects = bb_info[bb_info["object_id"] == objid]
    if cluster_id == "" or cluster_id == "-1" or cluster_id == -1:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    else:
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    if EXCLUDE_BBG:
        bb_objects = bb_objects[bb_objects.worker_id != 3]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    sampleNworkers = sample_specs[sample_name][0]
    if sampleNworkers > 0 and sampleNworkers < len(bb_objects):
        bb_objects = bb_objects.sample(n=sample_specs[sample_name][0], random_state=sample_specs[sample_name][1])
    worker_ids_in_sample = list(bb_objects["worker_id"])
    if cluster_id == "" or cluster_id == "-1" or cluster_id == -1:
        # For the no cluster cases, sampling data from table
        worker_ids = worker_ids_in_sample
    else:
        clust_df = pd.read_csv("spectral_clustering_all_hard_obj.csv")
        worker_ids_in_cluster = list(clust_df[(clust_df["objid"] == objid) & (clust_df["cluster"] == cluster_id)]["wid"])
        worker_ids_in_cluster = set(worker_ids_in_cluster)
        worker_ids_in_sample = set(worker_ids_in_sample)
        worker_ids = list(worker_ids_in_cluster.intersection(worker_ids_in_sample))
    with open('{}worker_ids.json'.format(outdir), 'w') as fp:
        fp.write(json.dumps(worker_ids))
    # Megamask creation
    img_name = img_info[img_info.id == int(object_tbl[object_tbl.id == objid]["image_id"])]["filename"].iloc[0]
    fname = ORIGINAL_IMG_DIR + img_name + ".png"
    width, height = get_size(fname)
    mega_mask = np.zeros((height, width))
    voted_workers_mask = np.zeros((height, width), dtype=object)

    for wid in worker_ids:
        mask = get_worker_mask(objid, wid)
        mega_mask += mask
        # Voted Mask
        voted_coord = np.where(mask == True)
        for x, y in zip(voted_coord[0], voted_coord[1]):
            if voted_workers_mask[x, y] == 0:
                voted_workers_mask[x, y] = [wid]
            else:
                voted_workers_mask[x, y].append(wid)
    if PLOT:
        # Visualize mega_mask
        plt.figure()
        plt.imshow(mega_mask, interpolation="none")  # ,cmap="rainbow")
        # plt.imshow(mask, interpolation="none")  # ,cmap="rainbow")
        plt.colorbar()
        #plt.savefig('{}mega_mask.png'.format(outdir))
    with open('{}mega_mask.pkl'.format(outdir), 'w') as fp:
        fp.write(pickle.dumps(mega_mask))
    with open('{}voted_workers_mask.pkl'.format(outdir), 'w') as fp:
        fp.write(pickle.dumps(voted_workers_mask))


def worker_prob_correct(mega_mask, w_mask, gt_mask, Nworkers, exclude_isovote=False):
    if exclude_isovote:
        # either fully voted  by all workers or not voted at all
        num_agreement = len(np.where((mega_mask == 0 and gt_mask == 0) | (mega_mask == Nworkers and gt_mask == 1))[0])
    else:
        num_agreement = 0
    # print "Num agreement:", num_agreement
    qj = float(len(np.where(w_mask == gt_mask)[0]) - num_agreement) / (len(gt_mask[0]) * len(gt_mask) - num_agreement)
    return qj


def aw_worker_prob_correct(mega_mask, w_mask, gt_mask, area_lst, Nworkers, exclude_isovote=False):
    # area weighted model for worker probability.
    if exclude_isovote:
        num_agreement = len(np.where((mega_mask == 0) | (mega_mask == Nworkers))[0])  # no votes
    else:
        num_agreement = 0
    numerator = 0
    denominator = sum(area_lst)
    for i in range(np.shape(gt_mask)[0]):
        for j in range(np.shape(gt_mask)[1]):
            gt = gt_mask[i][j]
            w = w_mask[i][j]
            m = mega_mask[i][j]
            if exclude_isovote:
                not_agreement = False
            else:
                not_agreement = True
            if m != 0 and m != Nworkers:
                not_agreement = True

            if not_agreement:
                if ((gt == 1) and (w == 1)) or (gt == 0 and w == 0):  # correct
                    numerator += 1
    qj = float(numerator) / (denominator - num_agreement)
    return qj


def GTworker_prob_correct(mega_mask, w_mask, gt_mask, Nworkers, exclude_isovote=False):
    # Testing
    gt_Ncorrect = 0
    gt_total = 0
    ngt_Ncorrect = 0
    ngt_total = 0
    for i in range(np.shape(gt_mask)[0]):
        for j in range(np.shape(gt_mask)[1]):
            gt = gt_mask[i][j]
            w = w_mask[i][j]
            m = mega_mask[i][j]
            if exclude_isovote:
                not_agreement = False
            else:
                not_agreement = True
            if m != 0 and m != Nworkers:
                not_agreement = True
            if not_agreement:
                if gt == 1:
                    gt_total += 1
                    if w == 1:
                        gt_Ncorrect += 1
                else:
                    ngt_total += 1
                    if w == 0:
                        ngt_Ncorrect += 1
    qp = float(gt_Ncorrect) / float(gt_total) if gt_total != 0 else 0.6
    qn = float(ngt_Ncorrect) / float(ngt_total) if ngt_total != 0 else 0.6
    return qp, qn


def GTLSAworker_prob_correct(mega_mask, w_mask, gt_mask, Nworkers, area_mask, tiles, exclude_isovote=False):
    gt_tiles = []
    ngt_tiles = []
    for t in tiles:
        numerator = 0
        for tidx in t:
            numerator += gt_mask[tidx]
        if len(tidx) != 0:
            gt_percentage = numerator / float(len(t))
        if gt_percentage > 0.6:
            gt_tiles.append(t)
        else:
            ngt_tiles.append(t)
    # tarea_lst = np.array(tarea_lst)
    # area_thresh_gt =  np.median(tarea_lst[gt_tiles])
    # area_thresh_ngt = np.median(tarea_lst[ngt_tiles])
    gt_areas = []
    for t in gt_tiles:
        gt_areas.append(area_mask[list(t)[0]])
    ngt_areas = []
    for t in ngt_tiles:
        ngt_areas.append(area_mask[list(t)[0]])
    if gt_areas != [] and ngt_areas != []:
        area_thresh_gt = (min(gt_areas)+max(gt_areas))/2.
        area_thresh_ngt = (min(ngt_areas)+max(ngt_areas))/2.
    else:
        print "Case where one of gt or ngt area list is empty, probably due to low number of datapoints (from one of the smaller , possibly mistaken, clusters)"
        gt_areas.extend(ngt_areas)
        area_thresh_gt = np.mean(gt_areas)
        area_thresh_ngt = np.mean(gt_areas)
        # print gt_areas
        # print area_thresh_gt,area_thresh_ngt
    # print "inside gt split: ", len(np.where(gt_areas<area_thresh_gt)[0]), len(np.where(gt_areas>=area_thresh_gt)[0])
    # gt_tiles= np.array(gt_tiles)
    # print "large tile idx:",np.where(gt_areas>=area_thresh_gt)[0]
    # print gt_tiles[np.where(gt_areas>=area_thresh_gt)[0]]
    # print "inside ngt split: ",len(np.where(ngt_areas<area_thresh_ngt)[0]),len(np.where(ngt_areas>=area_thresh_ngt)[0])
    # print min(gt_areas),max(gt_areas), area_thresh_gt,len(gt_areas)
    # print min(ngt_areas),max(ngt_areas),area_thresh_ngt,len(ngt_areas)

    large_gt_Ncorrect, large_gt_total, large_ngt_Ncorrect, large_ngt_total = 0, 0, 0, 0
    small_gt_Ncorrect, small_gt_total, small_ngt_Ncorrect, small_ngt_total = 0, 0, 0, 0
    # print np.shape(gt_mask)
    # print gt_mask
    for i in range(np.shape(gt_mask)[0]):
        for j in range(np.shape(gt_mask)[1]):
            gt = gt_mask[i][j]
            w = w_mask[i][j]
            m = mega_mask[i][j]
            a = area_mask[i][j]
            if exclude_isovote:
                not_agreement = False
            else:
                not_agreement = True
            if m != 0 and m != Nworkers:
                not_agreement = True
            if not_agreement:
                if (gt == 1) and (a >= area_thresh_gt):
                    large_gt_total += 1
                    if w == 1:
                        large_gt_Ncorrect += 1
                if (gt == 0) and (a >= area_thresh_ngt):
                    large_ngt_total += 1
                    if w == 0:
                        large_ngt_Ncorrect += 1
                if (gt == 1) and (a < area_thresh_gt):
                    small_gt_total += 1
                    if w == 1:
                        small_gt_Ncorrect += 1
                if (gt == 0) and (a < area_thresh_ngt):
                    small_ngt_total += 1
                    if w == 0:
                        small_ngt_Ncorrect += 1
    qp1 = float(large_gt_Ncorrect)/float(large_gt_total) if large_gt_total != 0 else 0.6
    qn1 = float(large_ngt_Ncorrect)/float(large_ngt_total) if large_ngt_total != 0 else 0.6
    qp2 = float(small_gt_Ncorrect)/float(small_gt_total) if small_gt_total != 0 else 0.6
    qn2 = float(small_ngt_Ncorrect)/float(small_ngt_total) if small_ngt_total != 0 else 0.6
    # print "qp1, qn1, qp2, qn2:", qp1, qn1, qp2, qn2
    return qp1, qn1, qp2, qn2, area_thresh_gt, area_thresh_ngt


def mask_log_probabilities(worker_masks, worker_qualities):
    worker_ids = worker_qualities.keys()
    log_probability_in_mask = np.zeros((
        len(worker_masks[worker_ids[0]]), len(worker_masks[worker_ids[0]][0])
    ))
    log_probability_not_in_mask = np.zeros((
        len(worker_masks[worker_ids[0]]), len(worker_masks[worker_ids[0]][0])
    ))

    for i in range(len(worker_masks[worker_ids[0]])):
        for j in range(len(worker_masks[worker_ids[0]][0])):
            for wid in worker_ids:
                log_probability_in_mask[i][j] += np.log(
                    worker_qualities[wid] if worker_masks[wid][i][j] == 1
                    else (1.0 - worker_qualities[wid])
                )
                log_probability_not_in_mask[i][j] += np.log(
                    (1.0 - worker_qualities[wid]) if worker_masks[wid][i][j] == 1
                    else worker_qualities[wid]
                )
    return log_probability_in_mask, log_probability_not_in_mask


def GTLSAmask_log_probabilities(worker_masks, qp1, qn1, qp2, qn2, area_mask, area_thresh_gt, area_thresh_ngt):
    worker_ids = qp1.keys()
    log_probability_in_mask = np.zeros((
        len(worker_masks[worker_ids[0]]), len(worker_masks[worker_ids[0]][0])
    ))
    log_probability_not_in_mask = np.zeros((
        len(worker_masks[worker_ids[0]]), len(worker_masks[worker_ids[0]][0])
    ))

    for i in range(len(worker_masks[worker_ids[0]])):
        for j in range(len(worker_masks[worker_ids[0]][0])):
            for wid in worker_ids:
                qp1i = qp1[wid]
                qn1i = qn1[wid]
                qp2i = qp2[wid]
                qn2i = qn2[wid]
                ljk = worker_masks[wid][i][j]
                large_gt = (area_mask[i][j] >= area_thresh_gt)  # would the tile qualify as large if in GT
                large_ngt = (area_mask[i][j] >= area_thresh_ngt)  # would the tile qualify as large if not in GT
                if ljk == 1:
                    if large_gt:
                        # update pInT masks
                        log_probability_in_mask[i][j] += np.log(qp1i)
                    else:
                        log_probability_in_mask[i][j] += np.log(qp2i)
                    if large_ngt:
                        # update pNotInT masks
                        log_probability_not_in_mask[i][j] += np.log(1.0 - qn1i)
                    else:
                        log_probability_not_in_mask[i][j] += np.log(1.0 - qn2i)
                else:
                    if large_gt:
                        # update pInT masks
                        log_probability_in_mask[i][j] += np.log(1.0 - qp1i)
                    else:
                        log_probability_in_mask[i][j] += np.log(1.0 - qp2i)
                    if large_ngt:
                        # update pNotInT masks
                        log_probability_not_in_mask[i][j] += np.log(qn1i)
                    else:
                        log_probability_not_in_mask[i][j] += np.log(qn2i)
    return log_probability_in_mask, log_probability_not_in_mask


def GTmask_log_probabilities(worker_masks, qp, qn):
    worker_ids = qp.keys()
    log_probability_in_mask = np.zeros((
        len(worker_masks[worker_ids[0]]), len(worker_masks[worker_ids[0]][0])
    ))
    log_probability_not_in_mask = np.zeros((
        len(worker_masks[worker_ids[0]]), len(worker_masks[worker_ids[0]][0])
    ))

    for i in range(len(worker_masks[worker_ids[0]])):
        for j in range(len(worker_masks[worker_ids[0]][0])):
            for wid in worker_ids:
                qpi = qp[wid]
                qni = qn[wid]
                ljk = worker_masks[wid][i][j]
                # tjkInT = gt_mask[i][j]
                if ljk == 1:
                    log_probability_in_mask[i][j] += np.log(qpi)
                    log_probability_not_in_mask[i][j] += np.log(1-qni)
                else:
                    log_probability_not_in_mask[i][j] += np.log(qni)
                    log_probability_in_mask[i][j] += np.log(1-qpi)
    return log_probability_in_mask, log_probability_not_in_mask


def estimate_gt_from(log_probability_in_mask, log_probability_not_in_mask, thresh=0):
    gt_est_mask = np.zeros((len(log_probability_in_mask), len(log_probability_in_mask[0])))

    passing_xs, passing_ys = np.where(log_probability_in_mask >= thresh + log_probability_not_in_mask)
    for i in range(len(passing_xs)):
        gt_est_mask[passing_xs[i]][passing_ys[i]] = 1

    return gt_est_mask


def compute_A_thres(condition, area_mask):
    # Compute the new area threshold based on the meidan area of high confidence  pixels
    high_confidence_pixel_area = []
    #print np.where(pInT >= pNotInT)
    passing_xs, passing_ys = np.where(condition)  # pInT >= pNotInT)
    #print passing_xs,passing_ys
    for i in range(len(passing_xs)):
        high_confidence_pixel_area.append(area_mask[passing_xs[i]][passing_ys[i]])
    A_thres = np.median(high_confidence_pixel_area)
    return A_thres


def tiles2AreaMask(tiles, mega_mask):
    tarea = [len(t) for t in tiles]
    mask = np.zeros_like(mega_mask)
    for tidx in range(len(tiles)):
        for i in list(tiles[tidx]):
            mask[i] = tarea[tidx]
    return mask


def do_GTLSA_EM_for(sample_name, objid, cluster_id="", rerun_existing=False, exclude_isovote=False, dump_output_at_every_iter=False, compute_PR_every_iter=False, PLOT=False):
    if exclude_isovote:
        mode = 'iso'
    else:
        mode = ''
    if DEBUG:
        print "Doing GTLSA mode=", mode
        start = time.time()
    if cluster_id != "" and cluster_id != -1:
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)

    print "Doing GTLSA mode=", mode
    if not rerun_existing:
        if os.path.isfile('{}{}GTLSA_EM_prj_best_thresh.json'.format(outdir, mode)):
            print "Already ran GTLSA, Skipped"
            return
    # initialize MV mask
    MV = get_MV_mask(sample_name, objid, cluster_id)
    gt_est_mask = MV
    # In the first step we use 50% MV for initializing T*, A thres is therefore the median area pixel based on votes and noVotes
    mega_mask = get_mega_mask(sample_name, objid, cluster_id)
    tiles = pickle.load(open("{}tiles.pkl".format(outdir)))
    worker_masks = get_all_worker_mega_masks_for_sample(sample_name, objid, cluster_id)
    Nworkers = len(worker_masks)
    area_mask = tiles2AreaMask(tiles, mega_mask)
    prev_gt_est = gt_est_mask
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
        for wid in worker_masks.keys():
            qp1[wid], qn1[wid], qp2[wid], qn2[wid], area_thresh_gt, area_thresh_ngt = GTLSAworker_prob_correct(mega_mask, worker_masks[wid], gt_est_mask, Nworkers, area_mask, tiles, exclude_isovote=exclude_isovote)
        if DEBUG:
            t1 = time.time()
            print "Time for worker prob calculation:", t1-t0
        # Compute pInMask and pNotInMask
        log_probability_in_mask, log_probability_not_in_mask = GTLSAmask_log_probabilities(worker_masks, qp1, qn1, qp2, qn2, area_mask, area_thresh_gt, area_thresh_ngt)
        if DEBUG:
            t2 = time.time()
            print "Time for mask log prob calculation:", t2-t1
        # gt_est_mask = estimate_gt_from(log_probability_in_mask, log_probability_not_in_mask,thresh=thresh)
        p, r, j, thresh, gt_est_mask = binarySearchDeriveBestThresh(sample_name, objid, cluster_id, log_probability_in_mask, log_probability_not_in_mask, MV, exclude_isovote=exclude_isovote, rerun_existing=rerun_existing)
        # Compute PR mask based on the EM estimate mask from every iteration
        if compute_PR_every_iter:
            [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
            with open('{}{}GTLSA_EM_prj_iter{}_thresh{}.json'.format(outdir, mode, it, thresh), 'w') as fp:
                fp.write(json.dumps([p, r, j]))
        if DEBUG:
            [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
            print qp1, qn1, qp2, qn2
            print "-->"+str([p, r, j])
        # compute jaccard between previous and current gt estimation mask
        [p_against_prev, r_against_prev, jaccard_against_prev_gt_est] = faster_compute_prj(gt_est_mask, prev_gt_est)
        if DEBUG:
            print "jaccard_against_prev_gt_est:", jaccard_against_prev_gt_est
        prev_gt_est = gt_est_mask
    [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
    with open('{}{}GTLSA_EM_prj_best_thresh.json'.format(outdir, mode), 'w') as fp:
        fp.write(json.dumps([p, r, j]))
    pickle.dump(gt_est_mask, open('{}{}GTLSA_gt_est_mask_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(log_probability_in_mask, open('{}{}GTLSA_p_in_mask_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(log_probability_not_in_mask, open('{}{}GTLSA_p_not_in_mask_best_thresh.pkl'.format(outdir, mode), 'w'))
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


def GT_EM_Qjinit(sample_name, objid, num_iterations=5, load_p_in_mask=False, thresh=0, rerun_existing=False, exclude_isovote=False, compute_PR_every_iter=False):
    print "Doing GT EM (Qj=0.6 initialization)"
    outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    if exclude_isovote:
        mode = 'isoQjinit'
    else:
        mode = ''
    # initialize MV mask
    gt_est_mask = get_MV_mask(sample_name, objid, cluster_id)
    worker_masks = get_all_worker_mega_masks_for_sample(sample_name, objid, cluster_id=cluster_id)
    Nworkers = len(worker_masks)
    mega_mask = get_mega_mask(sample_name, objid, cluster_id)
    for it in range(num_iterations):
        print "Iteration #", it
        qp = dict()
        qn = dict()
        for wid in worker_masks.keys():
            if it == 0:
                qp[wid], qn[wid] = 0.6, 0.6
        else:
            qp[wid], qn[wid] = GTworker_prob_correct(mega_mask, worker_masks[wid], gt_est_mask, Nworkers, exclude_isovote=exclude_isovote)
        # Compute pInMask and pNotInMask
        log_probability_in_mask, log_probability_not_in_mask = GTmask_log_probabilities(worker_masks, qp, qn)
        gt_est_mask = estimate_gt_from(log_probability_in_mask, log_probability_not_in_mask, thresh=thresh)
        if exclude_isovote:
            invariant_mask = np.zeros_like(mega_mask, dtype=bool)
            invariant_mask_yes = np.ma.masked_where((mega_mask == Nworkers), invariant_mask).mask
            invariant_mask_no = np.ma.masked_where((mega_mask == 0), invariant_mask).mask
            gt_est_mask = gt_est_mask+invariant_mask_yes-invariant_mask_no
            gt_est_mask[gt_est_mask == -1] = 0
        if compute_PR_every_iter:
            # Compute PR mask based on the EM estimate mask from every iteration
            [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
            with open('{}{}GT_EM_prj_iter{}_thresh{}.json'.format(outdir, mode, it, thresh), 'w') as fp:
                fp.write(json.dumps([p, r, j]))
        with open('{}{}GT_gt_est_mask_{}_thresh{}.pkl'.format(outdir, mode, it, thresh), 'w') as fp:
            fp.write(pickle.dumps(gt_est_mask))
        pickle.dump(log_probability_in_mask, open('{}{}GT_p_in_mask_{}_thresh{}.pkl'.format(outdir, mode, it, thresh), 'w'))
        pickle.dump(log_probability_not_in_mask, open('{}{}GT_p_not_in_mask_{}_thresh{}.pkl'.format(outdir, mode, it, thresh), 'w'))
        pickle.dump(qp, open('{}{}GT_qp_{}_thresh{}.pkl'.format(outdir, mode, it, thresh), 'w'))
        pickle.dump(qn, open('{}{}GT_qn_{}_thresh{}.pkl'.format(outdir, mode, it, thresh), 'w'))
    if not compute_PR_every_iter:
        # Compute PR mask based on the EM estimate mask from the last iteration
        [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
        with open('{}{}GT_EM_prj_thresh{}.json'.format(outdir, mode, thresh), 'w') as fp:
            fp.write(json.dumps([p, r, j]))
    plt.figure()
    plt.imshow(gt_est_mask, interpolation="none")  # ,cmap="rainbow")
    plt.colorbar()
    plt.savefig('{}{}GT_EM_mask_thresh{}.png'.format(outdir, mode, thresh))


def do_GT_EM_for(sample_name, objid, cluster_id="", rerun_existing=False, exclude_isovote=False, compute_PR_every_iter=False, PLOT=False):
    if exclude_isovote:
        mode = 'iso'
    else:
        mode = ''
    if DEBUG:
        print "Doing GT mode=", mode
        start = time.time()
    if cluster_id != "" and cluster_id != -1:
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    if not rerun_existing:
        if os.path.isfile('{}GT_EM_prj_best_thresh.json'.format(outdir)):
            print "Already ran GT, Skipped"
            return

    # initialize MV mask
    MV = get_MV_mask(sample_name, objid, cluster_id)
    gt_est_mask = MV
    worker_masks = get_all_worker_mega_masks_for_sample(sample_name, objid, cluster_id)
    Nworkers = len(worker_masks)
    mega_mask = get_mega_mask(sample_name, objid, cluster_id)
    prev_gt_est = gt_est_mask
    jaccard_against_prev_gt_est = 0
    it = 0
    max_iter = 6
    while (jaccard_against_prev_gt_est < 0.999 or it <= 1):
    # for it in range(num_iterations):
        if (it >= max_iter):
            break
        if DEBUG:
            print "iteration:", it
        it += 1
        qp = dict()
        qn = dict()
        for wid in worker_masks.keys():
            qp[wid], qn[wid] = GTworker_prob_correct(mega_mask, worker_masks[wid], gt_est_mask, Nworkers, exclude_isovote=exclude_isovote)
            # Compute pInMask and pNotInMask
            log_probability_in_mask, log_probability_not_in_mask = GTmask_log_probabilities(worker_masks, qp, qn)
        p, r, j, thresh, gt_est_mask = binarySearchDeriveBestThresh(sample_name, objid, cluster_id, log_probability_in_mask, log_probability_not_in_mask, MV, exclude_isovote=exclude_isovote, rerun_existing=rerun_existing)
        # gt_est_mask = estimate_gt_from(log_probability_in_mask, log_probability_not_in_mask,thresh=thresh)
        if compute_PR_every_iter:
            # Compute PR mask based on the EM estimate mask from every iteration
            [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
            with open('{}{}GT_EM_prj_iter{}_thresh{}.json'.format(outdir, mode, it, thresh), 'w') as fp:
                fp.write(json.dumps([p, r, j]))
        if DEBUG:
            [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
            print qp, qn
            print "-->"+str([p, r, j])
        # compute jaccard between previous and current gt estimation mask
        [p_against_prev, r_against_prev, jaccard_against_prev_gt_est] = faster_compute_prj(gt_est_mask, prev_gt_est)
        if DEBUG:
            print "jaccard_against_prev_gt_est:", jaccard_against_prev_gt_est
        prev_gt_est = gt_est_mask
    # Save only during the last iteration
    pickle.dump(gt_est_mask, open('{}{}GT_gt_est_mask_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(log_probability_in_mask, open('{}{}GT_p_in_mask_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(log_probability_not_in_mask, open('{}{}GT_p_not_in_mask_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(qp, open('{}{}GT_qp_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(qn, open('{}{}GT_qn_best_thresh.pkl'.format(outdir, mode), 'w'))

    # Compute PR mask based on the EM estimate mask from the last iteration
    [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
    with open('{}{}GT_EM_prj_best_thresh.json'.format(outdir, mode), 'w') as fp:
        fp.write(json.dumps([p, r, j]))
    if PLOT:
        plt.figure()
        plt.imshow(gt_est_mask, interpolation="none")  # ,cmap="rainbow")
        plt.colorbar()
        plt.savefig('{}{}GT_EM_mask_thresh{}.png'.format(outdir, mode, thresh))
    if DEBUG:
        end = time.time()
        print "Time:{:.2f}".format(end-start)


def GroundTruth_doM_once(sample_name, objid, algo, cluster_id="", num_iterations=5, load_p_in_mask=False, rerun_existing=False, compute_PR_every_iter=False, exclude_isovote=False):
    print "Doing GroundTruth_doM_once, algo={},exclude_isovote={}".format(algo, exclude_isovote)
    if cluster_id != "":
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    if exclude_isovote:
        mode = 'iso'
    else:
        mode = ''
    if not rerun_existing:
        # pixel_em/25workers_rand0/obj47/basic_p_in_mask_ground_truth.pkl
        if os.path.isfile('{}{}{}_p_in_mask_ground_truth.pkl'.format(outdir, mode, algo)):
            print "Already ran ground truth experiment, Skipped"
            print '{}{}{}_p_in_mask_ground_truth.pkl'.format(outdir, mode, algo)
            return

    # initialize MV mask
    mega_mask = get_mega_mask(sample_name, objid, cluster_id)
    tiles = pickle.load(open("{}tiles.pkl".format(outdir)))
    area_mask = tiles2AreaMask(tiles, mega_mask)
    gt_est_mask = get_gt_mask(objid)
    worker_masks = get_all_worker_mega_masks_for_sample(sample_name, objid, cluster_id=cluster_id)
    Nworkers = len(worker_masks)
    if algo == 'basic':
        q = dict()
        for wid in worker_masks.keys():
            q[wid] = worker_prob_correct(mega_mask, worker_masks[wid], gt_est_mask, Nworkers, exclude_isovote=exclude_isovote)
        # Compute pInMask and pNotInMask
        log_probability_in_mask, log_probability_not_in_mask = mask_log_probabilities(worker_masks, q)
        pickle.dump(q, open('{}{}{}_q_ground_truth.pkl'.format(outdir, mode, algo), 'w'))
    elif algo == 'GT':
        qp = dict()
        qn = dict()
        for wid in worker_masks.keys():
            qp[wid], qn[wid] = GTworker_prob_correct(mega_mask, worker_masks[wid], gt_est_mask, Nworkers, exclude_isovote=exclude_isovote)
        # Compute pInMask and pNotInMask
        log_probability_in_mask, log_probability_not_in_mask = GTmask_log_probabilities(worker_masks, qp, qn)
        pickle.dump(qp, open('{}{}{}_qp_ground_truth.pkl'.format(outdir, mode, algo), 'w'))
        pickle.dump(qn, open('{}{}{}_qn_ground_truth.pkl'.format(outdir, mode, algo), 'w'))
    elif algo == 'GTLSA':
        qp1 = dict()
        qn1 = dict()
        qp2 = dict()
        qn2 = dict()
        for wid in worker_masks.keys():
            qp1[wid], qn1[wid], qp2[wid], qn2[wid], area_thresh_gt, area_thresh_ngt = GTLSAworker_prob_correct(mega_mask, worker_masks[wid], gt_est_mask, Nworkers, area_mask, tiles, exclude_isovote=exclude_isovote)
            # print "area_thresh_gt,area_thresh_ngt:",area_thresh_gt, area_thresh_ngt
        log_probability_in_mask, log_probability_not_in_mask = GTLSAmask_log_probabilities(worker_masks, qp1, qn1, qp2, qn2, area_mask, area_thresh_gt, area_thresh_ngt)
        pickle.dump(qp1, open('{}{}{}_qp1_ground_truth.pkl'.format(outdir, mode, algo), 'w'))
        pickle.dump(qn1, open('{}{}{}_qn1_ground_truth.pkl'.format(outdir, mode, algo), 'w'))
        pickle.dump(qp2, open('{}{}{}_qp2_ground_truth.pkl'.format(outdir, mode, algo), 'w'))
        pickle.dump(qn2, open('{}{}{}_qn2_ground_truth.pkl'.format(outdir, mode, algo), 'w'))
    '''
    elif algo =="AW":
        worker_qualities = dict()
        for wid in worker_masks.keys():
            worker_qualities[wid] = aw_worker_prob_correct(mega_mask,worker_masks[wid], gt_est_mask,area_lst,Nworkers,exclude_isovote=exclude_isovote)
        #Compute pInMask and pNotInMask
        log_probability_in_mask, log_probability_not_in_mask = mask_log_probabilities(worker_masks,worker_qualities)
    pickle.dump(worker_qualities,open('{}{}{}_q_ground_truth.pkl'.format(outdir,mode,algo), 'w'))
    '''
    if algo == 'GTLSA':
        # Testing:
        area_thres = open("area_thres.txt", 'a')
        gt_areas = area_mask[gt_est_mask == True]
        # print "gt split: ", len(np.where(gt_areas<area_thresh_gt)[0]), len(np.where(gt_areas>=area_thresh_gt)[0])
        ngt_areas = area_mask[gt_est_mask == False]
        # print "ngt split: ",len(np.where(ngt_areas<area_thresh_ngt)[0]),len(np.where(ngt_areas>=area_thresh_ngt)[0])
        area_thres.write("{},{},{},{},{}\n".format(sample_name, objid, algo, area_thresh_gt, area_thresh_ngt))
        area_thres.close()
    pickle.dump(log_probability_in_mask, open('{}{}{}_p_in_mask_ground_truth.pkl'.format(outdir, mode, algo), 'w'))
    pickle.dump(log_probability_not_in_mask, open('{}{}{}_p_not_in_ground_truth.pkl'.format(outdir, mode, algo), 'w'))


def deriveGTinGroundTruthExperiments(sample_name, objid, algo, thresh_lst, cluster_id="", exclude_isovote=False, SAVE_GT_MASK=False, rerun_existing=False):
    all_prjs = []
    if cluster_id != "" and cluster_id != -1:
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    if exclude_isovote:
        mode = 'iso'
    else:
        mode = ''
    if (not rerun_existing) and os.path.exists('{}{}{}_ground_truth_EM_prj_thresh4.json'.format(outdir, mode, algo)):
        print '{}{}{}_ground_truth_EM_prj_thresh4.json'.format(outdir, mode, algo)+" already exist"
        return
    # print outdir
    log_probability_in_mask = pickle.load(open('{}{}{}_p_in_mask_ground_truth.pkl'.format(outdir, mode, algo)))
    log_probability_not_in_mask = pickle.load(open('{}{}{}_p_not_in_ground_truth.pkl'.format(outdir, mode, algo)))
    if exclude_isovote:
        Nworkers = int(sample_name.split("workers")[0])
        mega_mask = get_mega_mask(sample_name, objid, cluster_id)
        invariant_mask = np.zeros_like(mega_mask, dtype=bool)
        invariant_mask_yes = np.ma.masked_where((mega_mask == Nworkers), invariant_mask).mask
        invariant_mask_no = np.ma.masked_where((mega_mask == 0), invariant_mask).mask
    for thresh in thresh_lst:
        outfile = '{}{}{}_ground_truth_EM_prj_thresh{}.json'.format(outdir, mode, algo, thresh)
        gt_est_mask = estimate_gt_from(log_probability_in_mask, log_probability_not_in_mask, thresh=thresh)
        if exclude_isovote:
            gt_est_mask = gt_est_mask+invariant_mask_yes-invariant_mask_no
            gt_est_mask[gt_est_mask < 0] = False
            gt_est_mask[gt_est_mask > 1] = True
            # gt_est_mask = gt_est_mask+invariant_mask_yes
        if SAVE_GT_MASK:
            pickle.dump(gt_est_mask, open('{}{}{}_gt_est_ground_truth_mask_thresh{}.pkl'.format(outdir, mode, algo, thresh), 'w'))
        [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
        # print "p,r,j:", p, r, j
        all_prjs.append([thresh, p, r, j])
        with open(outfile, 'w') as fp:
            fp.write(json.dumps([p, r, j]))
    # return gt_est_mask
    return all_prjs


def onlineDeriveGTinGroundTruthExperiments(sample_name, objid, algo, thresh, cluster_id="", exclude_isovote=False, SAVE_GT_MASK=False, rerun_existing=False, compareWith="gt"):
    if cluster_id != "" and cluster_id != -1:
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    if exclude_isovote:
        mode = 'iso'
    else:
        mode = ''
    if (not rerun_existing) and os.path.exists('{}{}{}_ground_truth_EM_prj_thresh4.json'.format(outdir, mode, algo)):
        print '{}{}{}_ground_truth_EM_prj_thresh4.json'.format(outdir, mode, algo)+" already exist"
        return
    # print outdir
    log_probability_in_mask = pickle.load(open('{}{}{}_p_in_mask_ground_truth.pkl'.format(outdir, mode, algo)))
    log_probability_not_in_mask = pickle.load(open('{}{}{}_p_not_in_ground_truth.pkl'.format(outdir, mode, algo)))
    if exclude_isovote:
        Nworkers = int(sample_name.split("workers")[0])
        mega_mask = get_mega_mask(sample_name, objid, cluster_id)
        invariant_mask = np.zeros_like(mega_mask, dtype=bool)
        invariant_mask_yes = np.ma.masked_where((mega_mask == Nworkers), invariant_mask).mask
        invariant_mask_no = np.ma.masked_where((mega_mask == 0), invariant_mask).mask
    outfile = '{}{}{}_ground_truth_EM_prj_thresh{}.json'.format(outdir, mode, algo, thresh)
    gt_est_mask = estimate_gt_from(log_probability_in_mask, log_probability_not_in_mask, thresh=thresh)
    if exclude_isovote:
        gt_est_mask = gt_est_mask+invariant_mask_yes-invariant_mask_no
        gt_est_mask[gt_est_mask < 0] = False
        gt_est_mask[gt_est_mask > 1] = True
        #gt_est_mask = gt_est_mask+invariant_mask_yes
    if SAVE_GT_MASK:
        pickle.dump(gt_est_mask, open('{}{}{}_gt_est_ground_truth_mask_thresh{}.pkl'.format(outdir, mode, algo, thresh), 'w'))
    if compareWith == "gt":
        compare_mask = get_gt_mask(objid)
    elif compareWith == "MV":
        compare_mask = get_MV_mask(sample_name, objid)
    [p, r, j] = faster_compute_prj(gt_est_mask, compare_mask)
    return [p, r, j], gt_est_mask


def do_EM_for(sample_name, objid, cluster_id="", rerun_existing=False, exclude_isovote=False, compute_PR_every_iter=False, PLOT=False, initMethod="MV"):
    if DEBUG:
        start = time.time()
    if exclude_isovote:
        mode = 'iso'
    else:
        mode = ''
    if cluster_id != "" and cluster_id != -1:
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    if DEBUG:
        print "Doing EM"
    if not rerun_existing:
        if os.path.isfile('{}EM_prj_best_thresh.json'.format(outdir)):
            print "Already ran EM, Skipped"
            return
    MV = get_MV_mask(sample_name, objid, cluster_id)
    if initMethod == "MV":
        # initialize MV mask
        gt_est_mask = MV
        prev_gt_est = gt_est_mask
    else:
        prev_gt_est = 0
    worker_masks = get_all_worker_mega_masks_for_sample(sample_name, objid, cluster_id=cluster_id)
    Nworkers = len(worker_masks)
    mega_mask = get_mega_mask(sample_name, objid, cluster_id)
    if DEBUG:
        t_load = time.time()
        print "Loading time:", t_load-start
    jaccard_against_prev_gt_est = 0
    it = 0
    max_iter = 6
    while (jaccard_against_prev_gt_est < 0.999 or it <= 1):
        worker_qualities = dict()
        if (it == 0 and initMethod == "0.6constant"):
            # a little better than random guess
            print "starting with q=0.6"
            for wid in worker_masks.keys():
                worker_qualities[wid] = 0.6
        if (it >= max_iter):
            break
        if DEBUG:
            print "iteration:", it
        it += 1
        #for it in range(num_iterations):
        if DEBUG:
            t0 = time.time()
        if (it != 0 and initMethod != "0.6constant"):
            # print "computing qj based on model"
            for wid in worker_masks.keys():
                worker_qualities[wid] = worker_prob_correct(mega_mask, worker_masks[wid], gt_est_mask, Nworkers, exclude_isovote=exclude_isovote)
        if DEBUG:
            t1 = time.time()
            print "Time for worker prob calculation:", t1-t0
        # Compute pInMask and pNotInMask
        # print worker_qualities
        # print worker_masks
        log_probability_in_mask, log_probability_not_in_mask = mask_log_probabilities(worker_masks, worker_qualities)
        if DEBUG:
            t2 = time.time()
            print "Time for mask log prob calculation:", t2-t1
        p, r, j, thresh, gt_est_mask = binarySearchDeriveBestThresh(sample_name, objid, cluster_id, log_probability_in_mask, log_probability_not_in_mask, MV, exclude_isovote=exclude_isovote, rerun_existing=rerun_existing)
        #gt_est_mask = estimate_gt_from(log_probability_in_mask, log_probability_not_in_mask,thresh=thresh)
        if DEBUG:
            t3 = time.time()
            print "Time for binary search :", t3-t2
        # Compute PR mask based on the EM estimate mask from the last iteration
        # if compute_PR_every_iter:
        #    # [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
        #    with open('{}{}EM_prj_iter{}_thresh{}.json'.format(outdir,mode,it,thresh), 'w') as fp:
        #        fp.write(json.dumps([p, r, j]))
        if DEBUG:
            print worker_qualities
            [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
            print "-->"+str([p, r, j])
        # compute jaccard between previous and current gt estimation mask
        [p_against_prev, r_against_prev, jaccard_against_prev_gt_est] = faster_compute_prj(gt_est_mask, prev_gt_est)
        if DEBUG:
            print "jaccard_against_prev_gt_est:", jaccard_against_prev_gt_est
        prev_gt_est = gt_est_mask
    # Only writing output at the end of all iterations:
    pickle.dump(gt_est_mask, open('{}{}gt_est_mask_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(log_probability_in_mask, open('{}{}p_in_mask_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(log_probability_not_in_mask, open('{}{}p_not_in_mask_best_thresh.pkl'.format(outdir, mode), 'w'))
    pickle.dump(worker_qualities, open('{}{}Qj_best_thresh.pkl'.format(outdir, mode), 'w'))
    # Compute PR mask based on the EM estimate mask from the last iteration
    [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
    with open('{}{}EM_prj_best_thresh.json'.format(outdir, mode), 'w') as fp:
        fp.write(json.dumps([p, r, j]))
    if PLOT:
        plt.figure()
        plt.imshow(gt_est_mask, interpolation="none")  # ,cmap="rainbow")
        plt.colorbar()
        plt.savefig('{}{}EM_mask_thresh{}.png'.format(outdir, mode, thresh))
    if DEBUG:
        end = time.time()
        print "Time:{:.2f}".format(end-start)


def compile_PR(mode="", ground_truth=False):
    import glob
    import csv
    if ground_truth:
        fname = '{}{}_ground_truth_full_PRJ_table.csv'.format(PIXEL_EM_DIR, mode)
    else:
        fname = '{}{}_full_PRJ_table.csv'.format(PIXEL_EM_DIR, mode)
    with open(fname, 'w') as csvfile:
        fieldnames = ['num_workers', 'sample_num', 'objid', 'thresh', 'clust', 'precision', 'recall', 'jaccard', 'FPR%', 'FNR%']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample_path in glob.glob('{}*_rand*/'.format(PIXEL_EM_DIR)):
            sample_name = sample_path.split('/')[-2]
            print "Working on ", sample_path
            num_workers = int(sample_name.split('w')[0])
            sample_num = int(sample_name.split('d')[-1])
            for obj_path in glob.glob('{}obj*/'.format(sample_path)):
                objid = int(obj_path.split('/')[-2].split('j')[1])
                for clust_path in glob.glob("{}/clust*/".format(obj_path))+[obj_path]:
                    # clust_path includes both original obj_path and the paths with clust*/ on it
                    if clust_path == obj_path:
                        cluster_id = -1  # unclustered flag
                    else:
                        cluster_id = int(clust_path.split("/clust")[-1][:-1])
                    # print "----clust path:",clust_path
                    # print cluster_id
                    p = None
                    r = None
                    j = None
                    fpr = None
                    fnr = None
                    thresh = "best"
                    if ground_truth:
                        pr_file = '{}{}_ground_truth_EM_prj_best_thresh.json'.format(clust_path, mode)
                        # pr_file = '{}{}_ground_truth_EM_prj_thresh{}.json'.format(clust_path,mode,thresh)
                    else:
                        # GT_EM_prj_best_thresh.json
                        pr_file = '{}{}_EM_prj_best_thresh.json'.format(clust_path, mode)
                        fpnr_file = '{}{}_EM_fpnr_best_thresh.json'.format(clust_path, mode)
                        if mode == "basic":
                            pr_file = '{}EM_prj_best_thresh.json'.format(clust_path)
                            fpnr_file = '{}EM_fpnr_best_thresh.json'.format(clust_path)
                        elif mode == "MV":
                            pr_file = '{}MV_prj.json'.format(clust_path)
                            fpnr_file = '{}MV_fpnr.json'.format(clust_path)
                    #print pr_file
                    #print  fpnr_file
                    if os.path.isfile(pr_file):
                        [p, r, j] = json.load(open(pr_file))
                    if os.path.isfile(fpnr_file):
                        [fpr, fnr] = json.load(open(fpnr_file))
                    else:
                        gt_fname = "{}/{}_gt_est_mask_best_thresh.pkl".format(clust_path, mode)
                        if mode == "basic":
                            gt_fname = "{}/gt_est_mask_best_thresh.pkl".format(clust_path)
                        elif mode == "MV":
                            gt_fname = "{}/MV_mask.pkl".format(clust_path)
                        if os.path.isfile(gt_fname):
                            result = pickle.load(open(gt_fname))
                            gt = get_gt_mask(objid)
                            [fpr, fnr] = TFPNR(result, gt)
                            with open(fpnr_file, 'w') as fp:
                                fp.write(json.dumps([fpr, fnr]))
                            print fpr, fnr
                    # print p,r,j,fpr,fnr
                    # print clust_path
                    if any([prj is not None for prj in [p, r, j]]):
                        writer.writerow({
                            'num_workers': num_workers,
                            'sample_num': sample_num,
                            'objid': objid,
                            'thresh': thresh,
                            'clust': cluster_id,
                            'precision': p,
                            'recall': r,
                            'jaccard': j,
                            'FPR%': fpr,
                            'FNR%': fnr
                        })
    print 'Compiled PR to :' + fname


def binarySearchDeriveGTinGroundTruthExperiments(sample, objid, algo, cluster_id="", exclude_isovote=False, rerun_existing=False, compareWith="gt"):
    thresh_min = -200
    thresh_max = 200
    if cluster_id != "" and cluster_id != -1:
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample, objid)
    if exclude_isovote:
        mode = 'iso'
    else:
        mode = ''
    if (not rerun_existing) and os.path.exists('{}{}{}_ground_truth_EM_prj_best_thresh.json'.format(outdir, mode, algo)):
        print '{}{}{}_ground_truth_EM_prj_best_thresh.json'.format(outdir, mode, algo)+" already exist"
        return
    delta = np.abs(thresh_max - thresh_min)
    thresh = (thresh_min+thresh_max)/2.
    p, r = 0, -1
    iterations = 0
    epsilon = 0.125
    while (iterations <= 100 or p == -1):  # continue iterations below max iterations or if p=-1
        # stop if p=r or if delta (range in x) gets below a certain threshold
        if (p == r) or (thresh_min + epsilon >= thresh_max):
            break
        [p, r, j], gt_est_mask = onlineDeriveGTinGroundTruthExperiments(sample, objid, algo, thresh, cluster_id=cluster_id, exclude_isovote=exclude_isovote, rerun_existing=True, compareWith=compareWith)
        delta = np.abs(thresh_max - thresh_min)
        if p > r:
            right = thresh_min + 0.75*delta
            thresh_max = right
        else:
            left = thresh_min + 0.25*delta
            thresh_min = left
        if p == -1:
            # if p =-1 then it is because the result area is zero, which means nothing was selected for gt
            # this meant that the threshold has overshot
            thresh_max = thresh_min+0.2*delta
        thresh = (thresh_min+thresh_max)/2.
        iterations += 1
        if DEBUG:
            print "----Trying threshold:", thresh, "-----"
            print p, r, j, thresh_max, thresh_min
            print "actual prj against GT", faster_compute_prj(gt_est_mask, get_gt_mask(objid))
            # plt.figure()
            # plt.title("Iter #"+str(iterations))
            # plt.imshow(gt_est_mask)
            # plt.colorbar()
    outfile = '{}{}{}_ground_truth_cw{}_EM_prj_best_thresh.json'.format(outdir, mode, algo, compareWith)
    if compareWith == "MV":
        [p, r, j] = faster_compute_prj(gt_est_mask, get_gt_mask(objid))
        print p, r, j
    with open(outfile, 'w') as fp:
        fp.write(json.dumps([p, r, j]))
    pickle.dump(gt_est_mask, open('{}{}{}_gt_est_ground_truth_mask_cw{}_best_thresh.pkl'.format(outdir, mode, algo, compareWith), 'w'))
    return p, r, j


##############################################
def estimate_gt_compute_PRJ_against_MV(sample_name, objid, cluster_id, log_probability_in_mask, log_probability_not_in_mask, MV, thresh, exclude_isovote=False):
    if exclude_isovote:
        Nworkers = int(sample_name.split("workers")[0])
        mega_mask = get_mega_mask(sample_name, objid, cluster_id)
        invariant_mask = np.zeros_like(mega_mask, dtype=bool)
        invariant_mask_yes = np.ma.masked_where((mega_mask == Nworkers), invariant_mask).mask
        invariant_mask_no = np.ma.masked_where((mega_mask == 0), invariant_mask).mask
    gt_est_mask = estimate_gt_from(log_probability_in_mask, log_probability_not_in_mask, thresh=thresh)
    if exclude_isovote:
        gt_est_mask = gt_est_mask+invariant_mask_yes-invariant_mask_no
        gt_est_mask[gt_est_mask < 0] = False
        gt_est_mask[gt_est_mask > 1] = True
        # gt_est_mask = gt_est_mask+invariant_mask_yes
    # PRJ values against MV
    [p, r, j] = faster_compute_prj(gt_est_mask, MV)
    return [p, r, j], gt_est_mask


def binarySearchDeriveBestThresh(sample_name, objid, cluster_id, log_probability_in_mask, log_probability_not_in_mask, MV, exclude_isovote=False, rerun_existing=False):
    thresh_min = -200
    thresh_max = 200
    delta = np.abs(thresh_max - thresh_min)
    thresh = (thresh_min+thresh_max)/2.
    p, r = 0, -1
    iterations = 0
    epsilon = 0.125
    while (iterations <= 100 or p == -1):  # continue iterations below max iterations or if p=-1
        # stop if p=r or if delta (range in x) gets below a certain threshold
        if (p == r) or (thresh_min + epsilon >= thresh_max):
            break
        [p, r, j], gt_est_mask = estimate_gt_compute_PRJ_against_MV(sample_name, objid, cluster_id, log_probability_in_mask, log_probability_not_in_mask, MV, thresh, exclude_isovote=exclude_isovote)
        delta = np.abs(thresh_max - thresh_min)
        if p > r:
            right = thresh_min + 0.75*delta
            thresh_max = right
        else:
            left = thresh_min + 0.25*delta
            thresh_min = left
        if p == -1:
            # if p =-1 then it is because the result area is zero, which means nothing was selected for gt
            # this meant that the threshold has overshot
            thresh_max = thresh_min+0.2*delta
        thresh = (thresh_min+thresh_max)/2.
        iterations += 1
        if DEBUG:
            print "----Trying threshold:", thresh, "-----"
            print p, r, j, thresh_max, thresh_min
            print "actual prj against GT", faster_compute_prj(gt_est_mask, get_gt_mask(objid))
            # plt.figure()
            # plt.title("Iter #"+str(iterations))
            # plt.imshow(gt_est_mask)
            # plt.colorbar()
    return p, r, j, thresh, gt_est_mask

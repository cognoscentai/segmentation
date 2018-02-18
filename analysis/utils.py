# from collections import defaultdict
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import csv
import os
import pickle
import json
import  pandas as pd
DEBUG = False
CURR_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
DATA_DIR = os.path.abspath(os.path.join(CURR_DIR, '..')) + '/data/'
VISION_DIR = CURR_DIR + 'vision-stuff/'
VISION_TILES_DIR = VISION_DIR + 'pixel-vision-tiles/'
PIXEL_EM_DIR = CURR_DIR + 'pixel_em/'


def hybrid_dir(sample_name, objid, k, expand_thresh, contract_thresh):
    hydir = '{}{}/obj{}/hybrid/{}/({},{})'.format(PIXEL_EM_DIR, sample_name, objid, k, expand_thresh, contract_thresh)
    if not os.path.isdir(hydir):
        os.makedirs(hydir)
    return hydir


def vision_pixtile_dir(img_id, k=500):
    vdir = '{}/{}/{}'.format(VISION_TILES_DIR, k, img_id)
    if not os.path.isdir(vdir):
        os.makedirs(vdir)
    return vdir


def vision_baseline_dir(objid, k=500, include_thresh=0.1):
    vdir = '{}obj{}/vision-only/{}/{}'.format(PIXEL_EM_DIR, objid, k, include_thresh)
    if not os.path.isdir(vdir):
        os.makedirs(vdir)
    return vdir


def show_mask(mask, figname=None):
    plt.figure()
    plt.imshow(mask, interpolation="none")  # ,cmap="rainbow")
    plt.colorbar()
    if figname is not None:
        plt.savefig(figname)
    else:
        plt.show()
    plt.close()


def get_mega_mask(sample_name, objid, cluster_id=""):
    if cluster_id != "":
        indir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    else:
        indir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    return pickle.load(open('{}mega_mask.pkl'.format(indir)))


def workers_in_sample(sample_name, objid, cluster_id=""):
    if cluster_id != "":
        indir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    else:
        indir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    return json.load(open('{}worker_ids.json'.format(indir)))


def get_all_worker_mega_masks_for_sample(sample_name, objid, cluster_id=""):
    worker_masks = dict()  # key = worker_id, value = worker mask
    worker_ids = workers_in_sample(sample_name, objid, cluster_id=cluster_id)
    for wid in worker_ids:
        worker_masks[wid] = get_worker_mask(objid, wid)
    return worker_masks


def get_MV_mask(sample_name, objid, cluster_id=""):
    if cluster_id != "":
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    if not os.path.exists('{}MV_mask.pkl'.format(outdir)):
        compute_PRJ_MV(sample_name, objid, cluster_id)
    return pickle.load(open('{}MV_mask.pkl'.format(outdir)))


def get_worker_mask(objid, worker_id):
    indir = '{}obj{}/'.format(PIXEL_EM_DIR, objid)
    return pickle.load(open('{}mask{}.pkl'.format(indir, worker_id)))


def get_gt_mask(objid):
    indir = '{}obj{}/'.format(PIXEL_EM_DIR, objid)
    return pickle.load(open('{}gt.pkl'.format(indir)))


def create_objid_to_clustid():
    # create and dump dictionary
    # clust_ids[sample_num][objid] ---> list of clust IDs
    # for example, clust_ids['5workers_rand0'][1] = [0, 1]
    from sample_worker_seeds import sample_specs
    from collections import defaultdict
    clust_ids = defaultdict(dict)
    df = pd.read_csv("spectral_clustering_all_hard_obj.csv")
    best_clust = pd.read_csv("best_clust_picking.csv")
    for sample in sample_specs.keys():
        for objid in df.objid.unique():
            cluster_ids = df[(df["objid"] == objid)].cluster.unique()
            for cluster_id in cluster_ids:
                if len(best_clust[(best_clust["sample"] == sample) & (best_clust["objid"] == objid) & (best_clust["clust"] == cluster_id)]) == 1:
                    #print sample + ":" + str(objid)+"clust"+str(cluster_id)
                    clust_ids[sample][int(objid)] = cluster_id
    with open('objid_to_clustid.json', 'w') as fp:
        fp.write(json.dumps(clust_ids))


def clusters(rerun=True):
    # return all valid cluster ids for a given obj
    if rerun or not os.path.isfile('objid_to_clustid.json'):
        create_objid_to_clustid()

    with open('objid_to_clustid.json', 'r') as fp:
        return json.loads(fp.read())


def faster_compute_prj(result, gt):
    intersection = len(np.where(((result == 1) | (gt == 1)) & (result == gt))[0])
    gt_area = float(len(np.where(gt == 1)[0]))
    result_area = float(len(np.where(result == 1)[0]))
    try:
        precision = intersection / result_area
    except(ZeroDivisionError):
        precision = -1
    try:
        recall = intersection / gt_area
    except(ZeroDivisionError):
        recall = -1
    try:
        jaccard = intersection / (gt_area + result_area - intersection)
    except(ZeroDivisionError):
        jaccard = -1
    return precision, recall, jaccard


def TFPNR(result, gt):
    # True False Positive Negative Rates
    # as defined in https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Definitions
    intersection = len(np.where(((result == 1) | (gt == 1)) & (result == gt))[0])
    gt_area = float(len(np.where(gt == 1)[0]))
    result_area = float(len(np.where(result == 1)[0]))

    TP = intersection
    FP = result_area - intersection
    FN = gt_area - intersection
    TN = np.product(np.shape(result)) - (gt_area+result_area-intersection)

    #TPR = TP/float(TP+FN)
    FPR = FP/float(FP+TN)
    FNR = FN/float(TP+FN)
    #TNR = TN/float(TN+FP)
    #assert TPR+FNR==1 and TNR+FPR==1

    #return  TPR,TNR#,FNR,TNR,FPR
    return FPR*100, FNR*100


def tiles_to_mask(tile_id_list, tile_to_pix_dict, base_mask):
    # given a list of chosen tids, converts that to a pixel mask of all chosen pixels
    tile_mask = np.zeros_like(base_mask)
    for tid in tile_id_list:
        for pix in tile_to_pix_dict[tid]:
            tile_mask[pix] = 1
    return tile_mask


def get_pixtiles(objid, k=500):
    obj_to_img_id = get_obj_to_img_id()
    img_id = obj_to_img_id[objid]
    vdir = vision_pixtile_dir(img_id, k)
    with open('{}/pixtile_mask.pkl'.format(vdir), 'r') as fp:
        mask = pickle.loads(fp.read())

    with open('{}/pixtile_list.pkl'.format(vdir), 'r') as fp:
        tiles = pickle.loads(fp.read())
    return mask, tiles


def get_img_id_to_name():
    img_id_to_name = {}
    with open('{}image.csv'.format(DATA_DIR), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_id_to_name[row['id']] = row['filename']
    return img_id_to_name


def get_obj_to_img_id():
    obj_to_img_id = {}
    with open('{}object.csv'.format(DATA_DIR), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                obj_to_img_id[int(row['id'])] = row['image_id']
            except:
                if DEBUG: print 'Reading object.csv table, skipped row: ', row
    return obj_to_img_id


def compute_PRJ_MV(sample_name, objid, cluster_id="", plot=False, mode=""):
    # worker_masks = get_all_worker_mega_masks_for_sample(sample_name, objid)
    if cluster_id != "":
        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid, cluster_id)
    else:
        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)

    if os.path.exists('{}MV_prj.json'.format(outdir)):
        print "MV already exist"
        return json.load(open('{}MV_prj.json'.format(outdir)))

    if mode == "":
        num_workers = len(workers_in_sample(sample_name, objid, cluster_id=cluster_id))
        mega_mask = get_mega_mask(sample_name, objid, cluster_id=cluster_id)
        MV_mask = np.zeros((len(mega_mask), len(mega_mask[0])))
        [xs, ys] = np.where(mega_mask > (num_workers / 2))
        for i in range(len(xs)):
            MV_mask[xs[i]][ys[i]] = 1
        with open('{}MV_mask.pkl'.format(outdir), 'w') as fp:
            fp.write(pickle.dumps(MV_mask))

        if plot:
            plt.figure()
            plt.imshow(MV_mask, interpolation="none")  # ,cmap="rainbow")
            plt.colorbar()
            plt.savefig('{}MV_mask.png'.format(outdir))
    elif mode == "compute_pr_only":
        MV_mask = pickle.load(open('{}MV_mask.pkl'.format(outdir)))

    # Computing MV PRJ against Ground Truth
    gt = get_gt_mask(objid)
    p, r, j = faster_compute_prj(MV_mask, gt)
    with open('{}MV_prj.json'.format(outdir), 'w') as fp:
        fp.write(json.dumps([p, r, j]))

    return p, r, j


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def visualize_test_gt_vision_overlay(batch, objid, k, test_mask, outdir=None):
    if outdir is None:
        outdir = VISION_DIR + 'visualizing_and_testing_stuff/{}/obj{}/{}'.format(batch, objid, k)

    vision_mask, vision_tiles = get_pixtiles(objid, k)
    gt_mask = get_gt_mask(objid)
    mega_mask = get_mega_mask(batch, objid)

    # pick all vision tiles that intersect with either test_mask or gt_mask
    # zoom in to area restricted by those tiles
    # plot overlay
    intersecting_viz_tiles = set()
    min_x = 10000
    max_x = 0
    min_y = 10000
    max_y = 0

    # num_satisfied = 0
    # num_not = 0
    for tid in range(len(vision_tiles)):
        for pix in vision_tiles[tid]:
            if test_mask[pix] or gt_mask[pix]:
                # num_satisfied += 1
                # print test_mask[pix], gt_mask[pix]
                intersecting_viz_tiles.add(tid)
                xs, ys = zip(*vision_tiles[tid])
                min_x = min([min_x, min(xs)])
                max_x = max([max_x, max(xs)])
                min_y = min([min_y, min(ys)])
                max_y = max([max_y, max(ys)])
                continue
            # else:
            #     num_not += 1

    # print len(vision_tiles), len(intersecting_viz_tiles)

    x_range = max_x - min_x + 1
    y_range = max_y - min_y + 1

    # print '[{},{}], [{},{}]'.format(min_x, max_x, min_y, max_y)
    zoomed_viz_mask = np.zeros((x_range, y_range))
    zoomed_test_mask = np.zeros((x_range, y_range))
    zoomed_gt_mask = np.zeros((x_range, y_range))
    zoomed_mega_mask = np.zeros((x_range, y_range))
    zoomed_test_gt_overlap = np.zeros((x_range, y_range))
    zoomed_test_extra = np.zeros((x_range, y_range))
    zoomed_test_missed = np.zeros((x_range, y_range))
    zoomed_test_correct = np.zeros((x_range, y_range))
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            zoomed_viz_mask[x - min_x][y - min_y] = vision_mask[x][y]
            zoomed_test_mask[x - min_x][y - min_y] = test_mask[x][y]
            zoomed_gt_mask[x - min_x][y - min_y] = gt_mask[x][y]
            zoomed_mega_mask[x - min_x][y - min_y] = mega_mask[x][y]
            zoomed_test_extra[x - min_x][y - min_y] = 1 if (test_mask[x][y] and not gt_mask[x][y]) else 0
            zoomed_test_missed[x - min_x][y - min_y] = 1 if (gt_mask[x][y] and not test_mask[x][y]) else 0
            zoomed_test_correct[x - min_x][y - min_y] = 1 if (gt_mask[x][y] and test_mask[x][y]) else 0
            zoomed_test_gt_overlap[x - min_x][y - min_y] = (
                0 if not test_mask[x][y] and not gt_mask[x][y]
                else 1 if test_mask[x][y] and not gt_mask[x][y]
                else 2 if not test_mask[x][y] and gt_mask[x][y]
                else 3
            )

    # visualize basic masks
    show_mask(zoomed_test_mask, figname='{}/test_mask.png'.format(outdir))
    show_mask(zoomed_viz_mask, figname='{}/vision_mask.png'.format(outdir))
    show_mask(zoomed_mega_mask, figname='{}/mega_mask.png'.format(outdir))

    # visualize test_mask overlaid on ground truth
    ax = plt.gca()
    ax.imshow(zoomed_test_correct, cmap='Greens', alpha=0.5)
    ax.imshow(zoomed_test_extra, cmap='Blues', alpha=0.3)
    ax.imshow(zoomed_test_missed, cmap='Reds', alpha=0.3)
    plt.draw()
    plt.savefig('{}/test_gt_overlay.png'.format(outdir))
    plt.close()

    # visualize test_mask overlaid on ground truth with vision tiles background
    ax = plt.gca()
    ax.imshow(zoomed_viz_mask, cmap="gray")
    ax.imshow(zoomed_test_correct, cmap='Greens', alpha=0.3)
    ax.imshow(zoomed_test_extra, cmap='Blues', alpha=0.3)
    ax.imshow(zoomed_test_missed, cmap='Reds', alpha=0.3)
    plt.draw()
    plt.savefig('{}/viz_test_gt_overlay.png'.format(outdir))
    plt.close()

    return zoomed_viz_mask, zoomed_test_mask, zoomed_gt_mask, zoomed_mega_mask


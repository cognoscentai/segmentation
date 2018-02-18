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

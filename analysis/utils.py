# from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import csv
import os
import pickle
import json


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


def get_worker_mask(objid, worker_id):
    indir = '{}obj{}/'.format(PIXEL_EM_DIR, objid)
    return pickle.load(open('{}mask{}.pkl'.format(indir, worker_id)))


def get_gt_mask(objid):
    indir = '{}obj{}/'.format(PIXEL_EM_DIR, objid)
    return pickle.load(open('{}gt.pkl'.format(indir)))


def get_MV_mask(sample_name, objid):
    indir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    return pickle.load(open('{}MV_mask.pkl'.format(indir)))


def get_mega_mask(sample_name, objid):
    indir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    return pickle.load(open('{}mega_mask.pkl'.format(indir)))


def workers_in_sample(sample_name, objid):
    indir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
    return json.load(open('{}worker_ids.json'.format(indir)))


def get_all_worker_mega_masks_for_sample(sample_name, objid):
    worker_masks = dict()  # key = worker_id, value = worker mask
    worker_ids = workers_in_sample(sample_name, objid)
    for wid in worker_ids:
        worker_masks[wid] = get_worker_mask(objid, wid)
    return worker_masks


def precision_and_recall(test_mask, gt_mask):
    num_intersection = 0.0  # float(len(np.where(test_mask == gt_mask)[0]))
    num_test = 0.0  # float(len(np.where(test_mask == 1)[0]))
    num_gt = 0.0  # float(len(np.where(gt_mask == 1)[0]))
    for i in range(len(gt_mask)):
        for j in range(len(gt_mask[i])):
            if test_mask[i][j] == 1 and gt_mask[i][j] == 1:
                num_intersection += 1
                num_test += 1
                num_gt += 1
            elif test_mask[i][j] == 1:
                num_test += 1
            elif gt_mask[i][j] == 1:
                num_gt += 1
    return (num_intersection / num_test), (num_intersection / num_gt)


def jaccard(test_mask, gt_mask):
    num_intersection = 0.0  # float(len(np.where(test_mask == gt_mask)[0]))
    num_test = 0.0  # float(len(np.where(test_mask == 1)[0]))
    num_gt = 0.0  # float(len(np.where(gt_mask == 1)[0]))
    for i in range(len(gt_mask)):
        for j in range(len(gt_mask[i])):
            if test_mask[i][j] == 1 and gt_mask[i][j] == 1:
                num_intersection += 1
                num_test += 1
                num_gt += 1
            elif test_mask[i][j] == 1:
                num_test += 1
            elif gt_mask[i][j] == 1:
                num_gt += 1
    return (num_intersection) / (num_test - num_intersection + num_gt)


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
                print 'Reading object.csv table, skipped row: ', row
    return obj_to_img_id

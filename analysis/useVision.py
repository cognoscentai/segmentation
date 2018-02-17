from utils import get_pixtiles, get_gt_mask, get_MV_mask, \
    hybrid_dir, vision_baseline_dir, faster_compute_prj, clusters
from collections import defaultdict
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pickle
import json


def compute_hybrid_mask(base_mask, agg_vision_mask, expand_thresh=0.8, contract_thresh=0, objid=None, vision_only=False, DEBUG=False):
    # objid only for debugging purposes
    intersection_area = defaultdict(float)  # key = v_tile id taken as value of agg_vision_mask[i][j]
    vtile_area = defaultdict(float)  # key = v_tile id taken as value of agg_vision_mask[i][j]
    base_mask_area = 0.0
    print 'Num unique vision tiles: ', len(np.unique(agg_vision_mask))
    for i in range(len(agg_vision_mask)):
        for j in range(len(agg_vision_mask[0])):
            vtile_id = agg_vision_mask[i][j]
            if vtile_id == 0:
                continue
            vtile_area[vtile_id] += 1
            if base_mask[i][j]:
                base_mask_area += 1
                intersection_area[vtile_id] += 1
    if vision_only:
        # only include or delete full vision tiles
        # only using base mask to decide which to include / delete
        final_mask = np.zeros_like(base_mask)
    else:
        final_mask = np.copy(base_mask)
    # for vtile_id in np.unique(agg_vision_mask):
    #     if vtile_id == 0:
    #         continue

    for vtile_id in vtile_area:
        # for each vtile, decide o either fill out leftovers of it, delete it's intersection with base_mask
        # or leave unchanged
        if intersection_area[vtile_id] == 0:
            continue
        elif DEBUG:
            print 'Intersection area: ', intersection_area[vtile_id]
            print 'vtile area: ', vtile_area[vtile_id]
            print 'Base mask area: ', base_mask_area
        if (float(intersection_area[vtile_id]) / vtile_area[vtile_id]) > expand_thresh:
            # expand mask to include entire vision tile
            final_mask[agg_vision_mask == vtile_id] = True
            if DEBUG:
                print 'Expanding'
        elif (float(intersection_area[vtile_id]) / vtile_area[vtile_id]) < contract_thresh:
            # delete mask to exclude entire vision tile
            final_mask[agg_vision_mask == vtile_id] = False
            if DEBUG:
                print 'Deleting'
        elif DEBUG:
            print 'Passing'
        if DEBUG:
            v_mask = np.copy(agg_vision_mask)
            v_mask[v_mask != vtile_id] = 0
            v_mask[v_mask == vtile_id] = 20
            plot_base_mask = np.copy(base_mask).astype(int) * 50
            plot_gt_mask = np.copy(get_gt_mask(objid)).astype(int) * 100
            plot_sum_mask = v_mask + plot_base_mask + plot_gt_mask
            plt.figure()
            plt.imshow(plot_sum_mask, interpolation="none")  # , cmap="hot")
            plt.colorbar()
            plt.show()
            plt.close()
    return final_mask


def create_and_store_hybrid_masks(sample_name, objid, clust="", base='MV', k=500, expand_thresh=0.8, contract_thresh=0.2):
    agg_vision_mask, _ = get_pixtiles(objid, k)
    MV_mask = get_MV_mask(sample_name, objid, cluster_id=clust)
    gt_mask = get_gt_mask(objid)

    clust_num = '-1' if clust == "" else str(clust)
    algo_name = base + '_' + clust_num

    if base == 'MV':
        # MV hybrid
        base_mask = get_MV_mask(sample_name, objid, cluster_id=clust)
    else:
        print 'Only supports MV base right now'
        raise NotImplementedError

    outdir = hybrid_dir(sample_name, objid, k, expand_thresh, contract_thresh)

    hybrid_mask = compute_hybrid_mask(base_mask, agg_vision_mask, expand_thresh=expand_thresh, contract_thresh=contract_thresh, objid=objid)
    with open('{}/{}_hybrid_mask.pkl'.format(outdir, algo_name), 'w') as fp:
        fp.write(pickle.dumps(hybrid_mask))

    sum_mask = hybrid_mask.astype(int) * 5 + MV_mask.astype(int) * 20 + gt_mask.astype(int) * 50

    plt.figure()
    plt.imshow(sum_mask, interpolation="none")  # , cmap="rainbow")
    plt.colorbar()
    plt.savefig('{}/{}_hybrid_mask.png'.format(outdir, algo_name))
    plt.close()

    p, r, j = faster_compute_prj(hybrid_mask, gt_mask)
    with open('{}/{}_hybrid_prj.json'.format(outdir, algo_name), 'w') as fp:
        fp.write(json.dumps([p, r, j]))


def create_and_store_vision_plus_gt_baseline(objid, k=500, include_thresh=0.5):
    agg_vision_mask, _ = get_pixtiles(objid)
    gt_mask = get_gt_mask(objid)

    outdir = vision_baseline_dir(objid, k, include_thresh)

    vision_only_mask = compute_hybrid_mask(gt_mask, agg_vision_mask, expand_thresh=include_thresh, contract_thresh=0, vision_only=True)
    with open('{}/vision_with_gt_mask.pkl'.format(outdir), 'w') as fp:
        fp.write(pickle.dumps(vision_only_mask))

    sum_mask = vision_only_mask.astype(int) * 5 + gt_mask.astype(int) * 10

    plt.figure()
    plt.imshow(sum_mask, interpolation="none")  # , cmap="rainbow")
    plt.colorbar()
    plt.savefig('{}/vision_with_gt_viz.png'.format(outdir))
    plt.close()

    p, r, j = faster_compute_prj(vision_only_mask, gt_mask)
    with open('{}vision_prj.json'.format(outdir), 'w') as fp:
        fp.write(json.dumps([p, r, j]))


def compile_PR():
    import glob
    import csv
    # compiles a PRJ table for all hybrid masks
    # dir structure:
    # pixel_em/batch_name/objid/hybrid/k/(expand_thresh,contract_thresh)/base_clust_hybrid_prj.json

    fname = 'pixel_em/hybrid_prj_table.csv'
    with open(fname, 'w') as csvfile:
        fieldnames = ['sample_num', 'objid', 'base_algo', 'clust', 'precision', 'recall', 'jaccard']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for path in glob.glob('pixel_em/*/*/hybrid/*/*/*_hybrid_prj.json'):
            splits = path.split('/')
            # print splits
            batch = splits[1]
            objid = splits[2].split('j')[1]
            k = splits[4]
            threshs = splits[5].split(',')
            expand_thresh = threshs[0][1:]
            contract_thresh = threshs[1][:-1]
            base_algo = '_'.join(splits[6].split('_')[:-3])
            clust_id = splits[6].split('_')[-3]

            print batch, objid, k, expand_thresh, contract_thresh, base_algo, clust_id

            with open(path, 'r') as fp:
                [p, r, j] = json.loads(fp.read())

                writer.writerow({
                    'sample_num': batch,
                    'objid': objid,
                    'base_algo': base_algo,
                    'clust': clust_id,
                    'precision': p,
                    'recall': r,
                    'jaccard': j
                })


if __name__ == '__main__':
    obj_clusters = clusters()
    for k in [500]:
        for objid in [2]: #range(1, 2):
            print '*****************************************************************'
            print 'Compute vision baseline for obj', objid
            create_and_store_vision_plus_gt_baseline(objid, k, include_thresh=0.5)
            for batch in ['5workers_rand0']:
		if objid in obj_clusters[batch]:
		    clusts = [""] + [obj_clusters[batch][objid]]
		else:
		    clusts = [""]
                for clust in clusts: 
                #for clust in [""]:
                    print 'Compute vision hybrid for batch', batch, 'clust:', clust
                    create_and_store_hybrid_masks(batch, objid, clust=clust, base='MV', k=500, expand_thresh=0.8, contract_thresh=0.2)

    compile_PR()

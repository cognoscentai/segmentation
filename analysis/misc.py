from utils import get_pixtiles, get_gt_mask, get_MV_mask, get_mega_mask, VISION_DIR, faster_compute_prj, \
    visualize_test_gt_vision_overlay
from useVision import compute_hybrid_mask
import os.path
import numpy as np


def testing_viz_test_gt_vision_overlay():
    # try out different result masks and overlay against vision and gt
    # test potential for improvement
    for batch in ['5workers_rand0']:
        for k in [500]:
            for objid in range(1, 2):
                print 'Visualizing stuff for obj ', objid
                outdir = VISION_DIR + 'visualizing_and_testing_stuff/obj{}/{}'.format(objid, k)
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                # all_worker_tiles = pickle.load(open('{}/obj{}/tiles.pkl'.format(batch_path, objid)))
                vision_mask, vision_tiles = get_pixtiles(objid, k)
                gt_mask = get_gt_mask(objid)
                worker_mega_mask = get_mega_mask(batch, objid)
                mv_mask = get_MV_mask(batch, objid)

                # worker_tiles_mask, num_votes, tile_area, tile_int_area = process_all_worker_tiles(
                #     all_worker_tiles, worker_mega_mask, gt_mask)

                print 'Computing simple hybrid mask...'
                print 'MV p, r, j: ', faster_compute_prj(mv_mask, gt_mask)
                all_voted_mask = np.zeros_like(worker_mega_mask)
                num_workers = int(batch.split('workers')[0])
                all_voted_mask[np.where(worker_mega_mask == num_workers)] = 1
                all_mv_but_low_confidence = mv_mask - all_voted_mask
                hybrid_mask = all_voted_mask + compute_hybrid_mask(all_mv_but_low_confidence, vision_mask, expand_thresh=0.8, contract_thresh=0.2, objid=objid, DEBUG=False)
                hybrid_mask[np.where(hybrid_mask > 1)] = 1
                # show_mask(hybrid_mask, figname='{}/hybrid_mask.png'.format(outdir))
                print 'Hybrid p, r, j: ', faster_compute_prj(hybrid_mask, gt_mask)

                visualize_test_gt_vision_overlay(batch, objid, k, hybrid_mask, outdir)
                print '----------------------------------------------------------------------------------'


if __name__ == '__main__':
    testing_viz_test_gt_vision_overlay()

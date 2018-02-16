from utils import CURR_DIR, DATA_DIR, VISION_DIR, VISION_TILES_DIR, \
    get_pixtiles, get_gt_mask, get_mega_mask, get_MV_mask, show_mask
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import os.path


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


def create_and_store_hybrid_masks(sample_name, objid, expand_thresh=0.8, contract_thresh=0.2):
    agg_vision_mask, _ = get_pixtiles(objid)

    # MV hybrid
    MV_mask = get_MV_mask(sample_name, objid)

    outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)

    mv_hybrid_mask = compute_hybrid_mask(MV_mask, agg_vision_mask, expand_thresh=expand_thresh, contract_thresh=contract_thresh, objid=objid)
    with open('{}MV_hybrid_mask.pkl'.format(outdir), 'w') as fp:
        fp.write(pickle.dumps(mv_hybrid_mask))

    sum_mask = mv_hybrid_mask.astype(int) * 5 + MV_mask.astype(int) * 20 + get_gt(objid).astype(int) * 50

    plt.figure()
    plt.imshow(sum_mask, interpolation="none")  # , cmap="rainbow")
    plt.colorbar()
    plt.savefig('{}MV_hybrid_mask.png'.format(outdir))
    plt.close()

    [p, r] = get_precision_and_recall(mv_hybrid_mask, get_gt_mask(objid))
    with open('{}MV_hybrid_pr.json'.format(outdir), 'w') as fp:
        fp.write(json.dumps([p, r]))


def create_and_store_vision_plus_gt_baseline(objid, include_thresh=0.5):
    agg_vision_mask, _ = get_pixtiles(objid)

    # MV hybrid
    gt_mask = get_gt_mask(objid)

    outdir = '{}obj{}/'.format(PIXEL_EM_DIR, objid)

    vision_only_mask = compute_hybrid_mask(gt_mask, agg_vision_mask, expand_thresh=include_thresh, contract_thresh=0, vision_only=True)
    with open('{}vision_with_gt_mask.pkl'.format(outdir), 'w') as fp:
        fp.write(pickle.dumps(vision_only_mask))

    sum_mask = vision_only_mask.astype(int) * 5 + gt_mask.astype(int) * 10

    plt.figure()
    plt.imshow(sum_mask, interpolation="none")  # , cmap="rainbow")
    plt.colorbar()
    plt.savefig('{}vision{}.png'.format(outdir, int(100*include_thresh)))
    plt.close()

    [p, r] = get_precision_and_recall(vision_only_mask, gt_mask)
    with open('{}vision{}_pr.json'.format(outdir, int(100*include_thresh)), 'w') as fp:
        fp.write(json.dumps([p, r]))


def overlay_viz_tiles(vision_mask, vision_tiles, tiles_missed, tiles_extra, tiles_correct, all_worker_tiles, figpath=None):
    missed_mask = tiles_to_mask(tiles_missed, all_worker_tiles, vision_mask)
    extra_mask = tiles_to_mask(tiles_extra, all_worker_tiles, vision_mask)
    correct_mask = tiles_to_mask(tiles_correct, all_worker_tiles, vision_mask)

    intersecting_viz_tiles = set()
    min_x = 10000
    max_x = 0
    min_y = 10000
    max_y = 0

    # num_satisfied = 0
    # num_not = 0
    for tid in range(len(vision_tiles)):
        for pix in vision_tiles[tid]:
            if missed_mask[pix] or extra_mask[pix] or correct_mask[pix]:
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

    print len(vision_tiles), len(intersecting_viz_tiles)

    x_range = max_x - min_x + 1
    y_range = max_y - min_y + 1

    print '[{},{}], [{},{}]'.format(min_x, max_x, min_y, max_y)
    zoomed_viz_mask = np.zeros((x_range, y_range))
    zoomed_test_extra = np.zeros((x_range, y_range))
    zoomed_test_missed = np.zeros((x_range, y_range))
    zoomed_test_correct = np.zeros((x_range, y_range))
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            zoomed_viz_mask[x - min_x][y - min_y] = vision_mask[x][y]
            zoomed_test_extra[x - min_x][y - min_y] = 1 if extra_mask[x][y] else 0
            zoomed_test_missed[x - min_x][y - min_y] = 1 if missed_mask[x][y] else 0
            zoomed_test_correct[x - min_x][y - min_y] = 1 if correct_mask[x][y] else 0

    import matplotlib as mpl
    ax = plt.gca()
    ax.imshow(zoomed_viz_mask, cmap="gray")
    ax.imshow(zoomed_test_correct, cmap='Greens', alpha=0.3)
    ax.imshow(zoomed_test_extra, cmap='Blues', alpha=0.3)
    ax.imshow(zoomed_test_missed, cmap='Reds', alpha=0.3)
    plt.draw()
    if figpath is not None:
        plt.savefig('{}/viz_comparison_to_gt_aware_overlay.png'.format(figpath))
    else:
        plt.show()
    plt.close()


def overlay_viz_test_gt_masks(vision_mask, vision_tiles, test_mask, gt_mask, mega_mask, hybrid_mask, figpath=None):
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

    print len(vision_tiles), len(intersecting_viz_tiles)

    x_range = max_x - min_x + 1
    y_range = max_y - min_y + 1

    print '[{},{}], [{},{}]'.format(min_x, max_x, min_y, max_y)
    zoomed_viz_mask = np.zeros((x_range, y_range))
    zoomed_test_mask = np.zeros((x_range, y_range))
    zoomed_gt_mask = np.zeros((x_range, y_range))
    zoomed_mega_mask = np.zeros((x_range, y_range))
    zoomed_hybrid_mask = np.zeros((x_range, y_range))
    zoomed_test_gt_overlap = np.zeros((x_range, y_range))
    zoomed_test_extra = np.zeros((x_range, y_range))
    zoomed_test_missed = np.zeros((x_range, y_range))
    zoomed_test_correct = np.zeros((x_range, y_range))
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            zoomed_viz_mask[x - min_x][y - min_y] = vision_mask[x][y]
            zoomed_test_mask[x - min_x][y - min_y] = test_mask[x][y]
            zoomed_hybrid_mask[x - min_x][y - min_y] = hybrid_mask[x][y]
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

    # show_mask(test_mask)
    # show_mask(zoomed_test_mask)

    import matplotlib as mpl
    ax = plt.gca()
    ax.imshow(zoomed_viz_mask, cmap="gray")
    # ax.imshow(zoomed_viz_mask, cmap="gray")
    # ax.imshow(zoomed_test_mask, cmap='gray', alpha=0.25)
    # ax.imshow(zoomed_test_mask, cmap='Reds', alpha=0.5)
    # ax.imshow(zoomed_gt_mask, cmap='gray', alpha=0.5)
    # ax.imshow(zoomed_test_gt_overlap, cmap='Greens', alpha=0.2)
    ax.imshow(zoomed_test_correct, cmap='Greens', alpha=0.3)
    ax.imshow(zoomed_test_extra, cmap='Blues', alpha=0.3)
    ax.imshow(zoomed_test_missed, cmap='Reds', alpha=0.3)
    # for i in range(len(zoomed_test_mask)):
    #     for j in range(len(zoomed_test_mask[i])):
    #         if zoomed_test_mask[i][j]:
    #             # print i, j, mv_mask[i][j]
    #             ax.add_patch(mpl.patches.Rectangle((j-.5, i-.5), 1, 1, hatch='/', fill=False, snap=False))
    #             # plt.plot([i], [j], 'x')
    plt.draw()
    if figpath is not None:
        plt.savefig('{}/viz_mv_gt_overlay.png'.format(figpath))
    else:
        plt.show()
    plt.close()

    if figpath is not None:
        show_mask(zoomed_test_mask, figname='{}/test_mask.png'.format(figpath))
        show_mask(zoomed_hybrid_mask, figname='{}/hybrid_mask.png'.format(figpath))
        show_mask(zoomed_viz_mask, figname='{}/vision_mask.png'.format(figpath))
        show_mask(zoomed_mega_mask, figname='{}/mega_mask.png'.format(figpath))

        ax = plt.gca()
        ax.imshow(zoomed_test_correct, cmap='Greens', alpha=0.5)
        ax.imshow(zoomed_test_extra, cmap='Blues', alpha=0.3)
        ax.imshow(zoomed_test_missed, cmap='Reds', alpha=0.3)
        plt.draw()
        plt.savefig('{}/mv_gt_overlay.png'.format(figpath))
        plt.close()

    return zoomed_viz_mask, zoomed_test_mask, zoomed_gt_mask, zoomed_mega_mask


def tiles_to_mask(tile_id_list, tile_to_pix_dict, base_mask):
    # given a list of chosen tids, converts that to a pixel mask of all chosen pixels
    tile_mask = np.zeros_like(base_mask)
    for tid in tile_id_list:
        for pix in tile_to_pix_dict[tid]:
            tile_mask[pix] = 1
    return tile_mask


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


def process_all_worker_tiles(all_worker_tiles, worker_mega_mask, gt_mask):
    print 'Processing all worker tiles...'
    print 'Num tiles: ', len(all_worker_tiles)
    worker_tiles_mask = np.zeros_like(worker_mega_mask)
    num_votes_mask = np.zeros_like(worker_mega_mask)
    num_votes = defaultdict(int)
    tile_area = defaultdict(int)
    tile_int_area = defaultdict(int)

    pix_frequency = defaultdict(int)

    num_pixs_total = 0
    for tid in range(len(all_worker_tiles)):
        int_area = 0
        num_votes[tid] = worker_mega_mask[next(iter(all_worker_tiles[tid]))]
        tile_area[tid] = len(all_worker_tiles[tid])
        for pix in all_worker_tiles[tid]:
            worker_tiles_mask[pix] = tid
            num_votes_mask[pix] += worker_mega_mask[pix]
            if worker_mega_mask[pix] != num_votes[tid]:
                print 'Vote mismatch, on pix coord {} (votes {}), on tile ID {} (votes {})'.format(
                    pix, worker_mega_mask[pix], tid, num_votes[tid])
            pix_frequency[pix] += 1
            num_pixs_total += 1
            if gt_mask[pix]:
                int_area += 1
        tile_int_area[tid] = int_area

    print 'Max number of tiles any pixel belongs to: ', max(pix_frequency.values())
    print 'Height x width of gt_mask = {}, num pixs accounted for: {}'.format(
        len(gt_mask)*len(gt_mask[0]), num_pixs_total)
    return worker_tiles_mask, num_votes, num_votes_mask, tile_area, tile_int_area


def inspect_object_pixel(batch='5workers_rand0', objid=1):
    # gt_aware: 0.890421868345, mv_weighted: 0.148028872647, for (batch x object): (5workers_rand0 x 29)
    # gt_aware: 0.916654994283, mv_weighted: 0.637160121509, for (batch x object): (5workers_rand0 x 37)
    # gt_aware: 0.928683856769, mv_weighted: 0.520829358357, for (batch x object): (15workers_rand0 x 7)

    batch_path = '{}/{}/{}'.format(BASE_DIR, 'pixel_em', batch)
    num_workers = int(batch.split('workers')[0])
    all_worker_tiles = pickle.load(open('{}/obj{}/tiles.pkl'.format(batch_path, objid)))
    # all_worker_tiles = get_tiles_from_mega_mask(batch, objid)
    vision_mask, vision_tiles = get_pixtiles(objid)
    gt_mask = get_gt_mask(objid)
    worker_mega_mask = get_mega_mask(batch, objid)
    mv_mask = get_MV_mask(batch, objid)

    worker_tiles_mask, num_votes, num_votes_mask, tile_area, tile_int_area = process_all_worker_tiles(
        all_worker_tiles, worker_mega_mask, gt_mask)

    # overlay_viz_test_gt_masks(vision_mask, vision_tiles, mv_mask, gt_mask)
    # return

    visualize_stuff = False
    if visualize_stuff:
        show_mask(worker_tiles_mask)
        show_mask(num_votes_mask)
        show_mask(worker_mega_mask)
        show_mask(mv_mask)
        show_mask(gt_mask)
        show_mask(vision_mask)

    # output from various greedy jaccard runs
    sorted_order_tids = dict()
    tile_added = dict()
    est_jacc_list = dict()

    # gt_aware greedy jaccard
    sorted_order_tids['gt_aware'], tile_added['gt_aware'], est_jacc_list['gt_aware'] = run_greedy_jaccard(tile_area, tile_int_area)
    print 'gt_aware jaccard: ', est_jacc_list['gt_aware'][-1]

    # MV
    print 'mv jaccard: ', jaccard(mv_mask, gt_mask)
    mv_int_area = {
        tid: (tile_area[tid] if num_votes[tid] > (num_workers / 2) else 0)
        for tid in tile_area
    }
    # print 'num votes 516: ', num_votes[516]
    # print 'int area of 516: ', mv_int_area[516.0]
    # for tid in tile_int_area:
    #     print tile_int_area[tid], num_votes[tid], mv_int_area[tid]
    sorted_order_tids['mv'], tile_added['mv'], est_jacc_list['mv'] = run_greedy_jaccard(tile_area, mv_int_area)
    mv_tile_ids_chosen = [tid for tid in tile_added['mv'].keys() if tile_added['mv'][tid]]
    calc_mv_mask = tiles_to_mask(mv_tile_ids_chosen, all_worker_tiles, gt_mask)

    for x in range(len(mv_mask)):
        for y in range(len(mv_mask[x])):
            if mv_mask[x][y] != calc_mv_mask[x][y]:
                print 'tid:', worker_tiles_mask[x][y]
                print num_votes[worker_tiles_mask[x][y]], num_votes_mask[x][y], worker_mega_mask[x][y], calc_mv_mask[x][y], mv_mask[x][y]
    print est_jacc_list['mv'][-1], jaccard(calc_mv_mask, gt_mask)
    # hybrid_mask = compute_hybrid_mask(mv_mask, vision_mask, expand_thresh=0.8, contract_thresh=0.2, objid=objid, DEBUG=True)
    # print jaccard(hybrid_mask, gt_mask)


def visualizing_stuff(batch='5workers_rand0', test=False):
    num_workers = int(batch.split('workers')[0])

    for objid in range(1, 48):
        print 'Visualizing stuff for obj ', objid
        outdir = VISION_DIR + 'visualizing_and_testing_stuff/obj{}'.format(objid)
        if test:
            outdir += '/test/'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # all_worker_tiles = pickle.load(open('{}/obj{}/tiles.pkl'.format(batch_path, objid)))
        vision_mask, vision_tiles = get_pixtiles(objid, test)
        gt_mask = get_gt_mask(objid)
        worker_mega_mask = get_mega_mask(batch, objid)
        mv_mask = get_MV_mask(batch, objid)

        # worker_tiles_mask, num_votes, tile_area, tile_int_area = process_all_worker_tiles(
        #     all_worker_tiles, worker_mega_mask, gt_mask)

        print 'Computing simple hybrid mask...'
        print 'MV jaccard: ', jaccard(mv_mask, gt_mask)
        all_voted_mask = np.zeros_like(worker_mega_mask)
        all_voted_mask[np.where(worker_mega_mask == num_workers)] = 1
        all_mv_but_low_confidence = mv_mask - all_voted_mask
        hybrid_mask = all_voted_mask + compute_hybrid_mask(all_mv_but_low_confidence, vision_mask, expand_thresh=0.8, contract_thresh=0.2, objid=objid, DEBUG=False)
        hybrid_mask[np.where(hybrid_mask > 1)] = 1
        # show_mask(hybrid_mask, figname='{}/hybrid_mask.png'.format(outdir))
        print 'Hybrid jaccard: ', jaccard(hybrid_mask, gt_mask)

        overlay_viz_test_gt_masks(vision_mask, vision_tiles, mv_mask, gt_mask, worker_mega_mask, hybrid_mask, figpath=outdir)
        print '----------------------------------------------------------------------------------'


def run_greedy_jaccard(tile_area, tile_int_area):
    # input: tile_area[tid], tile_int_area[tid]
    # output: chosen_tiles = [tid], est_jacc = xx
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
        # print 'IA: {}, OA: {}'.format(IA, OA)
        # print curr_jacc, new_jacc_if_added
        if new_jacc_if_added >= curr_jacc:
            # add tile if it improves jaccard
            tile_added[tid] = True
            curr_total_int_area += IA
            curr_total_out_area += OA
            curr_jacc = new_jacc_if_added
        est_jacc_list.append(curr_jacc)

    return sorted_order_tids, tile_added, est_jacc_list



def testing_diff_greedy_jaccs(batch, objid):
    num_workers = int(batch.split('worker')[0])
    all_worker_tiles = get_tiles_from_mega_mask(batch, objid)
    # all_worker_tiles = pickle.load(open('{}/obj{}/tiles.pkl'.format(batch_path, objid)))
    gt_mask = get_gt_mask(objid)
    worker_mega_mask = get_mega_mask(batch, objid)
    mv_mask = get_MV_mask(batch, objid)

    worker_tiles_mask, num_votes, num_votes_mask, tile_area, tile_int_area = process_all_worker_tiles(
        all_worker_tiles, worker_mega_mask, gt_mask)

    # output from various greedy jaccard runs
    sorted_order_tids = dict()
    tile_added = dict()
    est_jacc_list = dict()

    # gt_aware greedy jaccard
    sorted_order_tids['gt_aware'], tile_added['gt_aware'], est_jacc_list['gt_aware'] = run_greedy_jaccard(tile_area, tile_int_area)
    print 'gt_aware jaccard: ', est_jacc_list['gt_aware'][-1]

    # MV
    print 'mv jaccard: ', jaccard(mv_mask, gt_mask)

    # vote weighted
    v_wtd_int_area = {
        tid: (float(tile_area[tid]) * (float(num_votes[tid]) / float(num_workers)))
        for tid in tile_area
    }
    # for tid in tile_int_area:
    #     print tile_int_area[tid], num_votes[tid], mv_int_area[tid]
    sorted_order_tids['v_wtd'], tile_added['v_wtd'], est_jacc_list['v_wtd'] = run_greedy_jaccard(tile_area, v_wtd_int_area)
    mv_tile_ids_chosen = [tid for tid in tile_added['v_wtd'].keys() if tile_added['v_wtd'][tid]]
    calc_mask = tiles_to_mask(mv_tile_ids_chosen, all_worker_tiles, gt_mask)
    print 'v_wtd jaccard: ', jaccard(calc_mask, gt_mask)

    # vision_mask, vision_tiles = get_pixtiles(objid)
    # all_voted_mask = np.zeros_like(worker_mega_mask)
    # all_voted_mask[np.where(worker_mega_mask == num_workers)] = 1
    # all_mv_but_low_confidence = mv_mask - all_voted_mask
    # hybrid_mask = all_voted_mask + compute_hybrid_mask(all_mv_but_low_confidence, vision_mask, expand_thresh=0.8, contract_thresh=0.2, objid=objid, DEBUG=False)
    # hybrid_mask[np.where(hybrid_mask > 1)] = 1
    # # show_mask(hybrid_mask, figname='{}/hybrid_mask.png'.format(outdir))
    # print 'Hybrid jaccard: ', jaccard(hybrid_mask, gt_mask)

    # overlay_viz_test_gt_masks(vision_mask, vision_tiles, mv_mask, gt_mask, worker_mega_mask, hybrid_mask, figpath=outdir)


def improving_num_votes_jacc(batch, objid, test=False):
    # MV and votes_wtd
    num_workers = int(batch.split('worker')[0])
    # all_worker_tiles = get_tiles_from_mega_mask(batch, objid)
    batch_path = '{}/{}/{}'.format(BASE_DIR, 'pixel_em', batch)
    all_worker_tiles = pickle.load(open('{}/obj{}/tiles.pkl'.format(batch_path, objid)))
    gt_mask = get_gt_mask(objid)
    worker_mega_mask = get_mega_mask(batch, objid)
    mv_mask = get_MV_mask(batch, objid)

    worker_tiles_mask, num_votes, num_votes_mask, tile_area, tile_int_area = process_all_worker_tiles(
        all_worker_tiles, worker_mega_mask, gt_mask)

    # output from various greedy jaccard runs
    sorted_order_tids = dict()
    tile_added = dict()
    est_jacc_list = dict()

    # gt_aware greedy jaccard
    sorted_order_tids['gt_aware'], tile_added['gt_aware'], est_jacc_list['gt_aware'] = run_greedy_jaccard(tile_area, tile_int_area)
    print 'gt_aware jaccard: ', est_jacc_list['gt_aware'][-1]

    # MV
    print 'mv jaccard: ', jaccard(mv_mask, gt_mask)

    # vote weighted
    v_wtd_int_area = {
        tid: (float(tile_area[tid]) * (float(num_votes[tid]) / float(num_workers)))
        for tid in tile_area
    }
    # for tid in tile_int_area:
    #     print tile_int_area[tid], num_votes[tid], mv_int_area[tid]
    sorted_order_tids['v_wtd'], tile_added['v_wtd'], est_jacc_list['v_wtd'] = run_greedy_jaccard(tile_area, v_wtd_int_area)
    mv_tile_ids_chosen = [tid for tid in tile_added['v_wtd'].keys() if tile_added['v_wtd'][tid]]
    calc_mask = tiles_to_mask(mv_tile_ids_chosen, all_worker_tiles, gt_mask)
    print 'v_wtd jaccard: ', jaccard(calc_mask, gt_mask)

    vision_mask, vision_tiles = get_pixtiles(objid, test)
    # all_voted_mask = np.zeros_like(worker_mega_mask)
    # all_voted_mask[np.where(worker_mega_mask == num_workers)] = 1
    # all_mv_but_low_confidence = mv_mask - all_voted_mask
    # hybrid_mask = all_voted_mask + compute_hybrid_mask(all_mv_but_low_confidence, vision_mask, expand_thresh=0.8, contract_thresh=0.2, objid=objid, DEBUG=False)
    # hybrid_mask[np.where(hybrid_mask > 1)] = 1
    # show_mask(hybrid_mask, figname='{}/hybrid_mask.png'.format(outdir))
    # print 'Hybrid jaccard: ', jaccard(hybrid_mask, gt_mask)

    # visualizing mistakes as compared to gt_aware
    tiles_missed = list(set(tile_added['gt_aware'].keys()) - set(tile_added['v_wtd'].keys()))
    tiles_extra = list(set(tile_added['v_wtd'].keys()) - set(tile_added['gt_aware'].keys()))
    tiles_correct = list(set(tile_added['v_wtd'].keys()) & set(tile_added['gt_aware'].keys()))
    outdir = VISION_DIR + 'visualizing_and_testing_stuff/obj{}'.format(objid)
    if test:
        outdir += '/test/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    overlay_viz_tiles(vision_mask, vision_tiles, tiles_missed, tiles_extra, tiles_correct, all_worker_tiles, figpath=outdir)

    # overlay_viz_test_gt_masks(vision_mask, vision_tiles, mv_mask, gt_mask, worker_mega_mask, hybrid_mask, figpath=outdir)


if __name__ == '__main__':
    visualizing_stuff(batch='5workers_rand0', test=False)
    # inspect_object_pixel()
    # for objid in range(1, 48):
    #     print '---------------------------------------'
    #     print 'Testing different greedy jaccards for obj', objid
    #     # testing_diff_greedy_jaccs('5workers_rand0', objid)
    #     # testing_diff_greedy_jaccs('15worker_rand2', objid)
    #     improving_num_votes_jacc('5workers_rand0', objid, test=True)

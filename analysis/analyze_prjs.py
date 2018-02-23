from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


samples_for_num = {
    5: range(10),
    10: range(8),
    15: range(6),
    20: range(4),
    25: range(2),
    30: range(1)
}


def read_algo_prj_table(algo):
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    filename = 'pixel_em/{}_full_PRJ_table.csv'.format(algo)
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            nworkers = int(float(row['num_workers']))
            sample_num = int(float(row['sample_num']))
            objid = int(float(row['objid']))
            clust = int(float(row['clust']))
            # print nworkers, sample_num, objid, clust
            all_data[nworkers][sample_num][objid][clust] = row
    return all_data


def read_best_clust():
    best_clust = defaultdict(lambda: defaultdict(dict))  # best_clust[nworkers][sample_num][objid] = clust
    with open('best_clust_picking.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            nworkers = int(row['num_workers'])
            sample_num = int(row['sample_num'])
            objid = int(row['objid'])
            clust = int(row['clust'])
            # print nworkers, sample_num, objid, clust
            best_clust[nworkers][sample_num][objid] = clust
    return best_clust


def sanity_check(algo):
    all_data = read_algo_prj_table(algo)
    best_clust = read_best_clust()
    obj_missing_noclust = []
    obj_with_noclust_best = []
    num_total = 0
    for nworkers in all_data.keys():
        for sample_num in all_data[nworkers]:
            for objid in all_data[nworkers][sample_num]:
                if -1 not in all_data[nworkers][sample_num][objid].keys():
                    # print '({}, {}, {})'.format(nworkers, sample_num, objid)
                    obj_missing_noclust.append((nworkers, sample_num, objid))
                if best_clust[nworkers][sample_num][objid] == -1:
                    obj_with_noclust_best.append((nworkers, sample_num, objid))
                num_total += 1
    print 'num (sample x obj) missing clust = -1 in {}: {}'.format(algo, len(obj_missing_noclust))
    print 'num (sample x obj) with best_clust = -1 in {}: {}'.format(algo, len(obj_with_noclust_best))
    print 'num total in {}: {}'.format(algo, num_total)


def clust_vs_noclust(algo='MV',metric="jaccard", filtered=False, PLOT=False):
    all_data = read_algo_prj_table(algo)
    best_clust = read_best_clust()
    jacc_noclust = defaultdict(list)  # jacc[nworkers] = []
    jacc_bestclust = defaultdict(list)
    for nworkers in all_data.keys():
        for sample_num in all_data[nworkers]:
            for objid in all_data[nworkers][sample_num]:
                # print nworkers, sample_num, objid
                best = best_clust[nworkers][sample_num][objid]
                if filtered and (best == -1 or -1 not in all_data[nworkers][sample_num][objid].keys()):
                    continue
                # TODO: handle metric==-1 or None?
                noclust_MV = float(all_data[nworkers][sample_num][objid][-1][metric])
                jacc_noclust[nworkers].append(noclust_MV)
                bestclust_MV = float(all_data[nworkers][sample_num][objid][best][metric])
                jacc_bestclust[nworkers].append(bestclust_MV)

    x = []
    y_noclust = []
    y_bestclust = []
    for nworkers in jacc_noclust.keys():
        x.append(nworkers)
        # print jacc_noclust[nworkers]
        # print jacc_bestclust[nworkers]
        y_noclust.append(np.mean(jacc_noclust[nworkers]))
        y_bestclust.append(np.mean(jacc_bestclust[nworkers]))

    assert set(x) == set([5, 10, 15, 20, 25, 30])

    if PLOT:
        if not os.path.isdir('temp_plots'):
            os.makedirs('temp_plots')
        plt.figure()
        plt.title('{}_{}_noclust_vs_bestclust'.format(algo, 'filtered' if filtered else 'unfiltered'))
        plt.plot(x, y_noclust, color='blue', label='noclust')
        plt.plot(x, y_bestclust, color='orange', label='bestclust')
        plt.legend()
        plt.savefig('{}/{}_{}_noclust_vs_bestclust.png'.format('temp_plots', algo, 'filtered' if filtered else 'unfiltered'))
        plt.close()

    return x, y_noclust, y_bestclust


if __name__ == '__main__':
    for algo in ['MV', 'GT', 'GTLSA', 'basic', 'isoGT', 'isoGTLSA', 'isobasic']:
        print '*****************************************************'
        print 'Algo: {}'.format(algo)
        print 'Running sanity checks...'
        sanity_check(algo)
        print '---------------------------------------------------'
        print 'Running clust vs no clust analysis'
        for filtered in [True, False]:
            print 'Filtered: {}'.format(filtered)
            x, y_noclust, y_bestclust = clust_vs_noclust(algo, filtered, PLOT=True)
            print 'x: {}'.format(x)
            print 'jacc_noclust: {}'.format(y_noclust)
            print 'jacc_bestclust: {}'.format(y_bestclust)

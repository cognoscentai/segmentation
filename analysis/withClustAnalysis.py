import json
import pandas as pd
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
metric_keys=[ u'P [GT]', u'R [GT]', u'J [GT]', u'P [isoGT]',
       u'R [isoGT]', u'J [isoGT]', u'P [GTLSA]', u'R [GTLSA]', u'J [GTLSA]',
       u'P [isoGTLSA]', u'R [isoGTLSA]', u'J [isoGTLSA]', u'P [basic]',
       u'R [basic]', u'J [basic]','P [MV]',u'R [MV]', u'J [MV]',
       'P [isobasic]','R [isobasic]', u'J [isobasic]']
metric_J= [u'J [MV]', u'J [GT]', u'J [isoGT]', u'J [GTLSA]', u'J [isoGTLSA]', u'J [basic]','J [isobasic]']
#include only the "whole" MV PRJ of objects that were unclustered
clust_df = pd.read_csv("spectral_clustering_all_hard_obj.csv")
noClust_obj =[obj for obj in object_lst if obj not in clust_df.objid.unique() ]
def compile_cluster_MV_prj_into_csv():
    # Cluster-based MV results 
    # check result by plotting: plt.plot(MV_df.groupby("num_workers")["MV_jaccard"].mean())
    MV = pd.read_csv("pixel_em/MV_PRJ_table.csv")
    MV["clust"]=-1
    #MV = MV[MV.objid.isin(noClust_obj)]
    # read the MV PRJ for the remaining clustered objects
    MV_clust = pd.read_csv("pixel_em/withClust_MV_PRJ_table.csv",index_col=0)
    MV_clust["num_workers"] = MV_clust["sample"].apply(lambda x: int(x.split("workers")[0]))
    MV_clust["sample_num"] = MV_clust["sample"].apply(lambda x: int(x.split("rand")[-1]))
    MV_clust.to_csv("pixel_em/withClust_MV_PRJ_table.csv")
    MV_clust = MV_clust.drop("sample",axis=1)
    # combine these to form a MV PRJ table for all clustered and unclustered PRJ (not containing the "whole" MV PRJ for clustered objects)
    MV_df = pd.concat([MV_clust,MV])
    MV_df.to_csv("pixel_em/all_MV_PRJ_table.csv")
    return MV_df
def compute_best_worker_picking():
    # pick the best MV performing cluster as the cluster to run
    # we store -1 for the rest of the unclustered objects 

    MV_clust = pd.read_csv("pixel_em/MV_full_PRJ_table.csv")
    #MV_clust["num_workers"] = MV_clust["sample"].apply(lambda x: int(x.split("workers")[0]))
    #MV_clust["sample_num"] = MV_clust["sample"].apply(lambda x: int(x.split("rand")[-1]))
    #clust_df = pd.read_csv("spectral_clustering_all_hard_obj.csv")

    # pick the cluster with the highest MV
    best_clust_df = MV_clust.loc[MV_clust.groupby(["num_workers","sample_num","objid"])["jaccard"].idxmax()]
    best_clust_df = best_clust_df.drop(['precision','recall','jaccard'],axis=1)
    best_clust_df = best_clust_df.rename(columns={'cluster':'clust'})
    # There can only be one best cluster for every sample objid
    assert int(best_clust_df.groupby(["num_workers","sample_num","objid"]).count()["clust"].unique())==1
    best_clust_df.to_csv("best_clust_picking.csv")
    return best_clust_df
def filter_best_clust(df,best_clust_df):
    # given a df to be filtered with clust_df (list of best clusters), 
    # pick rows based on whether it is in clust_df or not
    keys  = [u'objid', u'clust', u'num_workers', u'sample_num']
    i1 = df.set_index(keys).index
    i2 = best_clust_df.set_index(keys).index
    best_clust_no_thresh_df = df[((df.clust!=-1)&(i1.isin(i2))|((df.clust==-1) & (df.objid.isin(noClust_obj))))]
    #best_clust_no_thresh_df = df[(i1.isin(i2))|(df.clust==-1)]
    # #Check that initially, there was 2~3 clusters in df for every ["num_workers","objid","sample_num","thresh"]
    # df.groupby(["num_workers","objid","sample_num","thresh"]).count()
    # # Now there is only one cluster in the filtered df for every ["num_workers","objid","sample_num","thresh"]
    #best_clust_no_thresh_df.groupby(["num_workers","objid","sample_num","thresh"]).count()
    #assert int(best_clust_no_thresh_df.groupby(["num_workers","objid","sample_num","thresh"]).count()["J [GTLSA]"].unique())==1
    # # visually check example of a no Cluster object
    # best_clust_no_thresh_df[(best_clust_no_thresh_df["num_workers"]==30)&(best_clust_no_thresh_df["objid"]==2)]
    # # visually check an example of a cluster object 
    # best_clust_no_thresh_df[(best_clust_no_thresh_df["num_workers"]==30)&(best_clust_no_thresh_df["objid"]==47)]
    # # check that the chosen cluster coincide with the chosen cluster in best_clust_df
    # best_clust_df[(best_clust_df["sample"]=="30workers_rand0")&(best_clust_df["objid"]==47)]
    return best_clust_no_thresh_df 
def compile_all_algo_PRJs(filter_best =False):
    clustObj = clust_df.objid.unique()
    df = pd.read_csv("pixel_em/MV_full_PRJ_table.csv")
    if filter_best:
        best_clust_df = compute_best_worker_picking()
        df = filter_best_clust(df,best_clust_df)
    df = df.rename(columns={"precision":"P [MV]",
                           "recall":"R [MV]",
                           "jaccard":"J [MV]",
                           "FPR%":"FPR% [MV]",
                           "FNR%":"FNR% [MV]"})
    df= df[((df["clust"]==-1) &(df["objid"].isin(noClust_obj)))|((df["clust"]!=-1) & df["objid"].isin(clustObj))]
    #MV contains rows that have only 0 or 1 annotations per cluster, we did not ran the algos on this
    # the merge (inner join) removes these elements
    for mode in ["basic","GT","isoGT","GTLSA","isoGTLSA","isobasic"]:
        data =  pd.read_csv("pixel_em/{}_full_PRJ_table.csv".format(mode))
        data = data.rename(columns={"precision":"P [{}]".format(mode),
                               "recall":"R [{}]".format(mode),
                               "jaccard":"J [{}]".format(mode),
                               "FPR%":"FPR% [{}]".format(mode),
                               "FNR%":"FNR% [{}]".format(mode)})
        if filter_best: data = filter_best_clust(data,best_clust_df)
        df = df.merge(data,on=['clust', 'num_workers','actualNworkers', 'objid','sample_num'])#,how="outer")
    #assert pd.isnull(df).sum().sum()==0
    if filter_best:
        df.to_csv("pixel_em/all_PRJ_table_filter_best.csv")
    else:
        df.to_csv("pixel_em/all_PRJ_table.csv")
    return df
def plot_PRcurve(objid,num_worker,sample_num=0):
    objdf = df[(df["num_workers"]==num_worker)&(df["sample_num"]==sample_num)&(df["objid"]==objid)]
    plt.figure()
    for algo in ['basic','GT','isoGT','GTLSA','isoGTLSA']:
        x= objdf["P [{}]".format(algo)]
        y = objdf["R [{}]".format(algo)]
        if len(x)<=0:
            return
        sortedx, sortedy = zip(*sorted(zip(x, y)))
        plt.plot(sortedx,sortedy,'.-',label=algo)
    plt.xlabel("Precision",fontsize=13)
    plt.ylabel("Recall",fontsize=13)
    plt.legend(loc="bottom left")
    plt.title("{}worker_rand{} [obj {};N={}]".format(num_worker,sample_num,objid,len(objdf)))
######################################################################################################################
from greedyPicking import *
from glob import glob 
def compile_noClust_greedy_algos_to_csv():
    globfnames = glob("greedy_old_results/greedy_result_*.csv")
    globfnames.remove("greedy_old_results/greedy_result_worker_fraction.csv")
    greedy_df = pd.read_csv(globfnames[0],index_col=0)
    for fname in globfnames: 
        greedy_df = greedy_df.append(pd.read_csv(fname,index_col=0))

    #assert len(greedy_df)==31*44*5

    greedy_df.to_csv("greedy_old_results/all_greedy_result.csv")

    greedy_df = pd.read_csv("greedy_old_results/all_greedy_result.csv",index_col=0)

    ground_truth_greedy_df = pd.read_csv("ground_truth_greedy_result.csv")
    greedy_df = greedy_df.append(ground_truth_greedy_df)
    worker_frac_greedy_df = pd.read_csv("greedy_old_results/greedy_result_worker_fraction.csv")
    greedy_df = greedy_df.append(worker_frac_greedy_df)

    ereedy_df["num_workers"] = greedy_df["sample"].apply(lambda x: int(x.split("workers")[0]))
    return greedy_df 
def compile_withClust_greedy_algos_to_csv():
    globfnames = glob("withClust_greedy_result_*.csv")
    globfnames.remove('withClust_greedy_result_worker_fraction.csv')
    greedy_df = pd.read_csv(globfnames[0],index_col=0)
    for fname in globfnames: 
        greedy_df = greedy_df.append(pd.read_csv(fname,index_col=0))
    # adding the rest of the baselines (ground truth, worker fraction) into compiled greedy csv
    # ground truth greedy result is independent of clustering 
    ground_truth_greedy_df = pd.read_csv("ground_truth_greedy_result.csv")
    ground_truth_greedy_df["cluster_id"] = -1

    # Need to combine the worker fraction greedy result from no Clust and Clust together 
    worker_frac_greedy_df = pd.read_csv("withClust_greedy_result_worker_fraction.csv")
    noClust_worker_frac_greedy_df = pd.read_csv("greedy_old_results/greedy_result_worker_fraction.csv")
    noClust_worker_frac_greedy_df["cluster_id"]=-1
    # we only want to keep the no clustering objects to combine it with the clustered worker fraction objects
    noClust_worker_frac_greedy_df = noClust_worker_frac_greedy_df[noClust_worker_frac_greedy_df.objid.isin(noClust_obj)]
    # ensure that all objects are inside either noClust_worker_frac_greedy_df or worker_frac_greedy_df, not both or neither
    assert len(object_lst) == len(np.concatenate([worker_frac_greedy_df.objid.unique(),noClust_worker_frac_greedy_df.objid.unique()]))

    greedy_df=pd.concat([greedy_df,ground_truth_greedy_df,worker_frac_greedy_df,noClust_worker_frac_greedy_df])
    #greedy_df["num_workers"] = greedy_df["sample"].apply(lambda x: int(x.split("workers")[0]))
    #greedy_df["sample_num"] = greedy_df["sample"].apply(lambda x: int(x.split("rand")[-1]))
    # remaining greedy results from no cluster case for algo [basic,GT,isoGT,isoGTLSA]
    noClust_greedy = pd.read_csv("greedy_old_results/all_greedy_result.csv",index_col=0)
    noClust_greedy["cluster_id"]=-1
    noClust_greedy = noClust_greedy[noClust_greedy.objid.isin(noClust_obj)] #take only objects who is not clustered
    #noClust_greedy["num_workers"] = noClust_greedy["sample"].apply(lambda x: int(x.split("workers")[0]))
    #noClust_greedy["sample_num"] = noClust_greedy["sample"].apply(lambda x: int(x.split("rand")[-1]))
    # ensure that all objects are inside either noClust or clustered greedy, not both or neither
    assert len(object_lst) == len(np.concatenate([greedy_df[greedy_df["algo"]=="basic"].objid.unique(),noClust_greedy.objid.unique()]))
    greedy_df = pd.concat([greedy_df,noClust_greedy])
    greedy_df = greedy_df.rename(columns={"cluster_id":"clust"})

    greedy_df["num_workers"] = greedy_df["sample"].apply(lambda x: int(x.split("workers")[0]))
    greedy_df["sample_num"] = greedy_df["sample"].apply(lambda x: int(x.split("rand")[-1]))
    greedy_df.to_csv("withClust_all_greedy_result.csv")
    return greedy_df

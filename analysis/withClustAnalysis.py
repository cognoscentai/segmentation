import json
import pandas as pd
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
metric_keys=[u'P [MV]',u'R [MV]', u'J [MV]', u'P [GT]', u'R [GT]', u'J [GT]', u'P [isoGT]',
       u'R [isoGT]', u'J [isoGT]', u'P [GTLSA]', u'R [GTLSA]', u'J [GTLSA]',
       u'P [isoGTLSA]', u'R [isoGTLSA]', u'J [isoGTLSA]', u'P [basic]',
       u'R [basic]', u'J [basic]']
metric_J= [u'J [MV]', u'J [GT]', u'J [isoGT]', u'J [GTLSA]', u'J [isoGTLSA]', u'J [basic]']
#include only the "whole" MV PRJ of objects that were unclustered
clust_df = pd.read_csv("spectral_clustering_all_hard_obj.csv")
noClust_obj =[obj for obj in object_lst if obj not in clust_df.objid.unique() ]
def compile_cluster_MV_prj_into_csv():
    # Cluster-based MV results 
    # check result by plotting: plt.plot(MV_df.groupby("num_workers")["MV_jaccard"].mean())
    MV = pd.read_csv("pixel_em/MV_PRJ_table.csv")
    MV["clust"]=-1
    MV = MV[MV.objid.isin(noClust_obj)]
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

def best_worker_picking():
    MV_clust = pd.read_csv("pixel_em/withClust_MV_PRJ_table.csv",index_col=0)
    clust_df = pd.read_csv("spectral_clustering_all_hard_obj.csv")
    #pick the cluster with the highest count
    # clust_df["count"]=clust_df.groupby(['objid','cluster']).transform("count")
    # clust_df=clust_df.drop('wid',axis=1)
    # best_clust_df = clust_df.loc[clust_df.groupby(['objid'])["count"].idxmax()]
    # best_clust_df = best_clust_df.drop(["count"],axis=1)

    # pick the cluster with the highest MV
    best_clust_df = MV_clust.loc[MV_clust.groupby(["sample_num","num_workers","objid"])["MV_jaccard"].idxmax()]
    best_clust_df = best_clust_df.drop(['MV_precision','MV_recall','MV_jaccard'],axis=1)

    # Note this does not account for the list of objects that have not been clustered
    # noClust_obj =[obj for obj in object_lst if obj not in clust_df.objid.unique() ]
    # noClust_df = pd.DataFrame(zip(noClust_obj,list(-1*np.ones_like(noClust_obj))),columns=["objid","clust"])
    # clust_df = pd.concat([best_clust_df,noClust_df])
    best_clust_df = best_clust_df.rename(columns={'cluster':'clust'})
    #best_clust contains the best cluster for each object sample. There should be no best clusters that is -1, since that is the unclustered case. 
    assert len(best_clust_df[best_clust_df["clust"]==-1])==0
    # There can only be one best cluster for every sample objid 
    assert int(best_clust_df.groupby(["sample","objid"]).count()["clust"].unique())==1 
    return best_clust_df
def compile_all_algo_PRJs():
    df = pd.read_csv("pixel_em/all_MV_PRJ_table.csv")
    df = df.rename(columns={"MV_precision":"P [MV]",
                           "MV_recall":"R [MV]",
                           "MV_jaccard":"J [MV]"})
    for mode in  ["GT","isoGT","GTLSA","isoGTLSA","basic"]:
        data =  pd.read_csv("pixel_em/{}_ground_truth_full_PRJ_table.csv".format(mode))
        data = data.rename(columns={"EM_precision":"P [{}]".format(mode),
                           "EM_recall":"R [{}]".format(mode),
                           "EM_jaccard":"J [{}]".format(mode),})
        df = df.merge(data)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.to_csv("pixel_em/all_PRJ_table.csv")
    return df
def compile_best_thresh_all_algo_PRJs():
    #using base tables create one algo for each row
    df = pd.read_csv("pixel_em/all_MV_PRJ_table.csv",index_col=0)
    df["algo"]='MV'
    df["thresh"]=0 #dummy value
    df = df.rename(columns={"MV_precision":"p",
                           "MV_recall":"r",
                           "MV_jaccard":"j"})
    for mode in  ["GT","isoGT","GTLSA","isoGTLSA","basic"]:
        data =  pd.read_csv("pixel_em/{}_ground_truth_full_PRJ_table.csv".format(mode))
        data["algo"]=mode
        data = data.rename(columns={"EM_precision":"p",
                           "EM_recall":"r",
                           "EM_jaccard":"j"})
        df = pd.concat([df,data]) 
    # get entry with the best threshold jaccard
    df_best_thresh = df.sort("j",ascending=False).groupby(["sample_num","num_workers","objid","clust","algo"], as_index=False).first()

    #df_best_thresh = df.loc[df.groupby(["sample_num","num_workers","objid","clust","algo"])["j"].idxmax()]

    #check that for some sample, the number of best threshold values is the same as the number of objects 
    assert len(df_best_thresh[(df_best_thresh["num_workers"]==5)&(df_best_thresh["sample_num"]==1)].objid.unique())==len(object_lst)
    # Check: in the unfiltered case there should be a whole array of thresholds (x5)
    #df[(df["num_workers"]==25)&(df["sample_num"]==1)&(df["objid"]==1)&(df["clust"]==0)&(df["algo"]=="isoGTLSA")]
    # visually check that only one best jaccard result gets returned
    assert len(df_best_thresh[(df_best_thresh["num_workers"]==25)&(df_best_thresh["sample_num"]==1)&(df_best_thresh["objid"]==1)&(df_best_thresh["clust"]==0)&(df_best_thresh["algo"]=="isoGTLSA")])==1
    # removing the -1 clusters for objects that are clustered
    df_best_thresh = df_best_thresh[(df_best_thresh.objid.isin(noClust_obj))|(df_best_thresh.clust!=-1)]
    # visually inspect that -1 clusters do not exist for objects that are clustered
    # df_best_thresh[(df_best_thresh["num_workers"]==30)&(df_best_thresh["objid"]==1)]
    # df[(df["num_workers"]==30)&(df["objid"]==1)]
    # best_clust_no_thresh_df[(best_clust_no_thresh_df["num_workers"]==30)&(best_clust_no_thresh_df["objid"]==7)]
    # best_clust_best_thresh_df[(best_clust_best_thresh_df["num_workers"]==30)&(best_clust_best_thresh_df["objid"]==7)]
    return df_best_thresh
def filter_best_clust(df,best_clust_df):
    # given a df to be filtered with clust_df (list of best clusters), 
    # pick rows based on whether it is in clust_df or not
    keys  = [u'objid', u'clust', u'num_workers', u'sample_num']
    i1 = df.set_index(keys).index
    i2 = best_clust_df.set_index(keys).index
    best_clust_no_thresh_df = df[(i1.isin(i2))|(df.clust==-1)]
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
    for fname in globfnames[1:]: 
        greedy_df = greedy_df.append(pd.read_csv(fname,index_col=0))

    assert len(greedy_df)==31*44*5

    greedy_df.to_csv("greedy_old_results/all_greedy_result.csv")

    greedy_df = pd.read_csv("greedy_old_results/all_greedy_result.csv",index_col=0)

    ground_truth_greedy_df = pd.read_csv("ground_truth_greedy_result.csv")
    greedy_df = greedy_df.append(ground_truth_greedy_df)
    worker_frac_greedy_df = pd.read_csv("greedy_old_results/greedy_result_worker_fraction.csv")
    greedy_df = greedy_df.append(worker_frac_greedy_df)

    greedy_df["num_workers"] = greedy_df["sample"].apply(lambda x: int(x.split("workers")[0]))
    return greedy_df 
def compile_withClust_greedy_algos_to_csv():
    globfnames = glob("withClust_greedy_result_*.csv")
    globfnames.remove('withClust_greedy_result_worker_fraction.csv')
    greedy_df = pd.read_csv(globfnames[0],index_col=0)
    for fname in globfnames[1:]: 
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
    greedy_df["num_workers"] = greedy_df["sample"].apply(lambda x: int(x.split("workers")[0]))

    # remaining greedy results from no cluster case for algo [basic,GT,isoGT,isoGTLSA]
    noClust_greedy = pd.read_csv("greedy_old_results/all_greedy_result.csv",index_col=0)
    noClust_greedy["cluster_id"]=-1
    noClust_greedy = noClust_greedy[noClust_greedy.objid.isin(noClust_obj)] #take only objects who is not clustered
    noClust_greedy["num_workers"] = noClust_greedy["sample"].apply(lambda x: int(x.split("workers")[0]))
    noClust_greedy["sample_num"] = noClust_greedy["sample"].apply(lambda x: int(x.split("rand")[-1]))
    # ensure that all objects are inside either noClust or clustered greedy, not both or neither
    assert len(object_lst) == len(np.concatenate([greedy_df[greedy_df["algo"]=="basic"].objid.unique(),noClust_greedy.objid.unique()]))
    greedy_df = pd.concat([greedy_df,noClust_greedy])
    greedy_df = greedy_df.rename(columns={"cluster_id":"clust"})
    greedy_df.to_csv("withClust_all_greedy_result.csv")
    return greedy_df

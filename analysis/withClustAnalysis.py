import json
import pandas as pd
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
def compile_cluster_MV_prj_into_csv():
    # Cluster-based MV results 
    # check result by plotting: plt.plot(MV_df.groupby("num_workers")["MV_jaccard"].mean())
    MV = pd.read_csv("pixel_em/MV_PRJ_table.csv")
    MV["clust"]=-1
    MV_clust = pd.read_csv("pixel_em/withClust_MV_PRJ_table.csv",index_col=0)
    MV_clust["num_workers"] = MV_clust["sample"].apply(lambda x: int(x.split("workers")[0]))
    MV_clust["sample_num"] = MV_clust["sample"].apply(lambda x: int(x.split("rand")[-1]))
    MV_clust.to_csv("pixel_em/withClust_MV_PRJ_table.csv")
    MV_clust = MV_clust.drop("sample",axis=1)
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
    return best_clust_df 

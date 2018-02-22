from sample_worker_seeds import *
from utils import * 
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
def compute_best_average_heuristics_workers_baselines(rerun_existing=False):
    outfile = "pixel_em/individual_worker_performance.csv"
    if os.path.exists(outfile) and not rerun_existing:
        return pd.read_csv(outfile)
    metric_keys = ['Precision [Self]', u'Recall [Self]','Jaccard [Self]','FPR% [Self]','FNR% [Self]']
    computed_wrt_gt = pd.read_csv("../data/computed_my_COCO_BBvals.csv",index_col=0)
    sample_lst = sample_specs.keys()
    obj_clusters = clusters()
    df_data =[]
    for batch in sample_lst:
        for objid in object_lst:
            if str(objid) in obj_clusters[batch]:
                clusts = ["-1"] + [obj_clusters[batch][str(objid)]]
            else:
                clusts = ["-1"]
            for clust in clusts:
                hydir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, batch, objid)
                if clust=="-1":
                    worker_ids = json.load(open(hydir+"worker_ids.json"))
                else:
                    worker_ids= json.load(open(hydir+"/clust{}/worker_ids.json".format(clust)))

                selected_annotations = computed_wrt_gt[(computed_wrt_gt["object_id"]==objid)&(computed_wrt_gt["worker_id"].isin(worker_ids))]
                best_num_pt_p,best_num_pt_r,best_num_pt_j,best_num_pt_fnr,best_num_pt_fpr = selected_annotations.loc[selected_annotations["Num Points"].idxmax()][metric_keys]
                best_ar_p,best_ar_r,best_ar_j,best_ar_fnr,best_ar_fpr = selected_annotations.loc[selected_annotations["Area Ratio"].idxmax()][metric_keys]
                max_p,max_r,max_j,max_fpr,max_fnr = selected_annotations.loc[selected_annotations["Jaccard [Self]"].idxmax()][metric_keys]
                avrg_p,avrg_r,avrg_j,avrg_fpr,avrg_fnr =  selected_annotations.mean()[metric_keys]
                df_data.append([batch, objid,clust, 
                                best_num_pt_p,best_num_pt_r,best_num_pt_j,best_num_pt_fnr,best_num_pt_fpr,\
                                 best_ar_p,best_ar_r,best_ar_j,best_ar_fnr,best_ar_fpr,\
                                avrg_p,avrg_r,avrg_j,avrg_fpr,avrg_fnr,max_p,\
                                max_r,max_j,max_fpr,max_fnr])
    df = pd.DataFrame(df_data,columns=["sample","objid","clust",
                    "P [NumPt]","R [NumPt]","J [NumPt]","FNR% [NumPt]","FPR% [NumPt]",
                    "P [AreaRatio]","R [AreaRatio]","J [AreaRatio]","FNR% [AreaRatio]","FPR% [AreaRatio]",
                    "P [AvrgWorker]","R [AvrgWorker]","J [AvrgWorker]","FNR% [AvrgWorker]","FPR% [AvrgWorker]",
                    "P [BestWorker]","R [BestWorker]","J [BestWorker]","FNR% [BestWorker]","FPR% [BestWorker]"])
    df["num_workers"]=df["sample"].apply(lambda x: int(x.split("workers")[0]))
    df.to_csv(outfile,index=None)
    return df
if __name__ =="__main__":
    df = compute_best_average_heuristics_workers_baselines()

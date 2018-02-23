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
def compute_worker_qualities_against_real_performance():
    import pickle as pkl
    from tqdm import tqdm
    sample_lst = sample_specs.keys()
    clust_df = pd.read_csv("spectral_clustering_all_hard_obj.csv")
    df =pd.read_csv("../data/computed_my_COCO_BBvals.csv",index_col=0)
    df_data = []
    for sample_name in tqdm(sample_lst):
        for objid in object_lst:
            cluster_ids = clust_df[(clust_df["objid"] == objid)].cluster.unique()
            for cluster_id in ["-1"] + list(cluster_ids):
                worker_ids = np.array(clust_df[(clust_df["objid"] == objid) & (clust_df["cluster"] == int(cluster_id))].wid)
                if len(worker_ids) > 1 or cluster_id == "-1":
                    if cluster_id!="" and cluster_id!="-1"  :
                        outdir = '{}{}/obj{}/clust{}/'.format(PIXEL_EM_DIR, sample_name, objid,cluster_id)
                    else:
                        outdir = '{}{}/obj{}/'.format(PIXEL_EM_DIR, sample_name, objid)
                    # worker qualities
                    qj,qp,qn,qp1,qn1,qp2,qn2 = None,None,None,None,None,None,None
                    #iso cases
                    iqj,iqp,iqn,iqp1,iqn1,iqp2,iqn2 = None,None,None,None,None,None,None
                    try:
                        qj = pkl.load(open("{}basic_q_best_thresh.pkl".format(outdir)))
                        iqj = pkl.load(open("{}isobasic_q_best_thresh.pkl".format(outdir)))
                        qp = pkl.load(open("{}GT_qp_best_thresh.pkl".format(outdir)))
                        qn = pkl.load(open("{}GT_qn_best_thresh.pkl".format(outdir)))
                        iqp = pkl.load(open("{}isoGT_qp_best_thresh.pkl".format(outdir)))
                        iqn = pkl.load(open("{}isoGT_qn_best_thresh.pkl".format(outdir)))
                        qp1 = pkl.load(open("{}GTLSA_qp1_best_thresh.pkl".format(outdir)))
                        qn1 = pkl.load(open("{}GTLSA_qn1_best_thresh.pkl".format(outdir)))
                        iqp1 = pkl.load(open("{}isoGTLSA_qp1_best_thresh.pkl".format(outdir)))
                        iqn1 = pkl.load(open("{}isoGTLSA_qn1_best_thresh.pkl".format(outdir)))
                        qp2 = pkl.load(open("{}GTLSA_qp2_best_thresh.pkl".format(outdir)))
                        qn2 = pkl.load(open("{}GTLSA_qn2_best_thresh.pkl".format(outdir)))
                        iqp2 = pkl.load(open("{}isoGTLSA_qp2_best_thresh.pkl".format(outdir)))
                        iqn2 = pkl.load(open("{}isoGTLSA_qn2_best_thresh.pkl".format(outdir)))
                    except(IOError):
                        print "can not find:",outdir
                    for wid in qj.keys():
                        q = qj[wid],qp[wid],qn[wid],qp1[wid],qn1[wid],qp2[wid],qn2[wid],iqj[wid],iqp[wid],iqn[wid],iqp1[wid],iqn1[wid],iqp2[wid],iqn2[wid]
                        if len(df[(df["object_id"]==objid)&(df["worker_id"]==wid)])==0:
                            print "can not find in df:",objid,"; worker:",wid
                        else:
                            worker_performance = df[(df["object_id"]==objid)&(df["worker_id"]==wid)][["Jaccard [Self]","Precision [Self]","Recall [Self]","FPR% [Self]","FNR% [Self]"]].values[0]
                        data = [sample_name,objid,cluster_id,wid]
                        data.extend(q) #worker qualities
                        data.extend(worker_performance) #actual worker performance against GT
                        df_data.append(data)
    df = pd.DataFrame(df_data,columns=["sample","objid","clust","wid",\
           "qj","qp","qn","qp1","qn1","qp2","qn2","iqj","iqp","iqn","iqp1","iqn1","iqp2","iqn2",\
           "Jaccard [Self]","Precision [Self]","Recall [Self]","FPR% [Self]","FNR% [Self]"])
    df.to_csv("EM_worker_qualities_against_real_performance.csv")
    return df
if __name__ =="__main__":
    df = compute_best_average_heuristics_workers_baselines()
    qj_df = compute_worker_qualities_against_real_performance()

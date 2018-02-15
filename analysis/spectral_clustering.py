import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocessing import * 
from sklearn import cluster
import sklearn

def compute_jaccard_affinity_matrix(object_id,exclude_lst=[]):
    bb_objects = bb_info[bb_info["object_id"]==object_id]
    worker_lst =  bb_objects.worker_id.unique()
    worker_lst = [w for w in worker_lst if w not in exclude_lst]
    prj_matrix = []
    for idx in worker_lst:
        prj_row =[]
        for jdx in worker_lst:
            #if idx!=jdx :
                worker_BB_polygon = worker_polygon(bb_objects,idx)
                worker_BB_polygon2 = worker_polygon(bb_objects,jdx)
                prj = BB_PRJ(worker_BB_polygon,worker_BB_polygon2)
                prj_row.append(prj[2])
        prj_matrix.append(prj_row)
    prj_matrix = np.array(prj_matrix)
    worker_lst=np.array(worker_lst)
    return prj_matrix,worker_lst

def run_spectral_clustering(obj,N,PLOT=True):
    spectral = cluster.SpectralClustering(
            n_clusters=N, eigen_solver='arpack',
            affinity="precomputed")
    aff_mat,worker_lst = compute_jaccard_affinity_matrix(obj)
    labels = spectral.fit_predict(aff_mat)
    obj_worker_cluster =[]
    if PLOT: 
        bb_objects = bb_info[bb_info["object_id"]==obj]
        plt.figure()
        plt.title("Obj {}".format(obj))
#         colors = ["blue","red","green","magenta","orange"]
#         for i,ylabel in enumerate(list(set(labels))):
        cmap = plt.cm.rainbow
        for i,ylabel in enumerate(list(set(labels))):
            c = cmap(i / float(len(list(set(labels)))))
            workers_in_cluster = np.where(labels==ylabel)[0]
            for widx in workers_in_cluster:
                plot_coords(worker_polygon(bb_objects,worker_lst[widx]),reverse_xy=True,color=c,fill_color="")
        plot_coords(ground_truth_T(obj),color="black",fill_color="",reverse_xy=True,lw=3,linestyle='--',invert_y=True)
        if not os.path.exists("cluster_img"):
            os.makedirs("cluster_img")
        plt.savefig("cluster_img/{}.png".format(obj))
    for i,ylabel in enumerate(list(set(labels))):
        workers_in_cluster = np.where(labels==ylabel)[0]
        for widx in workers_in_cluster:
            obj_worker_cluster.append([obj,worker_lst[widx],ylabel])
    return obj_worker_cluster

if __name__ == '__main__':
    objN_lst = [(1,2),(4,2),(7,2),(8,3),(10,2),(20,5),(15,2),(18,2),(21,2),(22,2),(25,2),(26,2),(27,4),(28,2),(29,3),(30,2),(31,3),(32,2),(33,2),(34,2),(35,2),(37,2),(40,2),(42,2),(47,3)]
    objN_lst = [(1,2)]
    obj_worker_clusters =[]
    for objN in objN_lst:
        print "working on obj {} for {} clusters".format(objN[0],objN[1])
        obj_worker_cluster = run_spectral_clustering(objN[0],objN[1])
        obj_worker_clusters.extend(obj_worker_cluster)
    df = pd.DataFrame(obj_worker_clusters,columns=["objid","wid","cluster"])
    df.to_csv("spectral_clustering_all_hard_obj.csv",index=None)

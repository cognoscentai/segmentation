# Clustering based on PRJ to eliminate type 1 and 2 errors
import pickle as pkl
from analysis_toolbox import *
import shapely
from collections import Counter
# Task difficulty
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
type_one_task_ambiguity = [15,20,22,27,31,40,41,42,47] #wrong object annotation
type_two_task_ambiguity = [1,4,7,8,9,10,18,20,21,25,28,29,30,31,32,33,34,35] #missing region

all_hard_tasks = list(set(np.concatenate((type_one_task_ambiguity,type_two_task_ambiguity))))
easy_tasks = [objid  for objid in object_lst if objid not in all_hard_tasks]
img_info,object_tbl,bb_info,hit_info=load_info()
def worker_polygon(bb_objects,worker_idx):
    bb = bb_objects[bb_objects["worker_id"]==worker_idx]
    xloc,yloc =  process_raw_locs([bb["x_locs"].iloc[0],bb["y_locs"].iloc[0]])    
    return Polygon(zip(xloc,yloc))
def compute_prjs(object_id,exclude_lst=[]):
    #PDF('../bb_object_pdfs/bb_object_{}.pdf'.format(object_id),size=(500,300))
    bb_objects = bb_info[bb_info["object_id"]==object_id]
    worker_lst =  bb_objects.worker_id.unique()
    worker_lst = [w for w in worker_lst if w not in exclude_lst]
    prj_matrix = []
    worker_ijdxs=[]
    for idx in worker_lst:
    #    prj_row =[]
        for jdx in worker_lst:
            if idx!=jdx :
                worker_BB_polygon = worker_polygon(bb_objects,idx)
                worker_BB_polygon2 = worker_polygon(bb_objects,jdx)
                prj = BB_PRJ(worker_BB_polygon,worker_BB_polygon2)
         #       prj_row.append(prj)
                worker_ijdxs.append([idx,jdx])
                prj_matrix.append(prj)
    prj_matrix = np.array(prj_matrix)
    worker_ijdxs = np.array(worker_ijdxs)
    return prj_matrix,worker_ijdxs
from sklearn.cluster import KMeans
import numpy as np
def cluster(object_id,prj_matrix,PLOT=False):
    kmeans= KMeans(n_clusters=4, random_state=0)
    ypred = kmeans.fit_predict(prj_matrix)
    if PLOT==True:
    	plt.figure()
    	plt.scatter(prj_matrix[:,0],prj_matrix[:,1],c=ypred)
    	plt.title("Obj {}".format(object_id),fontsize=14)
    	plt.xlabel("Precision",fontsize=13)
    	plt.ylabel("Recall",fontsize=13)
    return kmeans,ypred
def BB_PRJ(a,b):
    # Compute the PRJ values 
    intersection_area = a.intersection(b).area
    union_area = a.union(b).area
    precision = intersection_area/float(a.area)
    recall = intersection_area/float(b.area)
    jaccard = intersection_area/float(union_area)
    return precision,recall,jaccard
# For displaying object on Jupyter notebook
class PDF(object):
  def __init__(self, pdf, size=(200,200)):
    self.pdf = pdf
    self.size = size
  def _repr_html_(self):
    return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)
  def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)
def ground_truth_T(object_id,reverse_xy = False):
    my_BBG  = pd.read_csv("../data/my_ground_truth.csv")
    ground_truth_match = my_BBG[my_BBG.object_id==object_id]
    if reverse_xy:
        x_locs,y_locs =  process_raw_locs([ground_truth_match["y_locs"].iloc[0],ground_truth_match["x_locs"].iloc[0]])
    else:
        x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    T = Polygon(zip(x_locs,y_locs))
    return T
def plot_coords(obj, color='red', reverse_xy=False, linestyle='-',lw=1, fill_color="red", hatch='', show=False, invert_y=False):
    #Plot shapely polygon coord
    if type(obj) != shapely.geometry.MultiPolygon and type(obj) != list:
        obj = [obj]

    for ob in obj:
        if ob.exterior is None:
            print 'Plotting bug: exterior is None (potentially a 0 area tile). Ignoring and continuing...'
            continue
        if reverse_xy:
            x, y = ob.exterior.xy
        else:
            y, x = ob.exterior.xy
        plt.plot(x, y, linestyle, linewidth=lw, color=color, zorder=1)
        if fill_color != "":
            plt.fill_between(x, y, facecolor=fill_color, hatch=hatch, linewidth=lw, alpha=0.5)
    if invert_y:
        plt.gca().invert_yaxis()
if __name__=="__main__":
    bad_worker_records =[]
    for obj in object_lst:
        type1_error_flag=False
        # First pass, get rid of type 1 error
        prj_matrix,worker_ijdxs = compute_prjs(obj)
        kmeans,ypred = cluster(obj,prj_matrix)
        bb_objects = bb_info[bb_info["object_id"]==obj]
        type1_error_cluster = np.where(kmeans.cluster_centers_[:,2]<0.1)[0]
        if len(type1_error_cluster)==0:
            print "obj",obj,": Not Type 1 error"
        else:
	    type1_error_flag=True
            type1_error_wids = np.where(ypred==type1_error_cluster[0])[0]
            wids_counts_dict = Counter(worker_ijdxs[type1_error_wids].flatten())
            mode = scipy.stats.mode(wids_counts_dict.values()).mode[0]
            bad_idx = np.where(wids_counts_dict.values()!=mode)[0]
            bad_widx1 = np.array(wids_counts_dict.keys())[bad_idx]
            #plt.figure()
            #plt.title("Obj {}".format(obj),fontsize=13)
            #for widx in bad_widx1:
            #    plot_coords(ground_truth_T(obj),color="blue",fill_color="blue",reverse_xy=True)
            #    plot_coords(worker_polygon(bb_objects,widx),reverse_xy=True)
            for bwidx in bad_widx1:
                bad_worker_records.append([obj,bwidx,1]) #obj,bad worker id, error type 1

        # Second pass, get rid of type 2 error
	if type1_error_flag:
            prj_matrix,worker_ijdxs = compute_prjs(obj,exclude_lst=bad_widx1)
	else: 
	    prj_matrix,worker_ijdxs = compute_prjs(obj)
        kmeans,ypred = cluster(obj,prj_matrix)
        bb_objects = bb_info[bb_info["object_id"]==obj]
        i = kmeans.cluster_centers_[:,0].argmin() #lowest precision cluster
        j = kmeans.cluster_centers_[:,1].argmin() #highest precision cluster

        idx = np.where(ypred==i)[0]
        jdx = np.where(ypred==j)[0]
        #plt.figure()
        #plt.scatter(prj_matrix[idx,0],prj_matrix[idx,1])
        #plt.scatter(prj_matrix[jdx,0],prj_matrix[jdx,1])

        all_bad_worker_pairs = np.concatenate([worker_ijdxs[idx],worker_ijdxs[jdx]])
        wids_counts_dict = Counter(all_bad_worker_pairs.flatten())
        # Type 2 object errors where only a few workers make mistake are often in the form of high number of modes with a few mistaken worker having very high count
        count_of_voted_pairs = Counter(wids_counts_dict.values()) 

        total_datapoints_in_bad_clusters = len(all_bad_worker_pairs)*2
	# Two criterions to check for whether it is type 2 error or not
        flag1 = len(count_of_voted_pairs)<4 #number of distinct values is less than 4
        flag2= max(count_of_voted_pairs.keys())> total_datapoints_in_bad_clusters*0.1 #the large counts must be larger than 10% of the total datapoints in the bad clusters

        if flag1 and flag2:
            mode = scipy.stats.mode(wids_counts_dict.values()).mode[0]
            bad_idx = np.where(wids_counts_dict.values()!=mode)[0]
            bad_widx2 = np.array(wids_counts_dict.keys())[bad_idx]
            #plt.figure()
            #plt.title("Obj {}".format(obj),fontsize=13)
	    if type1_error_flag:
            	bad_widx = np.concatenate([bad_widx1,bad_widx2])
	    else:
		bad_widx = bad_widx2
            good_widx = [wid for wid in bb_objects.worker_id.unique() if wid not in bad_widx ]
    #         for widx in bad_widx:
    #             plot_coords(worker_polygon(bb_objects,widx),reverse_xy=True)
    #             plot_coords(ground_truth_T(obj),color="blue",fill_color="blue",reverse_xy=True)
    #        for widx in good_widx:
    #            plot_coords(worker_polygon(bb_objects,widx),reverse_xy=True)
            for bwidx in bad_widx2:
                bad_worker_records.append([obj,bwidx,2]) #obj,bad worker id, error type 1
        else:
            print "obj",obj,": Not Type 2 error"
            print flag1,flag2
    df = pd.DataFrame(bad_worker_records,columns=["objid","bad worker id","error type"])
    df.to_csv("bad_worker_records.csv")

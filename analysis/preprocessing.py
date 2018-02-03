# Clustering based on PRJ to eliminate type 1 and 2 errors
import pickle as pkl
from analysis_toolbox import *
import shapely
from collections import Counter
img_info,object_tbl,bb_info,hit_info=load_info()
def worker_polygon(bb_objects,worker_idx):
    bb = bb_objects[bb_objects["worker_id"]==worker_idx]
    xloc,yloc =  process_raw_locs([bb["x_locs"].iloc[0],bb["y_locs"].iloc[0]])    
    return Polygon(zip(xloc,yloc))
def compute_prjs(object_id,exclude_lst=[]):
    PDF('../bb_object_pdfs/bb_object_{}.pdf'.format(object_id),size=(500,300))
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
def cluster_and_plot(object_id,prj_matrix):
    kmeans= KMeans(n_clusters=4, random_state=0)
    ypred = kmeans.fit_predict(prj_matrix)
    #print kmeans.cluster_centers_
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
def plot_coords(obj, color='red', reverse_xy=False, linestyle='-',lw=0, fill_color="red", hatch='', show=False, invert_y=False):
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


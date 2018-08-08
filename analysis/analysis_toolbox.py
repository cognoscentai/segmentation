import os 
import csv
import sqlite3
from glob import glob
from os.path import expanduser
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from qualityBaseline import *
import scipy
def save_db_as_csv(db="crowd-segment",connect=True,postgres=True):
	'''
	Create CSV file of each table from app.db
	db = "segment" (local) ,"crowd-segment" (heroku remote)
	'''
	path = "/Users/dorislee/Desktop/Research/seg/data/"
	table_names = ["bounding_box","image","object","object_location","worker","hit"]
	for table_name in table_names :
		if postgres:
			if db=="crowd-segment" and connect==True:
				# Connect onto the DB on Heroku 
				os.system("bash herokuDBupdate.sh")
			os.system("psql {2}  -F , --no-align  -c  'SELECT * FROM {0}' > {1}/{0}.csv".format(table_name,path,db))
		else:
			# sqlite
			conn = sqlite3.connect(glob(expanduser('../web-app/app.db'))[0])
			cursor = conn.cursor()
			cursor.execute("select * from {};".format(table_name))
			with open("{}.csv".format(table_name), "wb") as csv_file:
				csv_writer = csv.writer(csv_file)
				csv_writer.writerow([i[0] for i in cursor.description]) # write headers
				csv_writer.writerows(cursor)
def COCO_convert_png_to_jpg():
	#Convert .jpg to .png
	os.chdir("app/static")
	for fname in glob.glob("COCO*"):
		os.system("convert {0} {1}".format(fname, fname.split(".")[0]+".png"))
from config import DATA_DIR 
def load_info(eliminate_self_intersection_bb=True):
    from shapely.validation import explain_validity
    old_path = os.getcwd()
    os.chdir(DATA_DIR)
    img_info = pd.read_csv("image.csv")
    object_info = pd.read_csv("object.csv")
    object_location = pd.read_csv("object_location.csv")
    object_tbl = object_info.merge(object_location,how="inner",left_on="id",right_on="object_id")
    bb_info = pd.read_csv("bounding_box.csv")
    bb_info=bb_info[bb_info["worker_id"]>3]#do not include BB<=3 since they are drawn for testing and wid=3 is ground truth
    if eliminate_self_intersection_bb:
        for bb in bb_info.iterrows():
            bb=bb[1]
            xloc,yloc =  process_raw_locs([bb["x_locs"],bb["y_locs"]]) 
            worker_BB_polygon=Polygon(zip(xloc,yloc))
            if explain_validity(worker_BB_polygon).split("[")[0]=='Self-intersection':
                bb_info.drop(bb.name, inplace=True)
    hit_info = pd.read_csv("hit.csv",skipfooter=1)
    os.chdir(old_path)
    return [img_info,object_tbl,bb_info,hit_info]

import matplotlib.image as mpimg
def visualize_bb_objects(object_id,img_bkgrnd=True,worker_id=-1,gtypes=['worker','self'],single=False,bb_info="",obj_pointer=False):
    '''
    Plot BB for the object corresponding to the given object_id
    #Still need to implement COCO later...
    gtypes: list specifying the types of BB to be plotted (worker=all worker's annotation, 'self'=self BBG)
    '''
    if not single:
        img_info,object_tbl,bb_info,hit_info=load_info()
    else:
        img_info,object_tbl,bb_info_bad,hit_info=load_info()
    plt.figure(figsize =(7,7))
    ground_truth = pd.read_csv("../data/object_ground_truth.csv")
    my_BBG  = pd.read_csv("../data/my_ground_truth.csv")
    if img_bkgrnd:
        img_name = img_info[img_info.id==int(object_tbl[object_tbl.id==object_id]["image_id"])]["filename"].iloc[0]
        fname = "../web-app/app/static/"+img_name+".png"
        img=mpimg.imread(fname)
        width,height = get_size(fname)
        img_id = int(img_name.split('_')[-1])
        plt.imshow(img)
        plt.xlim(0,width)
        plt.ylim(height,0)
        plt.axis("off")   
    else:
        plt.gca().invert_yaxis()
    plt.title("Object {0} [{1}]".format(object_id,object_tbl[object_tbl.object_id==object_id]["name"].iloc[0]))
#         plt.fill_between(x_locs,y_locs,color='none',facecolor='#f442df', alpha=0.5)
    if 'worker' in gtypes:
        bb_objects = bb_info[bb_info["object_id"]==object_id]
        if worker_id!=-1:
            bb = bb_objects[bb_objects["worker_id"]==worker_id]
            xloc,yloc =  process_raw_locs([bb["x_locs"].iloc[0],bb["y_locs"].iloc[0]])    
        
            plt.plot(xloc,yloc,'-',color='cyan',linewidth=3)
            plt.fill_between(xloc,yloc,color='none',facecolor='#f442df', alpha=0.01)
        else:
            for x,y in zip(bb_objects["x_locs"],bb_objects["y_locs"]):
                xloc,yloc = process_raw_locs([x,y])
                if single:
                    plt.plot(xloc,yloc,'-',color='#f442df',linewidth=4)
                else:
                    plt.plot(xloc,yloc,'-',color='#f442df',linewidth=1)
                    plt.fill_between(xloc,yloc,color='none',facecolor='#f442df', alpha=0.01)
    if 'self' in gtypes:
        ground_truth_match = my_BBG[my_BBG.object_id==object_id]
        x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
        if single:
            plt.plot(x_locs,y_locs,'--',color='#0000ff',linewidth=2)
        else: 
            plt.plot(x_locs,y_locs,'-',color='#0000ff',linewidth=4)
    # elif gtype=='COCO':
    #     ground_truth_match = my_BBG[my_BBG.object_id==object_id]
    if obj_pointer:
        objloc = pd.read_csv("../data/object_location.csv")
	xloc = objloc[objloc["object_id"]==object_id]["x_loc"].values[0]
	yloc = objloc[objloc["object_id"]==object_id]["y_loc"].values[0]
        plt.plot(xloc,yloc,'s',ms=8,color='red')
        plt.plot(xloc,yloc,'s',ms=4,color='white')
    if not single:plt.savefig("bb_object_{}.pdf".format(object_id))
def visualize_bb_worker(worker_id,gtypes=['worker','self']):
    '''
    Plot BB for the object corresponding to the given object_id
    #Still need to implement COCO later...
    gtypes: list specifying the types of BB to be plotted (worker=all worker's annotation, 'self'=self BBG)
    '''
    img_info,object_tbl,bb_info,hit_info=load_info()
    ground_truth = pd.read_csv("../data/object_ground_truth.csv")
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    filtered_bb_info=bb_info[bb_info["worker_id"]==worker_id]
    for object_id in list(filtered_bb_info.object_id):
        visualize_bb_objects(object_id,single=True,gtypes=gtypes,bb_info=filtered_bb_info)
    plt.savefig("bb_worker_{0}_object_{1}.pdf".format(worker_id,object_id))
def visualize_all_ground_truth_bb():
    '''
	Plot all Ground truth bounding box drawn by me
	'''
    ground_truth = pd.read_csv("../../data/object_ground_truth.csv")
    worker_info = pd.read_csv("../../data/worker.csv",skipfooter=1)
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    for i in np.arange(len(img_info)):
        img_name = img_info["filename"][i]
        if 'COCO' in img_name:
            fname = "../web-app/app/static/"+img_name+".png"
            img=mpimg.imread(fname)
            width,height = get_size(fname)
            img_id = int(img_name.split('_')[-1])
            plt.figure(figsize =(10,10))
            plt.imshow(img)
            plt.axis("off")

            filtered_object_tbl = object_tbl[object_tbl["image_id"]==i+1]

            #for oid,bbx_path,bby_path in zip(bb_info["object_id"],bb_info["x_locs"],bb_info["y_locs"]):
            for bb in bb_info.iterrows():
                oid = bb[1]["object_id"]
                bbx_path= bb[1]["x_locs"]
                bby_path= bb[1]["y_locs"]
                if int(object_tbl[object_tbl.object_id==oid].image_id) ==i+1:
    #                 worker_x_locs,worker_y_locs= process_raw_locs([bbx_path,bby_path])
                    ground_truth_match = my_BBG[my_BBG.object_id==oid]
                    x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
                    plt.plot(x_locs,y_locs,'-',color='#f442df',linewidth=0.5)
                    plt.fill_between(x_locs,y_locs,color='none',facecolor='#f442df', alpha=0.01)

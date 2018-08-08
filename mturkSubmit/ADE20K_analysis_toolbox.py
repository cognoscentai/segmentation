import os 
import csv
import sqlite3
from glob import glob
from os.path import expanduser
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import ast
DATA_DIR = "../web-app/ADE20K_data_info/"
def load_ADE20K_info():
    img_info = pd.read_csv(DATA_DIR+"image.csv")
    object_info = pd.read_csv(DATA_DIR+"object.csv")
    object_location = pd.read_csv(DATA_DIR+"object_location.csv")
    object_tbl = object_info.merge(object_location,how="inner",left_on="id",right_on="object_id")
    BB_count_info = pd.read_csv(DATA_DIR+"BB_count_tbl.csv")
    return [img_info,object_tbl,BB_count_info]
def get_size(fname):
    from PIL import Image
    #Open image for computing width and height of image 
    im = Image.open(fname)
    width = im.size[0]
    height = im.size[1]
    return width, height

def visualize_bb_objects(object_id,img_bkgrnd=True,worker_id=-1,gtypes=['worker','self'],single=False,bb_info="",obj_pointer=False):
    '''
    Plot BB for the object corresponding to the given object_id
    #Still need to implement COCO later...
    gtypes: list specifying the types of BB to be plotted (worker=all worker's annotation, 'self'=self BBG)
    '''
    img_info, object_tbl, BB_count_info = load_ADE20K_info()
    bb_objects = pd.read_csv("bounding_box.csv")
    bb_info = bb_objects[bb_objects["worker_id"]>3]

    plt.figure(figsize =(7,7))
    # ground_truth = pd.read_csv("../data/object_ground_truth.csv")
    # my_BBG  = pd.read_csv("../data/my_ground_truth.csv")
    if img_bkgrnd:
        img_name = img_info[img_info.id==int(object_tbl[object_tbl.id==object_id]["image_id"])]["filename"].iloc[0]
        fname = "../web-app/app/static/"+img_name+".png"
        img=mpimg.imread(fname,0)
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
    # if 'self' in gtypes:
    #     ground_truth_match = my_BBG[my_BBG.object_id==object_id]
    #     x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    #     if single:
    #         plt.plot(x_locs,y_locs,'--',color='#0000ff',linewidth=2)
    #     else: 
    #         plt.plot(x_locs,y_locs,'-',color='#0000ff',linewidth=4)
    
    if obj_pointer:
        objloc = pd.read_csv(DATA_DIR+"object_location.csv")
	xloc = objloc[objloc["object_id"]==object_id]["x_loc"].values[0]
	yloc = objloc[objloc["object_id"]==object_id]["y_loc"].values[0]
        plt.plot(xloc,yloc,'s',ms=8,color='red')
        plt.plot(xloc,yloc,'s',ms=4,color='white')
    plt.savefig("bb_object_{}.pdf".format(object_id))
def process_raw_locs(segmentation):
    '''
    Given a raw string of x and y coordinates, process it
    return a list of x_locs and y_locs
    '''
    x_locs=[]
    y_locs=[]
    bbx_path,bby_path = segmentation
    x_locs = ast.literal_eval(bbx_path[1:-1])
    y_locs = ast.literal_eval(bby_path[1:-1]) 
    # Append the starting point again in the end to close the BB
    if len(x_locs)>0:
        x_locs.append(x_locs[0])
        y_locs.append(y_locs[0])
    return x_locs,y_locs
